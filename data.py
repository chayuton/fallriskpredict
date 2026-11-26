"""
Data loading, causal GRU windows and hip-descent velocity.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config as C

# MediaPipe Pose landmark indices used here
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24


def lm(idx: int, axis: str) -> str:
    """Column name of one landmark coordinate, e.g. lm(23, 'y') -> 'lm23_y'."""
    return C.LANDMARK_COL_FMT.format(idx=idx, axis=axis)


def read_table(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_frames(path) -> pd.DataFrame:
    """Load the frame-level table, sort it and add a timestamp column if missing."""
    df = read_table(path)
    missing = [c for c in (C.SUBJECT_COL, C.VIDEO_COL, C.FRAME_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    df[C.SUBJECT_COL] = df[C.SUBJECT_COL].astype(str)
    df[C.VIDEO_COL] = df[C.VIDEO_COL].astype(str)
    df = df.sort_values([C.VIDEO_COL, C.FRAME_COL], kind="stable").reset_index(drop=True)
    if C.TIME_COL not in df.columns:
        df[C.TIME_COL] = df[C.FRAME_COL] / C.FPS
    n_subj = df.groupby(C.VIDEO_COL)[C.SUBJECT_COL].nunique()
    if (n_subj > 1).any():
        bad = n_subj[n_subj > 1].index.tolist()[:5]
        raise ValueError(f"Each video_id must belong to one subject; offending videos: {bad}")
    dup = df.duplicated([C.VIDEO_COL, C.FRAME_COL])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated (video_id, frame_idx) rows")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    if C.FEATURE_COLS:
        missing = [c for c in C.FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"FEATURE_COLS not found in data: {missing[:5]}")
        return list(C.FEATURE_COLS)
    meta = {C.SUBJECT_COL, C.VIDEO_COL, C.FRAME_COL, C.TIME_COL, C.LABEL_COL}
    cols = [
        c for c in df.columns
        if c.startswith(C.FEATURE_PREFIX) and c not in meta and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        raise ValueError(f"No feature columns starting with '{C.FEATURE_PREFIX}'")
    return cols


def encode_labels(series: pd.Series) -> np.ndarray:
    """Map labels (names or integer ids) to integer ids following CLASS_NAMES."""
    names = list(C.CLASS_NAMES)
    if pd.api.types.is_numeric_dtype(series):
        ids = series.to_numpy().astype(int)
        if ids.min() < 0 or ids.max() >= len(names):
            raise ValueError(f"Integer labels must be in 0..{len(names) - 1}")
        return ids
    mapping = {n: i for i, n in enumerate(names)}
    unknown = set(series.astype(str).unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Labels {sorted(unknown)} are not in CLASS_NAMES={names}")
    return series.astype(str).map(mapping).to_numpy().astype(int)


def positive_ids() -> list[int]:
    return [C.CLASS_NAMES.index(c) for c in C.POSITIVE_CLASSES]


def prob_columns() -> list[str]:
    return [f"prob_{c}" for c in C.CLASS_NAMES]


def make_windows(df: pd.DataFrame, feat_cols: list[str], with_labels: bool = True,
                 window: int = C.WINDOW, stride: int = C.STRIDE, label_mode: str = C.WINDOW_LABEL):
    """
    Build causal windows that never cross a video boundary.
    Window k ends at frame t (inclusive) and contains frames t-window+1 .. t, i.e. exactly
    what the online system has in its buffer at frame t.

    Returns X (n, window, n_feat) float32, y (n,) int or None, meta DataFrame
    (subject_id, video_id, frame_idx, timestamp of the window's LAST frame).
    """
    X_parts, y_parts, meta_parts = [], [], []
    y_all = encode_labels(df[C.LABEL_COL]) if with_labels else None
    for vid, idx in df.groupby(C.VIDEO_COL, sort=False).indices.items():
        g = df.iloc[idx]
        F = g[feat_cols].to_numpy(np.float32)
        n = len(F)
        if n < window:
            warnings.warn(f"video {vid}: {n} frames < window {window}; skipped")
            continue
        ends = np.arange(window - 1, n)[::stride]
        win = np.lib.stride_tricks.sliding_window_view(F, window, axis=0)  # (n-w+1, f, w)
        X_parts.append(np.ascontiguousarray(win[ends - (window - 1)].transpose(0, 2, 1)))
        if with_labels:
            L = y_all[idx]
            if label_mode == "last":
                y_parts.append(L[ends])
            elif label_mode == "majority":
                lw = np.lib.stride_tricks.sliding_window_view(L, window)[ends - (window - 1)]
                y_parts.append(np.array([np.bincount(r, minlength=len(C.CLASS_NAMES)).argmax() for r in lw]))
            else:
                raise ValueError("WINDOW_LABEL must be 'last' or 'majority'")
        meta_parts.append(pd.DataFrame({
            C.SUBJECT_COL: g[C.SUBJECT_COL].to_numpy()[ends],
            C.VIDEO_COL: vid,
            C.FRAME_COL: g[C.FRAME_COL].to_numpy()[ends],
            C.TIME_COL: g[C.TIME_COL].to_numpy()[ends],
        }))
    if not X_parts:
        raise ValueError("No windows could be built (videos shorter than WINDOW?)")
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts) if with_labels else None
    meta = pd.concat(meta_parts, ignore_index=True)
    return X, y, meta


# ----------------------------------------------------------------------------
# Hip descent velocity  (Eq. 3-4 of the paper)
# ----------------------------------------------------------------------------
def _mid(g: pd.DataFrame, a: int, b: int, axis: str) -> np.ndarray:
    return (g[lm(a, axis)].to_numpy(float) + g[lm(b, axis)].to_numpy(float)) / 2.0


def hip_center_y(g: pd.DataFrame) -> np.ndarray:
    """Normalized image y of the hip midpoint (0 = top, 1 = bottom), NaN if not visible."""
    y = _mid(g, L_HIP, R_HIP, "y")
    vis_cols = [lm(L_HIP, "v"), lm(R_HIP, "v")]
    if all(c in g.columns for c in vis_cols):
        vis = np.minimum(g[vis_cols[0]].to_numpy(float), g[vis_cols[1]].to_numpy(float))
        y = np.where(vis >= C.MIN_VISIBILITY, y, np.nan)
    return y


def frame_size(g: pd.DataFrame):
    """Image width/height in pixels per row: columns frame_w / frame_h if present
    (prepare_open_data.py writes them, so datasets with different resolutions can be mixed),
    otherwise FRAME_W / FRAME_H from config.py."""
    if "frame_w" in g.columns and "frame_h" in g.columns:
        return g["frame_w"].to_numpy(float), g["frame_h"].to_numpy(float)
    n = len(g)
    return np.full(n, float(C.FRAME_W)), np.full(n, float(C.FRAME_H))


def torso_length_px(g: pd.DataFrame) -> np.ndarray:
    """2-D distance between shoulder midpoint and hip midpoint, in pixels."""
    w, h = frame_size(g)
    dx = (_mid(g, L_SHOULDER, R_SHOULDER, "x") - _mid(g, L_HIP, R_HIP, "x")) * w
    dy = (_mid(g, L_SHOULDER, R_SHOULDER, "y") - _mid(g, L_HIP, R_HIP, "y")) * h
    return np.hypot(dx, dy)


def torso_reference_px(g: pd.DataFrame) -> np.ndarray:
    """
    Causal reference torso length: running median over TORSO_REF_WIN frames that end
    TORSO_REF_LAG frames before t (so a fall in progress does not shrink it).
    The first frames fall back to the expanding median up to t (still causal).
    """
    s = pd.Series(torso_length_px(g))
    ref = s.rolling(C.TORSO_REF_WIN, min_periods=1).median().shift(C.TORSO_REF_LAG)
    ref = ref.fillna(s.expanding(min_periods=1).median())
    return ref.to_numpy()


def hip_descent_velocity(g: pd.DataFrame, mode: str | None = None, lag: int | None = None) -> np.ndarray:
    """
    Downward hip velocity for ONE video (rows sorted by frame). Positive = moving down.
    Causal: v(t) only uses frames t-lag .. t.

    >>> EDIT  Make sure this reproduces Eq. (3)-(4) of the paper exactly
              (which landmarks, which lag, any smoothing, which scale factor).
    """
    mode = mode or C.VELOCITY_MODE
    lag = lag or C.VEL_LAG_FRAMES
    t = g[C.TIME_COL].to_numpy(float)
    y = hip_center_y(g)
    v = np.full(len(y), np.nan)
    if len(y) > lag:
        v[lag:] = (y[lag:] - y[:-lag]) / (t[lag:] - t[:-lag])   # image heights per second
    if mode == "meters":
        return v * C.FRAME_H_METERS
    _, h = frame_size(g)
    if mode == "pixels":
        return v * h
    if mode == "torso":
        ref = torso_reference_px(g)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ref > 1e-6, v * h / ref, np.nan)
    raise ValueError(f"Unknown VELOCITY_MODE '{mode}'")
