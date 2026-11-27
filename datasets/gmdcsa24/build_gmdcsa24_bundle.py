#!/usr/bin/env python3
"""
Build the GMDCSA24 upload bundle for the fallriskpredict web app (frames.csv, trials.csv, predictions.csv plus
landmarks, labels, impact-review images and reference results). Run from the repository root, in order:

  git clone https://github.com/ekramalam/gmdcsa24-a-dataset-for-human-fall-detection-in-videos data/raw/gmdcsa24
  python datasets/gmdcsa24/gmdcsa24_labels.py data/raw/gmdcsa24 data/gmdcsa24       # manifest + segments
  python prepare_open_data.py extract --dir data/gmdcsa24                            # MediaPipe Pose (slow)
  python datasets/gmdcsa24/build_gmdcsa24_bundle.py label    # 30 fps, impacts, split labels, trials, review images
  python datasets/gmdcsa24/build_gmdcsa24_bundle.py train    # reference GRU, leave-one-subject-out (4 folds)
  python datasets/gmdcsa24/build_gmdcsa24_bundle.py verify   # web-app analysis engine on the upload files
  python datasets/gmdcsa24/build_gmdcsa24_bundle.py package --videos data/raw/gmdcsa24   # folder + zip

Options: --dir (work folder, default data/gmdcsa24), --out (bundle folder, default
results/gmdcsa24_upload_bundle), --videos (dataset clone, for its LICENSE), --skip-prep (label: reuse
frames_30fps.csv and trials_suggested.csv), --seed (train).

Impact times: GMDCSA24 gives the start of each fall but not the impact. Impact is taken as the first contact of
hands, knees, hip or back with the floor (or bed), estimated from the pose (end of the first main descent of the
hip, trunk centre or lower knee) and corrected after looking at frame strips (MANUAL_IMPACT_S below). Check the
images in impact_check/ before reporting lead times.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TK = HERE.parents[1]                                   # repository root (config.py, prepare_open_data.py, ...)
D = TK / "data" / "gmdcsa24"                           # work folder; set by --dir
OUT = TK / "results" / "gmdcsa24_upload_bundle"        # bundle folder; set by --out
VIDEOS: Path | None = None                             # dataset clone; set by --videos
BUNDLE_DOCS = [HERE / "README_TH.md", HERE / "GET_MORE_DATA_TH.md", TK / "get_public_data.py"]
sys.path.insert(0, str(TK))
import config as C  # noqa: E402

FPS = 30
LABEL_MAP = {"fall": "loss_of_balance", "fallen": "fallen"}
APP_LM = [f"lm{i}_{a}" for i in (11, 12, 23, 24) for a in ("x", "y", "v")]
APP_COLS = ["subject_id", "video_id", "dataset", "frame_idx", "timestamp", "label", "frame_w", "frame_h",
            "detected"] + APP_LM
TRIAL_COLS = ["trial_id", "video_id", "subject_id", "dataset", "trial_type", "scenario", "onset_frame",
              "impact_frame", "impact_source", "fall_end_frame"]
SETTINGS = {"velocity_mode": "torso", "v_thresh": 3.0, "p_thresh": 0.5, "persist": 3, "persist_on": "gru",
            "fps": FPS, "latency_ms": 0}
EXTEND_FALLEN_MAX_S = 1.0     # fall label ends <= this before the video ends -> "fallen" runs to the last frame
MIN_PART_DROP_TL = 0.3        # a body part must drop at least this (torso lengths) to count as landing
MIN_ONSET_S = 0.5             # falls starting earlier have < 15 frames of history -> left out of trials.csv
# Impact times (s) set after looking at the frame strips in impact_check/ (strip_*.jpg); still to be verified.
MANUAL_IMPACT_S: dict[str, tuple[float, str]] = {
    # first contact of hands, knees, hip or back with the floor/bed, read from frame strips (about +-0.1 s)
    "gmdcsa24/S1/Fall/01": (3.60, "hand reaches the floor at about 3.60 s, knees 3.70 s"),
    "gmdcsa24/S1/Fall/02": (2.05, "hand reaches the floor at about 2.05 s, knees 2.20 s"),
    "gmdcsa24/S1/Fall/04": (2.38, "crouches off the chair; hands and a knee on the floor by about 2.38 s"),
    "gmdcsa24/S1/Fall/05": (2.00, "drops to the knees at about 2.00 s"),
    "gmdcsa24/S1/Fall/06": (2.42, "drops to the knees at about 2.42 s"),
    "gmdcsa24/S1/Fall/08": (1.24, "drops to the knees at about 1.24 s"),
    "gmdcsa24/S1/Fall/15": (2.22, "hand reaches the floor at about 2.22 s"),
    "gmdcsa24/S1/Fall/16": (3.30, "hands reach the floor at about 3.30 s"),
    "gmdcsa24/S2/Fall/01": (2.25, "slow bend; hand reaches the floor at about 2.25 s, knees 2.60 s"),
    "gmdcsa24/S2/Fall/04": (4.18, "hands and hip reach the floor at about 4.18 s"),
    "gmdcsa24/S2/Fall/09": (2.35, "hands reach the floor at about 2.35 s"),
    "gmdcsa24/S2/Fall/10": (2.87, "hand on the floor at about 2.87 s, sitting at 3.13 s (15 fps video)"),
    "gmdcsa24/S2/Fall/11": (6.66, "rolls off the bed in a dark video (pose found in 22% of frames); on the floor at about 6.66 s"),
    "gmdcsa24/S2/Fall/13": (3.50, "lands sitting backward at about 3.50 s"),
    "gmdcsa24/S2/Fall/16": (1.05, "knees land at about 1.05 s"),
    "gmdcsa24/S2/Fall/19": (8.35, "drops to the knees at about 8.35 s"),
    "gmdcsa24/S2/Fall/20": (8.50, "already kneeling; hands reach the floor at about 8.50 s"),
    "gmdcsa24/S2/Fall/21": (6.80, "hand and hip reach the floor at about 6.80 s"),
    "gmdcsa24/S2/Fall/22": (1.47, "drops to the knees at about 1.47 s"),
    "gmdcsa24/S3/Fall/10": (1.80, "falls onto the bed at about 1.80 s"),
    "gmdcsa24/S3/Fall/11": (5.03, "drops to the knees at about 5.03 s"),
    "gmdcsa24/S3/Fall/14": (8.50, "lands sitting at about 8.50 s"),
    "gmdcsa24/S3/Fall/16": (2.05, "dark 15 fps video; hand reaches the floor at about 2.05 s"),
    "gmdcsa24/S4/Fall/01": (2.05, "drops to the knees on the bed at about 2.05 s"),
    "gmdcsa24/S4/Fall/02": (1.71, "rolls off the bed; hand reaches the floor at about 1.71 s, body 2.0 s"),
    "gmdcsa24/S4/Fall/03": (2.12, "person close to the camera; hands reach the floor at about 2.12 s"),
    "gmdcsa24/S4/Fall/04": (1.30, "dives forward; hands reach the floor at about 1.30 s"),
    "gmdcsa24/S4/Fall/07": (2.48, "knee reaches the floor at about 2.48 s"),
    "gmdcsa24/S4/Fall/10": (2.45, "hands reach the floor at about 2.45 s"),
    "gmdcsa24/S4/Fall/11": (1.50, "dives forward; hands reach the floor at about 1.50 s"),
    "gmdcsa24/S4/Fall/12": (1.88, "hands reach the floor at about 1.88 s"),
}
# Every fall left in trials.csv was looked at in frame strips; these were confirmed as estimated (+-0.1 s).
VISUALLY_CONFIRMED = True
# Falls kept for training but left out of trials.csv because the impact itself is ambiguous.
MANUAL_EXCLUDE = {
    "gmdcsa24/S4/Fall/06": "slow 2.5 s slide off the bed with hands on the floor: no single impact",
    "gmdcsa24/S4/Fall/14": "slow slide off the bed with hands on the floor: no single impact",
    "gmdcsa24/S2/Fall/15": "slow slide off the bed with hands on the floor: no single impact",
}


def run(*cmd):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=TK)


# ----------------------------------------------------------------------------------------------- label
def hip_torso(g: pd.DataFrame):
    """hip height (normalized y, larger = lower) and torso length in pixels, per frame."""
    w, h = g["frame_w"].to_numpy(float), g["frame_h"].to_numpy(float)
    hy = (g["lm23_y"] + g["lm24_y"]).to_numpy(float) / 2
    hx = (g["lm23_x"] + g["lm24_x"]).to_numpy(float) / 2
    sy = (g["lm11_y"] + g["lm12_y"]).to_numpy(float) / 2
    sx = (g["lm11_x"] + g["lm12_x"]).to_numpy(float) / 2
    torso = np.hypot((sx - hx) * w, (sy - hy) * h)
    ok = g["detected"].to_numpy(bool)
    return np.where(ok, hy, np.nan), np.where(ok, torso, np.nan)


def shoulder_y(g: pd.DataFrame):
    sy = (g["lm11_y"] + g["lm12_y"]).to_numpy(float) / 2
    return np.where(g["detected"].to_numpy(bool), sy, np.nan)


def body_impact(t, y, smooth=3, frac=0.5, rel=0.5):
    """End of the FIRST main descent phase: the first local peak of downward velocity that reaches `rel` of the
    largest one, then the first frame at which the velocity has dropped below `frac` of that peak (the body is
    being stopped by the floor, knees, hands or a bed). With a single descent this equals
    lead_time.suggest_impact; in slow kneel-then-lie falls it marks the first landing instead of the last.
    Returns an index into t, or None."""
    y = pd.Series(y).interpolate(limit_direction="both")
    if len(y) < 3 or y.isna().all():
        return None
    y = y.rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    v = np.gradient(y, t)
    vmax = float(v.max())
    if vmax <= 0:
        return None
    p = int(np.argmax(v))
    for k in range(1, len(v) - 1):
        if v[k] >= rel * vmax and v[k] >= v[k - 1] and v[k] >= v[k + 1]:
            p = k
            break
    after = np.flatnonzero(v[p:] <= frac * v[p])
    return p + (int(after[0]) if len(after) else 0)


def lower_knee_y(g: pd.DataFrame):
    det = g["detected"].to_numpy(bool)
    ys = [np.where(det & (g[f"lm{i}_v"].to_numpy(float) >= 0.3), g[f"lm{i}_y"].to_numpy(float), np.nan) for i in (25, 26)]
    with np.errstate(all="ignore"):
        return np.nanmax(np.vstack(ys), axis=0)


def choose_impacts(trials, frames):
    """Impact = first landing of the body: the earliest end of a main descent among the hip centre, the trunk
    centre (hips + shoulders) and the lower knee, counting only parts that drop at least MIN_PART_DROP_TL
    torso lengths. Visual corrections (MANUAL_IMPACT_S) override the estimate."""
    by_vid = dict(tuple(frames.groupby("video_id")))
    out = trials.copy()
    for c in ("impact_hip_s", "impact_trunk_s", "impact_knee_s", "hip_drop_TL"):
        out[c] = np.nan
    out["exclude_reason"] = ""
    for i, r in out[out["trial_type"] == "fall"].iterrows():
        g = by_vid[r.video_id].sort_values("frame_idx")
        t, fidx = g["timestamp"].to_numpy(float), g["frame_idx"].to_numpy(int)
        hy, torso = hip_torso(g)
        on = int(np.searchsorted(fidx, int(r.onset_frame)))
        ref = torso[max(0, on - 30):on + 1]
        ref = np.nanmedian(ref) if np.isfinite(ref).any() else np.nanmedian(torso)
        to_tl = g["frame_h"].to_numpy(float)[on] / ref
        cand = {}
        for name, y in (("hip", hy), ("trunk", (hy + shoulder_y(g)) / 2), ("knee", lower_knee_y(g))):
            k = body_impact(t[on:], y[on:])
            if k is None:
                continue
            k += on
            with np.errstate(all="ignore"):
                drop = (np.nanmax(y[k:k + 4]) - np.nanmin(y[max(0, on - 3):on + 1])) * to_tl
            cand[name] = (k, drop)
            out.loc[i, f"impact_{name}_s"] = t[k]
        out.loc[i, "hip_drop_TL"] = cand["hip"][1] if "hip" in cand else np.nan
        ok = {n: kd for n, kd in cand.items() if kd[1] >= MIN_PART_DROP_TL}
        if ok:
            name = min(ok, key=lambda n: ok[n][0])
            k, src = ok[name][0], f"suggested_first_landing_{name} (VERIFY)"
        elif "trunk" in cand:
            k, src = cand["trunk"][0], "suggested_first_landing_trunk (VERIFY)"
        else:
            k, src = None, "none"
        if r.trial_id in MANUAL_IMPACT_S:
            t_man, note = MANUAL_IMPACT_S[r.trial_id]
            k = int(np.argmin(np.abs(t - t_man)))
            src = f"visual_check (VERIFY): {note}"
        elif VISUALLY_CONFIRMED and k is not None and r.trial_id not in MANUAL_EXCLUDE and t[on] >= MIN_ONSET_S:
            src = src.replace(" (VERIFY)", ", confirmed in frame strips (VERIFY)")
        out.loc[i, "impact_frame"] = int(fidx[k]) if k is not None else np.nan
        out.loc[i, "impact_source"] = src
        if r.trial_id in MANUAL_EXCLUDE:
            out.loc[i, "exclude_reason"] = MANUAL_EXCLUDE[r.trial_id]
        elif t[on] < MIN_ONSET_S:
            out.loc[i, "exclude_reason"] = (f"fall starts {t[on]:.2f} s into the video: the GRU needs "
                                            f"{C.WINDOW} frames of history, so no fair lead time")
    return out


def split_segments(segs, trials, frames):
    ts = {v: g.set_index("frame_idx")["timestamp"] for v, g in frames.groupby("video_id")}
    vend = frames.groupby("video_id")["timestamp"].max()
    out, notes = [], []
    for vid, s in segs.groupby("video_id", sort=False):
        s = s.sort_values("start_s")
        falls = s[s["label"] == "fall"]
        if falls.empty:
            out.append(s)
            continue
        f = falls.iloc[0].to_dict()
        tail = float(vend.get(vid, f["end_s"])) - float(f["end_s"])
        t_end = float(vend[vid]) + 1.0 / FPS if tail <= EXTEND_FALLEN_MAX_S else float(f["end_s"])
        pre = s[(s["label"] != "fall") & (s["start_s"] < f["start_s"])].copy()
        pre["end_s"] = pre["end_s"].clip(upper=f["start_s"])
        row = trials[(trials["video_id"] == vid) & (trials["trial_type"] == "fall")]
        imp = row["impact_frame"].iloc[0] if len(row) else np.nan
        if pd.notna(imp):
            t_imp = float(ts[vid].loc[int(imp)])
            new = [{**f, "end_s": t_imp},
                   {**f, "start_s": t_imp, "end_s": t_end, "label": "fallen",
                    "source_label": f"{f['source_label']} (after impact)"}]
        else:
            new = [{**f, "end_s": t_end}]
            notes.append(f"{vid}: no impact suggested - whole fall segment kept as 'fall'")
        out.append(pd.concat([pre, pd.DataFrame(new)], ignore_index=True))
        notes.append(f"{vid}: fall label ends {tail:.2f} s before the video end") if tail > EXTEND_FALLEN_MAX_S else None
    res = pd.concat(out, ignore_index=True)
    return res[segs.columns.tolist()], [n for n in notes if n]


def scenario_names(trials, segs):
    names = []
    for r in trials.itertuples():
        s = segs[segs["video_id"] == r.video_id].sort_values("start_s")
        if r.trial_type == "fall":
            f = s[s["label"] == "fall"].iloc[0]
            before = s[(s["label"] != "fall") & (s["start_s"] < f["start_s"])]
            prev = before.iloc[-1]["label"] if len(before) else "unknown"
            direction = f.get("fall_direction") if isinstance(f.get("fall_direction"), str) else "unknown"
            names.append(f"{prev}->fall_{direction}")
        else:
            dur = (s["end_s"] - s["start_s"]).groupby(s["label"]).sum()
            names.append(str(dur.idxmax()) if len(dur) else "adl")
    return names


def review_table(trials, frames):
    rows = []
    by_vid = dict(tuple(frames.groupby("video_id")))
    for r in trials[trials["trial_type"] == "fall"].itertuples():
        g = by_vid[r.video_id].sort_values("frame_idx")
        t, fidx = g["timestamp"].to_numpy(float), g["frame_idx"].to_numpy(int)
        hy, torso = hip_torso(g)
        sy = shoulder_y(g)
        on = int(np.searchsorted(fidx, r.onset_frame))
        rec = {"trial_id": r.trial_id, "subject_id": r.subject_id, "scenario": r.scenario,
               "onset_s": round(t[on], 3), "impact_s": np.nan, "impact_source": r.impact_source,
               "impact_hip_s": round(r.impact_hip_s, 3), "impact_trunk_s": round(r.impact_trunk_s, 3),
               "impact_knee_s": round(r.impact_knee_s, 3),
               "fall_duration_s": np.nan, "hip_drop_torso_lengths": round(r.hip_drop_TL, 2),
               "peak_descent_TL_per_s": np.nan, "detected_share_onset_to_impact": np.nan,
               "flag": "", "excluded_from_trials": r.exclude_reason}
        if pd.notna(r.impact_frame):
            im = int(np.searchsorted(fidx, int(r.impact_frame)))
            ref = torso[max(0, on - 30):on + 1]
            ref = np.nanmedian(ref) if np.isfinite(ref).any() else np.nanmedian(torso)
            y = hy
            v = np.diff(y) / np.diff(t) * g["frame_h"].to_numpy(float)[1:] / ref
            seg_v = v[on:im] if im > on else np.array([np.nan])
            rec.update({"impact_s": round(t[im], 3), "fall_duration_s": round(t[im] - t[on], 3),
                        "peak_descent_TL_per_s": round(float(np.nanmax(seg_v)), 2) if np.isfinite(seg_v).any() else np.nan,
                        "detected_share_onset_to_impact": round(float(g["detected"].to_numpy()[on:im + 1].mean()), 2)})
            flags = []
            if rec["fall_duration_s"] < 0.25:
                flags.append("very short onset->impact")
            if rec["fall_duration_s"] > 2.0:
                flags.append("long onset->impact")
            est = [x for x in (r.impact_hip_s, r.impact_trunk_s, r.impact_knee_s) if pd.notna(x)]
            if est and max(est) - min(est) > 0.4 and not str(r.impact_source).startswith("visual"):
                flags.append(f"hip/trunk/knee estimates spread {max(est) - min(est):.2f} s")
            if rec["detected_share_onset_to_impact"] < 0.8:
                flags.append("pose missing during the fall")
            rec["flag"] = "; ".join(flags)
        else:
            rec["flag"] = "no impact suggested"
        rows.append(rec)
    return pd.DataFrame(rows)


def dense_strips(trials, manifest, out_dir, per_image=3, thumb_w=150, per_row=10, step=2, max_span=2.0,
                 prefix="dense"):
    """Every `step`-th video frame around the impact candidates (H = hip, T = trunk, red = chosen),
    for the trials whose suggestion needs a closer look. Returns {trial_id: image name}."""
    import cv2
    media = manifest.set_index("video_id")["media"]
    blocks, names, k_img = [], {}, 1

    def flush():
        nonlocal blocks, k_img
        if blocks:
            w = max(b.shape[1] for b in blocks)
            blocks = [cv2.copyMakeBorder(b, 0, 6, 0, w - b.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                      for b in blocks]
            cv2.imwrite(str(out_dir / f"{prefix}_{k_img:02d}.jpg"), np.vstack(blocks), [cv2.IMWRITE_JPEG_QUALITY, 82])
            blocks, k_img = [], k_img + 1

    for r in trials.itertuples():
        cands = [c for c in (r.impact_hip_s, r.impact_trunk_s, r.impact_knee_s, r.impact_s) if pd.notna(c)]
        t0 = max(0.0, min([r.onset_s] + cands) - 0.2)
        t1 = min(max(cands) + 0.4, t0 + max_span)
        cap = cv2.VideoCapture(str(media[r.video_id]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i0, i1 = int(round(t0 * fps)), int(round(t1 * fps))
        keep = set(range(i0, i1 + 1, step)) | {int(round(c * fps)) for c in cands}
        frames, i = {}, 0
        while i <= i1:
            ok, img = cap.read()
            if not ok:
                break
            if i in keep:
                frames[i] = img
            i += 1
        cap.release()
        thumbs = []
        for fi in sorted(frames):
            img = frames[fi]
            th = cv2.resize(img, (thumb_w, int(img.shape[0] * thumb_w / img.shape[1])))
            tt = fi / fps
            tags = "".join(tag for tag, c in (("H", r.impact_hip_s), ("T", r.impact_trunk_s), ("K", r.impact_knee_s))
                           if pd.notna(c) and int(round(c * fps)) == fi)
            chosen = pd.notna(r.impact_s) and int(round(r.impact_s * fps)) == fi
            cv2.rectangle(th, (0, 0), (thumb_w, 15), (0, 0, 0), -1)
            cv2.putText(th, f"{tt:.2f}s {tags}", (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 0, 255) if chosen else ((0, 200, 255) if tags else (255, 255, 255)), 1, cv2.LINE_AA)
            if chosen or tags:
                cv2.rectangle(th, (0, 0), (th.shape[1] - 1, th.shape[0] - 1),
                              (0, 0, 255) if chosen else (0, 200, 255), 2)
            thumbs.append(th)
        if not thumbs:
            continue
        h = max(t.shape[0] for t in thumbs)
        blank = np.full((h, thumb_w, 3), 255, np.uint8)
        rows_ = []
        for j in range(0, len(thumbs), per_row):
            row = thumbs[j:j + per_row]
            row += [blank] * (per_row - len(row))
            rows_.append(np.hstack([cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 1, cv2.BORDER_CONSTANT,
                                                       value=(255, 255, 255)) for x in row]))
        body = np.vstack(rows_)
        band = np.full((20, body.shape[1], 3), 235, np.uint8)
        txt = (f"{r.trial_id} | {r.scenario} | onset {r.onset_s:.2f}s | H={r.impact_hip_s:.2f} "
               f"T={r.impact_trunk_s:.2f} K={r.impact_knee_s:.2f} | chosen {r.impact_s:.2f}s")
        cv2.putText(band, txt, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        blocks.append(np.vstack([band, body]))
        names[r.trial_id] = f"{prefix}_{k_img:02d}.jpg"
        if len(blocks) == per_image:
            flush()
    flush()
    return names


def contact_sheets(trials, review, manifest, out_dir, per_sheet=6, thumb_w=224):
    """One row per fall: frames at onset, impact-0.2 s, impact-0.1 s, impact, impact+0.2 s (for checking by eye)."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    media = manifest.set_index("video_id")["media"]
    rev = review.set_index("trial_id")
    falls = trials[trials["trial_type"] == "fall"].reset_index(drop=True)
    sheet_of = {}
    offsets = [("onset", None), ("imp-0.2s", -0.2), ("imp-0.1s", -0.1), ("IMPACT", 0.0), ("imp+0.2s", 0.2)]
    rows_img, k_sheet = [], 1

    def flush():
        nonlocal rows_img, k_sheet
        if rows_img:
            cv2.imwrite(str(out_dir / f"impact_check_{k_sheet:02d}.jpg"), np.vstack(rows_img),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
            rows_img, k_sheet = [], k_sheet + 1

    for r in falls.itertuples():
        rv = rev.loc[r.trial_id]
        if pd.isna(rv["impact_s"]):
            continue
        cap = cv2.VideoCapture(str(media[r.video_id]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        want = [rv["onset_s"] if off is None else rv["impact_s"] + off for _, off in offsets]
        idx = [max(0, int(round(t * fps))) for t in want]
        got, i = {}, 0
        while i <= max(idx):
            ok, img = cap.read()
            if not ok:
                break
            if i in idx:
                got[i] = img
            i += 1
        cap.release()
        last = got[max(got)] if got else None
        thumbs = []
        for (tag, _), t, fi in zip(offsets, want, idx):
            img = got.get(fi, last)
            if img is None:
                img = np.zeros((480, 640, 3), np.uint8)
            th = cv2.resize(img, (thumb_w, int(img.shape[0] * thumb_w / img.shape[1])))
            col = (0, 0, 255) if tag == "IMPACT" else (255, 255, 255)
            cv2.rectangle(th, (0, 0), (thumb_w, 18), (0, 0, 0), -1)
            cv2.putText(th, f"{tag} {t:.2f}s", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            if tag == "IMPACT":
                cv2.rectangle(th, (0, 0), (th.shape[1] - 1, th.shape[0] - 1), (0, 0, 255), 3)
            thumbs.append(th)
        hmax = max(t.shape[0] for t in thumbs)
        thumbs = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40))
                  for t in thumbs]
        band = np.full((22, sum(t.shape[1] for t in thumbs), 3), 245, np.uint8)
        label = f"{r.trial_id} | {r.scenario} | onset->impact {rv['fall_duration_s']:.2f}s"
        if isinstance(rv["flag"], str) and rv["flag"]:
            label += f" | CHECK: {rv['flag']}"
        cv2.putText(band, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        rows_img.append(np.vstack([band, np.hstack(thumbs)]))
        sheet_of[r.trial_id] = f"impact_check_{k_sheet:02d}.jpg"
        if len(rows_img) == per_sheet:
            flush()
    flush()
    return sheet_of


def stage_label(args):
    from prepare_open_data import label_frames
    native, f30, sug = D / "frames.csv", D / "frames_30fps.csv", D / "trials_suggested.csv"
    if not args.skip_prep:
        run(sys.executable, "prepare_open_data.py", "extract", "--dir", D, "--merge-only")
        run(sys.executable, "prepare_open_data.py", "resample", "--frames", native, "--fps", FPS, "--out", f30)
        run(sys.executable, "prepare_open_data.py", "trials", "--dir", D, "--frames", f30, "--mode", "video",
            "--impact", "suggest", "--out", sug)

    frames = pd.read_csv(f30)
    segs = pd.read_csv(D / "segments.csv")
    sugg = pd.read_csv(sug)
    sugg["scenario"] = scenario_names(sugg, segs)
    trials = choose_impacts(sugg, frames)
    split, notes = split_segments(segs, trials, frames)
    split.to_csv(D / "segments_split.csv", index=False)
    for n in notes:
        print("  note:", n)

    frames["label"] = label_frames(frames, split, LABEL_MAP, "normal").to_numpy()
    frames.to_csv(f30, index=False, float_format="%.5f")
    nat = pd.read_csv(native)
    nat["label"] = label_frames(nat, split, LABEL_MAP, "normal").to_numpy()
    nat.to_csv(native, index=False, float_format="%.5f")
    print("labels (30 fps):\n" + frames["label"].value_counts().to_string())

    final = trials[trials["exclude_reason"] == ""]
    final[[c for c in TRIAL_COLS if c in final.columns]].to_csv(D / "trials.csv", index=False)
    review = review_table(trials, frames)
    manifest = pd.read_csv(D / "manifest.csv")
    chk = D / "impact_check"
    if chk.exists():
        shutil.rmtree(chk)
    chk.mkdir()
    review["contact_sheet"] = review["trial_id"].map(contact_sheets(trials, review, manifest, chk))
    keep = review[review["excluded_from_trials"] == ""].merge(trials[["trial_id", "video_id"]], on="trial_id")
    review["frame_strip"] = review["trial_id"].map(dense_strips(keep, manifest, chk, prefix="strip"))
    review.to_csv(D / "impact_review.csv", index=False)
    print(f"trials.csv: {int((final.trial_type == 'fall').sum())} fall, {int((final.trial_type == 'adl').sum())} ADL "
          f"(left out: {len(trials) - len(final)}); flagged: {int((review['flag'] != '').sum())}")
    print(review["impact_source"].str.split(" ").str[0].value_counts().to_string())
    print(review[["fall_duration_s", "hip_drop_torso_lengths", "peak_descent_TL_per_s"]].describe().round(2).to_string())


# ----------------------------------------------------------------------------------------------- train
def stage_train(args):
    f30 = D / "frames_30fps.csv"
    run(sys.executable, "loso.py", "--frames", f30, "--mode", "loso", "--out", D / "results",
        "--models", D / "models", "--seed", str(args.seed))
    oof = pd.read_csv(D / "results" / "loso" / "oof_predictions.csv")
    keep = ["subject_id", "video_id", "frame_idx", "fold"] + [f"prob_{c}" for c in C.CLASS_NAMES]
    oof[keep].to_csv(D / "predictions.csv", index=False, float_format="%.5f")
    print(f"wrote {D / 'predictions.csv'}: {len(oof)} rows")


# ----------------------------------------------------------------------------------------------- verify
def write_upload():
    up = D / "upload"
    up.mkdir(exist_ok=True)
    fr = pd.read_csv(D / "frames_30fps.csv")
    fr[[c for c in APP_COLS if c in fr.columns]].to_csv(up / "frames.csv", index=False, float_format="%.5f")
    shutil.copy(D / "trials.csv", up / "trials.csv")
    shutil.copy(D / "predictions.csv", up / "predictions.csv")
    return up


def stage_verify(args):
    import gzip
    sys.path.insert(0, str(TK / "webapp"))
    from webapp.analysis import analyze
    up = write_upload()
    sizes = {}
    for name in ("frames.csv", "trials.csv", "predictions.csv"):
        raw = (up / name).read_bytes()
        sizes[name] = {"MB": round(len(raw) / 1e6, 2), "gzip_MB": round(len(gzip.compress(raw, 6)) / 1e6, 2)}
    res = analyze(up / "frames.csv", up / "trials.csv", up / "predictions.csv", settings=SETTINGS)
    keep = {k: res[k] for k in ("settings", "meta", "summary", "summary_row", "by_scenario", "ablation",
                                 "loso_thresholds", "runtime_s") if k in res}
    keep["upload_sizes"] = sizes
    (D / "reference_analysis.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False, default=str))
    pd.DataFrame(res["per_trial"]).to_csv(D / "reference_per_trial.csv", index=False)
    print(json.dumps({"meta": keep["meta"], "summary_row": keep["summary_row"], "sizes": sizes}, indent=1,
                     ensure_ascii=False, default=str))


# ----------------------------------------------------------------------------------------------- package
def stage_package(args):
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "upload").mkdir(parents=True)
    for n in ("frames.csv", "trials.csv", "predictions.csv"):
        shutil.copy(D / "upload" / n, OUT / "upload" / n)
    full = OUT / "full_landmarks"
    full.mkdir()
    shutil.copy(D / "frames_30fps.csv", full / "frames_30fps_all33.csv")
    lab = OUT / "labels"
    lab.mkdir()
    for n in ("manifest.csv", "segments.csv", "segments_split.csv", "trials_suggested.csv", "impact_review.csv"):
        shutil.copy(D / n, lab / n)
    man = pd.read_csv(lab / "manifest.csv")
    man["media"] = (man["media"].str.replace(r"^.*?[\\/](Subject \d+[\\/])", r"\1", regex=True)
                    .str.replace("\\", "/", regex=False))                            # drop local paths
    man.to_csv(lab / "manifest.csv", index=False)
    shutil.copytree(D / "impact_check", OUT / "impact_check")
    ref = OUT / "reference_results"
    ref.mkdir()
    for n in ("reference_analysis.json", "reference_per_trial.csv"):
        shutil.copy(D / n, ref / n)
    for n in ("fold_metrics.csv", "summary.md", "summary.json", "confusion_matrix.png"):
        p = D / "results" / "loso" / n
        if p.exists():
            shutil.copy(p, ref / f"loso_{n}")
    sc = OUT / "scripts"
    sc.mkdir()
    for p in (HERE / "gmdcsa24_labels.py", Path(__file__).resolve()):
        shutil.copy(p, sc / p.name)
    for p in BUNDLE_DOCS:
        if p.exists():
            shutil.copy(p, OUT / p.name)
    lic = VIDEOS / "LICENSE" if VIDEOS else None
    if lic is not None and lic.exists():
        shutil.copy(lic, OUT / "LICENSE_GMDCSA24_MIT.txt")
    else:
        print("note: pass --videos <GMDCSA24 clone> to include the dataset's MIT licence notice in the bundle")
    zpath = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


def main():
    global D, OUT, VIDEOS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["label", "train", "verify", "package"])
    ap.add_argument("--dir", default=str(D), help="work folder with manifest.csv, segments.csv and parts/")
    ap.add_argument("--out", default=str(OUT), help="bundle folder (a .zip with the same name is written next to it)")
    ap.add_argument("--videos", help="GMDCSA24 clone (package: copies its LICENSE into the bundle)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-prep", action="store_true", help="label: reuse frames_30fps.csv + trials_suggested.csv")
    a = ap.parse_args()
    D, OUT = Path(a.dir).resolve(), Path(a.out).resolve()
    VIDEOS = Path(a.videos).resolve() if a.videos else None
    {"label": stage_label, "train": stage_train, "verify": stage_verify, "package": stage_package}[a.stage](a)


if __name__ == "__main__":
    main()
