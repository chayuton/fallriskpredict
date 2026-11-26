"""
Decision rule of the running system and trial-level evaluation (shared by lead_time.py and ablation.py).

    alert(t) = [GRU positive for N consecutive frames]  AND  [hip descent velocity >= V]

Everything is causal: the decision at frame t only uses frames <= t, exactly like the live system.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C
from data import hip_center_y, hip_descent_velocity, positive_ids, prob_columns, read_table


@dataclass(frozen=True)
class Rule:
    name: str = "GRU + persistence + velocity (proposed)"
    use_gru: bool = True
    p_thresh: float = C.P_THRESH
    persist: int = C.PERSIST_FRAMES
    persist_on: str = C.PERSIST_ON      # "gru" | "combined"
    use_velocity: bool = True
    v_thresh: float = C.V_THRESH
    vel_window: int = C.VEL_WINDOW


def run_length(b: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive True values ending at each position."""
    b = b.astype(np.int64)
    c = np.cumsum(b)
    reset = np.maximum.accumulate(np.where(b == 0, c, 0))
    return c - reset


def alert_mask(p: np.ndarray, v: np.ndarray, rule: Rule) -> np.ndarray:
    n = len(p)
    # no GRU output yet (first WINDOW-1 frames) -> not positive
    g = np.nan_to_num(p, nan=-1.0) >= rule.p_thresh if rule.use_gru else np.ones(n, bool)
    if rule.use_velocity:
        vv = v if rule.vel_window <= 1 else pd.Series(v).rolling(rule.vel_window, min_periods=1).max().to_numpy()
        ok = np.nan_to_num(vv, nan=-np.inf) >= rule.v_thresh
    else:
        ok = np.ones(n, bool)
    if rule.persist_on == "combined":
        return run_length(g & ok) >= rule.persist
    return (run_length(g) >= rule.persist) & ok


# ----------------------------------------------------------------------------
# per-video traces: time, GRU positive probability, hip velocity
# ----------------------------------------------------------------------------
@dataclass
class Trace:
    video_id: str
    subject_id: str
    frame: np.ndarray
    t: np.ndarray
    p: np.ndarray      # GRU probability of the warning class(es); NaN before the first full window
    v: np.ndarray      # hip descent velocity (VELOCITY_MODE units), positive = downward
    hip_y: np.ndarray  # normalized hip y (for plots / impact suggestion)


def load_predictions(path) -> pd.DataFrame:
    df = read_table(path)
    df[C.VIDEO_COL] = df[C.VIDEO_COL].astype(str)
    cols = prob_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing probability columns {missing}")
    df["p_pos"] = df[[cols[i] for i in positive_ids()]].sum(axis=1)
    return df[[C.VIDEO_COL, C.FRAME_COL, "p_pos"]]


def build_traces(frames: pd.DataFrame, preds: pd.DataFrame | None, videos=None) -> dict[str, Trace]:
    pgroups = {}
    if preds is not None:
        pgroups = {v: g.set_index(C.FRAME_COL)["p_pos"] for v, g in preds.groupby(C.VIDEO_COL)}
    traces = {}
    for vid, idx in frames.groupby(C.VIDEO_COL, sort=False).indices.items():
        if videos is not None and vid not in videos:
            continue
        g = frames.iloc[idx]
        fr = g[C.FRAME_COL].to_numpy()
        p = pgroups[vid].reindex(fr).to_numpy(float) if vid in pgroups else np.full(len(g), np.nan)
        if preds is not None and vid not in pgroups:
            warnings.warn(f"video {vid}: no GRU predictions found")
        traces[vid] = Trace(vid, str(g[C.SUBJECT_COL].iloc[0]), fr, g[C.TIME_COL].to_numpy(float),
                            p, hip_descent_velocity(g), hip_center_y(g))
    return traces


def slice_trace(tr: Trace, start_frame=None, end_frame=None) -> Trace:
    """Part of a trace between two frame numbers (inclusive); NaN bounds = open."""
    m = np.ones(len(tr.frame), bool)
    if start_frame is not None and pd.notna(start_frame):
        m &= tr.frame >= start_frame
    if end_frame is not None and pd.notna(end_frame):
        m &= tr.frame <= end_frame
    return Trace(tr.video_id, tr.subject_id, tr.frame[m], tr.t[m], tr.p[m], tr.v[m], tr.hip_y[m])


# ----------------------------------------------------------------------------
# trial annotations
# ----------------------------------------------------------------------------
def load_trials(path, frames: pd.DataFrame, require_impact: bool = True) -> pd.DataFrame:
    tr = read_table(path)
    if C.VIDEO_COL not in tr.columns or "trial_type" not in tr.columns:
        raise ValueError(f"{path}: needs columns '{C.VIDEO_COL}' and 'trial_type' (fall/adl)")
    tr[C.VIDEO_COL] = tr[C.VIDEO_COL].astype(str)
    tr["trial_type"] = tr["trial_type"].astype(str).str.strip().str.lower()
    bad = ~tr["trial_type"].isin(["fall", "adl"])
    if bad.any():
        raise ValueError(f"trial_type must be 'fall' or 'adl': {tr.loc[bad, 'trial_type'].unique()[:5]}")
    known = set(frames[C.VIDEO_COL].unique())
    unknown = sorted(set(tr[C.VIDEO_COL]) - known)
    if unknown:
        warnings.warn(f"{len(unknown)} trial videos not in the frame table, ignored: {unknown[:5]}")
        tr = tr[tr[C.VIDEO_COL].isin(known)].copy()
    v2s = frames.groupby(C.VIDEO_COL)[C.SUBJECT_COL].first()
    tr[C.SUBJECT_COL] = tr[C.VIDEO_COL].map(v2s).astype(str)
    if "scenario" not in tr.columns:
        tr["scenario"] = tr["trial_type"]
    for col in ("impact_frame", "onset_frame", "start_frame", "end_frame"):
        if col not in tr.columns:
            tr[col] = np.nan
    # several trials may share one long recording: give each a trial_id and a frame window
    if "trial_id" not in tr.columns:
        tr["trial_id"] = tr[C.VIDEO_COL]
    tr["trial_id"] = tr["trial_id"].astype(str)
    dup = tr["trial_id"].duplicated(keep=False)
    if dup.any():
        tr.loc[dup, "trial_id"] = tr.loc[dup, "trial_id"] + "#" + tr[dup].groupby("trial_id").cumcount().astype(str)
    if require_impact:
        miss = tr[(tr["trial_type"] == "fall") & tr["impact_frame"].isna()]
        if len(miss):
            raise ValueError(f"{len(miss)} fall trials without impact_frame "
                             f"(run lead_time.py --suggest-impact to get suggestions): "
                             f"{miss[C.VIDEO_COL].tolist()[:5]}")
    return tr.reset_index(drop=True)


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
@dataclass
class PreparedTrial:
    row: dict
    trace: Trace
    is_fall: bool
    t_impact: float = np.nan
    t_onset: float = np.nan
    win_start: float = np.nan
    win_end: float = np.nan


def prepare_trials(trials: pd.DataFrame, traces: dict[str, Trace]) -> list[PreparedTrial]:
    out = []
    for row in trials.to_dict("records"):
        tr = traces.get(row[C.VIDEO_COL])
        if tr is None:
            continue
        if pd.notna(row.get("start_frame")) or pd.notna(row.get("end_frame")):
            tr = slice_trace(tr, row.get("start_frame"), row.get("end_frame"))
            if len(tr.frame) == 0:
                warnings.warn(f"trial {row['trial_id']}: empty frame window, skipped")
                continue
        pt = PreparedTrial(row, tr, row["trial_type"] == "fall")
        if pt.is_fall:
            pt.t_impact = float(np.interp(row["impact_frame"], tr.frame, tr.t))
            if pd.notna(row.get("onset_frame")):
                pt.t_onset = float(np.interp(row["onset_frame"], tr.frame, tr.t))
                pt.win_start = pt.t_onset - C.ONSET_TOL_S
            else:
                pt.win_start = pt.t_impact - C.MAX_PRE_IMPACT_S
            pt.win_end = pt.t_impact + C.MAX_POST_IMPACT_S
        out.append(pt)
    return out


def count_events(times: np.ndarray) -> int:
    """Alerts closer than REFRACTORY_S to the start of the current alarm belong to the same alarm."""
    n, start = 0, -np.inf
    for x in times:
        if x - start > C.REFRACTORY_S:
            n, start = n + 1, x
    return n


def evaluate_trial(pt: PreparedTrial, rule: Rule, latency_ms: float = 0.0) -> dict:
    tr = pt.trace
    alert = alert_mask(tr.p, tr.v, rule)
    a_idx = np.flatnonzero(alert)
    a_t = tr.t[a_idx]
    res = {"rule": rule.name, "trial_id": pt.row["trial_id"], C.VIDEO_COL: tr.video_id, C.SUBJECT_COL: tr.subject_id,
           "trial_type": pt.row["trial_type"], "scenario": pt.row["scenario"],
           "onset_frame": pt.row.get("onset_frame"), "impact_frame": pt.row.get("impact_frame"),
           "detected": False, "pre_impact": False, "alert_frame": np.nan, "lead_ms": np.nan,
           "eff_lead_ms": np.nan, "delay_from_onset_ms": np.nan,
           "duration_s": float(tr.t[-1] - tr.t[0] + 1.0 / C.FPS)}
    if pt.is_fall:
        early = a_t < pt.win_start
        inwin = (a_t >= pt.win_start) & (a_t <= pt.win_end)
        if inwin.any():
            k = a_idx[inwin][0]
            lead = (pt.t_impact - tr.t[k]) * 1000.0
            res.update(detected=True, alert_frame=tr.frame[k], lead_ms=lead,
                       eff_lead_ms=lead - latency_ms, pre_impact=bool(lead - latency_ms > 0))
            if not np.isnan(pt.t_onset):
                res["delay_from_onset_ms"] = (tr.t[k] - pt.t_onset) * 1000.0
        res["n_false_alarm_events"] = count_events(a_t[early])
    else:
        res["n_false_alarm_events"] = count_events(a_t)
    res["false_alarm"] = res["n_false_alarm_events"] > 0
    return res


def evaluate(prepared: list[PreparedTrial], rule: Rule, latency_ms: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame([evaluate_trial(pt, rule, latency_ms) for pt in prepared])


def summarize(res: pd.DataFrame) -> dict:
    f = res[res["trial_type"] == "fall"]
    a = res[res["trial_type"] == "adl"]
    d = f[f["detected"].astype(bool)]
    n_f, n_a = len(f), len(a)
    lo, hi = C.LEAD_TARGET_MS
    lead = d["eff_lead_ms"].astype(float)
    nan = float("nan")
    adl_fa = int(a["false_alarm"].sum()) if n_a else 0
    adl_hours = float(a["duration_s"].sum()) / 3600.0 if n_a else 0.0
    n_pre = int(f["pre_impact"].sum())
    s = {
        "rule": res["rule"].iloc[0] if len(res) else "",
        "n_falls": n_f, "n_detected": len(d),
        "detection_rate": len(d) / n_f if n_f else nan,
        "n_pre_impact": n_pre, "pre_impact_rate": n_pre / n_f if n_f else nan,
        "lead_mean_ms": lead.mean() if len(d) else nan,
        "lead_sd_ms": lead.std(ddof=1) if len(d) > 1 else nan,
        "lead_median_ms": lead.median() if len(d) else nan,
        "lead_q1_ms": lead.quantile(0.25) if len(d) else nan,
        "lead_q3_ms": lead.quantile(0.75) if len(d) else nan,
        "lead_min_ms": lead.min() if len(d) else nan,
        "lead_max_ms": lead.max() if len(d) else nan,
        "algorithmic_lead_median_ms": d["lead_ms"].median() if len(d) else nan,
        "rate_lead_ge_target_min": (f["eff_lead_ms"] >= lo).sum() / n_f if n_f else nan,
        "rate_lead_in_target": ((f["eff_lead_ms"] >= lo) & (f["eff_lead_ms"] <= hi)).sum() / n_f if n_f else nan,
        "delay_from_onset_median_ms": d["delay_from_onset_ms"].median() if len(d) else nan,
        "fall_trials_with_early_false_alarm": int(f["false_alarm"].sum()) if n_f else 0,
        "n_adl": n_a, "adl_trials_with_false_alarm": adl_fa,
        "specificity": 1 - adl_fa / n_a if n_a else nan,
        "adl_false_alarm_events": int(a["n_false_alarm_events"].sum()) if n_a else 0,
        "adl_hours": adl_hours,
        "false_alarms_per_hour": a["n_false_alarm_events"].sum() / adl_hours if adl_hours > 0 else nan,
        "trial_accuracy": (len(d) + n_a - adl_fa) / (n_f + n_a) if n_f + n_a else nan,
        "trial_accuracy_pre_impact": (n_pre + n_a - adl_fa) / (n_f + n_a) if n_f + n_a else nan,
    }
    return s


def summary_row(s: dict, fa_rate: bool = True) -> dict:
    """Human-readable row for Markdown tables."""
    def frac(k, n):
        return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "–"

    lead = ("–" if np.isnan(s["lead_median_ms"]) else
            f"{s['lead_median_ms']:.0f} [{s['lead_q1_ms']:.0f}, {s['lead_q3_ms']:.0f}]")
    lo = C.LEAD_TARGET_MS[0]
    row = {
        "Configuration": s["rule"],
        "Falls detected": frac(s["n_detected"], s["n_falls"]),
        "Warned before impact": frac(s["n_pre_impact"], s["n_falls"]),
        "Lead time ms, median [IQR]": lead,
        f"Lead ≥ {lo} ms": "–" if np.isnan(s["rate_lead_ge_target_min"]) else f"{100 * s['rate_lead_ge_target_min']:.1f}%",
        "ADL trials with false alarm": frac(s["adl_trials_with_false_alarm"], s["n_adl"]),
        "Specificity": "–" if np.isnan(s["specificity"]) else f"{100 * s['specificity']:.1f}%",
        "False alarms / h (ADL)": "–" if np.isnan(s["false_alarms_per_hour"]) else f"{s['false_alarms_per_hour']:.1f}",
        "Trial accuracy": "–" if np.isnan(s["trial_accuracy"]) else f"{100 * s['trial_accuracy']:.1f}%",
    }
    if not fa_rate:
        row.pop("False alarms / h (ADL)")
    return row
