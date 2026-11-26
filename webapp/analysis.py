"""
Analysis service behind the web app: lead time, rule ablation, threshold sensitivity and
LOSO threshold selection, computed with the same code as lead_time.py / ablation.py
(decision.py), but returning JSON instead of files. No TensorFlow, no matplotlib.
"""
from __future__ import annotations

import io
import math
import time

import numpy as np
import pandas as pd

import config as C
from data import load_frames
from decision import Rule, alert_mask, build_traces, evaluate, load_trials, prepare_trials, summarize, summary_row

# settings the app may change: name -> (config attribute, type, min, max)
SETTINGS = {
    "velocity_mode": ("VELOCITY_MODE", str, None, None),
    "v_thresh": ("V_THRESH", float, 0.0, 1e4),
    "p_thresh": ("P_THRESH", float, 0.0, 1.0),
    "persist": ("PERSIST_FRAMES", int, 1, 30),
    "persist_on": ("PERSIST_ON", str, None, None),
    "vel_window": ("VEL_WINDOW", int, 1, 30),
    "vel_lag": ("VEL_LAG_FRAMES", int, 1, 15),
    "fps": ("FPS", float, 1.0, 240.0),
    "frame_w": ("FRAME_W", int, 16, 10000),
    "frame_h": ("FRAME_H", int, 16, 10000),
    "frame_h_meters": ("FRAME_H_METERS", float, 0.1, 100.0),
    "max_pre_impact_s": ("MAX_PRE_IMPACT_S", float, 0.1, 30.0),
    "onset_tol_s": ("ONSET_TOL_S", float, 0.0, 10.0),
    "max_post_impact_s": ("MAX_POST_IMPACT_S", float, 0.0, 30.0),
    "refractory_s": ("REFRACTORY_S", float, 0.0, 600.0),
}
CHOICES = {"velocity_mode": ("meters", "torso", "pixels"), "persist_on": ("gru", "combined")}
DEFAULTS = {k: getattr(C, attr) for k, (attr, *_rest) in SETTINGS.items()}
MAX_GRID_CONFIGS = 400


def apply_settings(settings: dict | None) -> dict:
    """Reset config to defaults (warm serverless instances are reused), then apply validated overrides."""
    used = dict(DEFAULTS)
    for k, v in (settings or {}).items():
        if k not in SETTINGS or v is None or v == "":
            continue
        attr, typ, lo, hi = SETTINGS[k]
        val = typ(v)
        if k in CHOICES and val not in CHOICES[k]:
            raise ValueError(f"{k} must be one of {CHOICES[k]}")
        if lo is not None and not (lo <= val <= hi):
            raise ValueError(f"{k} must be between {lo} and {hi}")
        used[k] = val
    for k, val in used.items():
        setattr(C, SETTINGS[k][0], val)
    return used


def _finite(x):
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _finite(v) for k, v in r.items()} for r in df.to_dict("records")]


def _summary(s: dict) -> dict:
    return {k: _finite(v) for k, v in s.items()}


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar test (binomial on discordant pairs), without scipy."""
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_predictions_frame(src) -> pd.DataFrame:
    """Predictions with a p_pos column, or prob_<class> columns + positive classes chosen in config."""
    df = pd.read_csv(src)
    df[C.VIDEO_COL] = df[C.VIDEO_COL].astype(str)
    if "p_pos" not in df.columns:
        cols = [f"prob_{c}" for c in C.POSITIVE_CLASSES]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"predictions need a p_pos column or {missing}")
        df["p_pos"] = df[cols].sum(axis=1)
    return df[[C.VIDEO_COL, C.FRAME_COL, "p_pos"]]


def ablation_rules(n: int):
    return [
        Rule("GRU only (N=1)", persist=1, use_velocity=False),
        Rule(f"GRU + persistence (N={n})", use_velocity=False),
        Rule("GRU + velocity (N=1)", persist=1),
        Rule(f"GRU + persistence (N={n}) + velocity (proposed)"),
        Rule("Velocity only (N=1)", use_gru=False, persist=1, persist_on="combined"),
        Rule(f"Velocity only (N={n})", use_gru=False, persist_on="combined"),
    ]


def _examples(prepared, res, rule, n=6):
    f = res[(res["trial_type"] == "fall")]
    d = f[f["detected"]].sort_values("lead_ms")
    picks = []
    if len(d):
        q = np.unique(np.round(np.linspace(0, len(d) - 1, max(1, n - 1))).astype(int))
        picks += d.iloc[q]["trial_id"].tolist()
    picks += f[~f["detected"]]["trial_id"].tolist()[:1]
    by = {pt.row["trial_id"]: pt for pt in prepared}
    out = []
    for tid in picks[:n]:
        pt, row = by[tid], res[res["trial_id"] == tid].iloc[0]
        tr = pt.trace
        x = (tr.t - pt.t_impact) * 1000.0
        keep = (x >= -max(C.MAX_PRE_IMPACT_S * 1000, 1500)) & (x <= C.MAX_POST_IMPACT_S * 1000)
        alert = alert_mask(tr.p, tr.v, rule)
        out.append({
            "trial_id": tid, "scenario": row["scenario"], "detected": bool(row["detected"]),
            "lead_ms": _finite(row["lead_ms"]), "eff_lead_ms": _finite(row["eff_lead_ms"]),
            "onset_ms": _finite((pt.t_onset - pt.t_impact) * 1000.0) if not np.isnan(pt.t_onset) else None,
            "t_ms": [round(float(v), 1) for v in x[keep]],
            "hip_height": [_finite(round(1 - float(v), 5)) if np.isfinite(v) else None for v in tr.hip_y[keep]],
            "velocity": [_finite(round(float(v), 4)) if np.isfinite(v) else None for v in tr.v[keep]],
            "p": [_finite(round(float(v), 4)) if np.isfinite(v) else None for v in tr.p[keep]],
            "alert": [bool(a) for a in alert[keep]],
        })
    return out


def _grid(prepared, rule, latency_ms):
    vgrid = list(C.GRID_V[C.VELOCITY_MODE])
    if rule.v_thresh not in vgrid:
        vgrid = sorted(vgrid + [rule.v_thresh])
    pgrid = sorted(set(C.GRID_P) | {rule.p_thresh})
    ngrid = sorted(set(C.GRID_PERSIST) | {rule.persist})
    while len(ngrid) * len(vgrid) * len(pgrid) > MAX_GRID_CONFIGS and len(pgrid) > 1:
        pgrid = [p for p in pgrid if p != max(p for p in pgrid if p != rule.p_thresh)]
    rows, parts = [], []
    for n in ngrid:
        for v in vgrid:
            for p in pgrid:
                r = evaluate(prepared, Rule(f"N={n}, V={v}, P={p}", persist=n, v_thresh=v, p_thresh=p,
                                            persist_on=rule.persist_on, vel_window=rule.vel_window), latency_ms)
                r["N"], r["V"], r["P"] = n, v, p
                parts.append(r[[C.SUBJECT_COL, "trial_type", "pre_impact", "false_alarm", "eff_lead_ms", "N", "V", "P"]])
                s = summarize(r)
                rows.append({"N": n, "V": v, "P": p, "pre_impact_rate": _finite(s["pre_impact_rate"]),
                             "adl_false_alarm_rate": _finite(1 - s["specificity"]) if s["n_adl"] else None,
                             "lead_median_ms": _finite(s["lead_median_ms"])})
    return rows, pd.concat(parts, ignore_index=True), {"N": ngrid, "V": vgrid, "P": pgrid}


def _loso_thresholds(grid_trials, prepared, latency_ms, base_rule):
    keys = ["N", "V", "P"]
    subjects = sorted(grid_trials[C.SUBJECT_COL].unique())
    if len(subjects) < 2:
        return None
    chosen, picks = [], []
    for s in subjects:
        tr = grid_trials[grid_trials[C.SUBJECT_COL] != s]
        f = tr[tr["trial_type"] == "fall"].groupby(keys).agg(pre=("pre_impact", "mean"), lead=("eff_lead_ms", "median"))
        a = tr[tr["trial_type"] == "adl"].groupby(keys).agg(fa=("false_alarm", "mean"))
        st = f.join(a, how="outer").fillna({"pre": 0.0, "fa": 0.0, "lead": -np.inf})
        st["J"] = st["pre"] - st["fa"]
        best = st.sort_values(["J", "lead"], ascending=False).index[0]
        picks.append({"held_out_subject": s, "N": int(best[0]), "V": float(best[1]), "P": float(best[2])})
        rule = Rule("LOSO-selected thresholds", persist=int(best[0]), v_thresh=float(best[1]), p_thresh=float(best[2]),
                    persist_on=base_rule.persist_on, vel_window=base_rule.vel_window)
        chosen.append(evaluate([pt for pt in prepared if pt.trace.subject_id == s], rule, latency_ms))
    sel = pd.concat(chosen, ignore_index=True)
    picks = pd.DataFrame(picks)
    same = int(((picks["N"] == base_rule.persist) & np.isclose(picks["V"], base_rule.v_thresh)
                & np.isclose(picks["P"], base_rule.p_thresh)).sum())
    return {"row": summary_row(summarize(sel)), "picks": _records(picks), "same_as_chosen": same, "folds": len(picks)}


def analyze(frames_src, trials_src, preds_src, settings=None, include_grid=True) -> dict:
    t0 = time.perf_counter()
    used = apply_settings(settings)
    latency_ms = float((settings or {}).get("latency_ms") or 0.0)
    if latency_ms < 0 or latency_ms > 5000:
        raise ValueError("latency_ms must be between 0 and 5000")
    frames = load_frames(frames_src) if not isinstance(frames_src, pd.DataFrame) else frames_src
    trials = load_trials(trials_src, frames)
    preds = load_predictions_frame(preds_src) if not isinstance(preds_src, pd.DataFrame) else preds_src
    traces = build_traces(frames, preds, videos=set(trials[C.VIDEO_COL]))
    prepared = prepare_trials(trials, traces)
    if not prepared:
        raise ValueError("no trial matched a video in the frames file")
    rule = Rule()
    res = evaluate(prepared, rule, latency_ms)
    s = summarize(res)

    by_sc = []
    for sc, g in res.groupby("scenario"):
        row = summary_row(summarize(g), fa_rate=False)
        row["Configuration"] = sc
        by_sc.append(row)

    abl_rows, all_abl = [], []
    for r in ablation_rules(rule.persist):
        rr = evaluate(prepared, r, latency_ms)
        all_abl.append(rr)
        abl_rows.append((r.name, summarize(rr)))
    ref = all_abl[3].set_index("trial_id")
    fall = (ref["trial_type"] == "fall").to_numpy()
    ablation = []
    for (name, sm), rr in zip(abl_rows, all_abl):
        row = summary_row(sm)
        cur = rr.set_index("trial_id").loc[ref.index]
        if name.endswith("(proposed)"):
            row["McNemar p (pre-impact)"] = row["McNemar p (ADL FA)"] = "ref."
        else:
            row["McNemar p (pre-impact)"] = round(mcnemar_p(cur["pre_impact"].to_numpy()[fall], ref["pre_impact"].to_numpy()[fall]), 4)
            row["McNemar p (ADL FA)"] = round(mcnemar_p(cur["false_alarm"].to_numpy()[~fall], ref["false_alarm"].to_numpy()[~fall]), 4)
        ablation.append(row)

    out = {
        "settings": {**used, "latency_ms": latency_ms, "velocity_unit": C.VEL_UNITS[C.VELOCITY_MODE]},
        "meta": {"frames": int(len(frames)), "videos": int(frames[C.VIDEO_COL].nunique()),
                 "subjects": int(frames[C.SUBJECT_COL].nunique()), "trials": len(prepared),
                 "fall_trials": int(sum(pt.is_fall for pt in prepared)),
                 "adl_trials": int(sum(not pt.is_fall for pt in prepared)),
                 "videos_without_predictions": int(sum(np.isnan(tr.p).all() for tr in traces.values()))},
        "summary": _summary(s), "summary_row": summary_row(s),
        "by_scenario": by_sc, "ablation": ablation,
        "lead_times_ms": [round(float(v), 2) for v in res.loc[res["detected"], "eff_lead_ms"]],
        "per_trial": _records(res[["trial_id", C.VIDEO_COL, C.SUBJECT_COL, "trial_type", "scenario", "detected",
                                   "pre_impact", "lead_ms", "eff_lead_ms", "delay_from_onset_ms",
                                   "n_false_alarm_events", "alert_frame", "impact_frame", "onset_frame"]]),
        "examples": _examples(prepared, res, rule),
    }
    if include_grid:
        rows, grid_trials, axes = _grid(prepared, rule, latency_ms)
        out["grid"] = {"rows": rows, "axes": axes}
        out["loso_thresholds"] = _loso_thresholds(grid_trials, prepared, latency_ms, rule)
    out["runtime_s"] = round(time.perf_counter() - t0, 2)
    return out


def demo_inputs(n_subjects: int = 4, seed: int = 0):
    """Synthetic frames + trials + simulated GRU probabilities (for the 'try demo data' button)."""
    import make_demo_data as demo

    rng = np.random.default_rng(seed)
    frames, trials = [], []
    for si in range(n_subjects):
        S = demo.Subject(f"S{si + 1:02d}", rng)
        plan = [("fall", k) for k in demo.FALLS] + [("adl", k) for k in demo.ADLS]
        for j, (ttype, kind) in enumerate(plan):
            vid = f"{S.sid}_{kind}_{j:02d}"
            gen = demo.gen_fall if ttype == "fall" else demo.gen_adl
            poses, shifts, labels, onset, impact = gen(S, kind, rng)
            frames.append(demo.render(S, vid, poses, shifts, labels, rng))
            trials.append({"video_id": vid, "trial_type": ttype, "scenario": kind,
                           "onset_frame": onset, "impact_frame": impact})
    fr = pd.concat(frames, ignore_index=True)
    fr["subject_id"] = fr["subject_id"].astype(str)
    fr = fr.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    # simulated GRU: noisy baseline, short spurious spikes in fast ADLs, rising after loss-of-balance onset
    p = np.zeros(len(fr))
    for vid, idx in fr.groupby("video_id").indices.items():
        lab = fr["label"].to_numpy()[idx]
        n = len(idx)
        noise = pd.Series(rng.normal(0, 0.12, n)).rolling(3, min_periods=1).mean().to_numpy()
        pv = 0.15 + noise
        lob = np.flatnonzero(lab == "loss_of_balance")
        if len(lob):
            d = int(rng.integers(3, 8))
            ramp = np.clip((np.arange(len(lob)) - d) / 4.0, 0, 1)
            pv[lob] = 0.15 + 0.8 * ramp + noise[lob]
        if "fall" not in vid and rng.random() < 0.6:
            k = int(rng.integers(20, max(21, n - 5)))
            pv[k:k + int(rng.integers(1, 4))] += 0.6
        pv[: C.WINDOW - 1] = np.nan
        p[idx] = np.clip(pv, 0, 1)
    preds = pd.DataFrame({C.VIDEO_COL: fr[C.VIDEO_COL], C.FRAME_COL: fr[C.FRAME_COL], "p_pos": p}).dropna()
    keep = ["subject_id", "video_id", "frame_idx", "timestamp"] + [f"lm{i}_{a}" for i in (11, 12, 23, 24) for a in "xyv"]
    trials_csv = pd.DataFrame(trials).to_csv(index=False)
    return fr[keep], io.StringIO(trials_csv), preds
