"""
Pre-impact warning lead time (Reviewer 1 #2, Reviewer 2 #2, Reviewer 3).

    lead time           = t_impact - t_first_alert      (ms; > 0 = warning BEFORE impact)
    effective lead time = lead time - processing latency (--latency-ms, from benchmark_edge.py)

The decision rule is the one in config.py (GRU + persistence + velocity). The GRU probabilities should
be subject-independent: use results/loso/oof_predictions.csv, or predict_trials.py output for
separately recorded trials.

Step 1 (optional) - help annotating impact frames:
  python lead_time.py --frames data/frames.csv --trials data/trials.csv --suggest-impact
  -> results/lead_time/trials_suggested.csv + suggest_plots/*.png  (CHECK EVERY ONE BY EYE)

Step 2 - measure:
  python lead_time.py --frames data/frames.csv --preds results/loso/oof_predictions.csv \
                      --trials data/trials.csv --latency-ms 60

Outputs (results/lead_time/)
  per_trial.csv, summary.json, summary.md, by_scenario.csv,
  lead_time_hist.png, lead_time_by_scenario.png, timelines/*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from data import load_frames  # noqa: E402
from decision import (Rule, alert_mask, build_traces, evaluate, load_predictions, load_trials,  # noqa: E402
                      prepare_trials, summarize, summary_row)
from utils import md_table, safe_name, save_json  # noqa: E402

UNIT = C.VEL_UNITS[C.VELOCITY_MODE]


# ----------------------------------------------------------------------------
# impact suggestion (offline helper for annotation - NOT used for the evaluation itself)
# ----------------------------------------------------------------------------
def suggest_impact(tr, smooth: int = 3, frac: float = 0.5):
    """
    Rough impact suggestion: first frame after the peak downward hip velocity at which the velocity has
    dropped below `frac` of the peak (the body is being stopped by the mat / floor).
    Always confirm by stepping through the video frame by frame.
    """
    if len(tr.t) < 3:
        return np.nan
    y = pd.Series(tr.hip_y).interpolate(limit_direction="both")
    if y.isna().all():
        return np.nan
    y = y.rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    v = np.gradient(y, tr.t)
    p = int(np.argmax(v))
    if v[p] <= 0:
        return np.nan
    after = np.flatnonzero(v[p:] <= frac * v[p])
    return int(tr.frame[p + int(after[0])]) if len(after) else int(tr.frame[p])


def run_suggest(trials, traces, out):
    from decision import slice_trace
    plot_dir = out / "suggest_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    trials = trials.copy()
    trials["impact_frame_suggested"] = np.nan
    for i, row in trials.iterrows():
        if row["trial_type"] != "fall" or row[C.VIDEO_COL] not in traces:
            continue
        tr = slice_trace(traces[row[C.VIDEO_COL]], row.get("start_frame"), row.get("end_frame"))
        if pd.notna(row.get("onset_frame")):
            tr = slice_trace(tr, row["onset_frame"], None)   # impact comes after the annotated onset
        imp = suggest_impact(tr)
        trials.loc[i, "impact_frame_suggested"] = imp
        fig, ax = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
        ax[0].plot(tr.frame, 1 - tr.hip_y, lw=1.2)
        ax[0].set_ylabel("hip height\n(1 - y_norm)")
        ax[1].plot(tr.frame, tr.v, lw=1.0, color="tab:orange")
        ax[1].set_ylabel(f"hip descent\nvelocity ({UNIT})")
        ax[1].set_xlabel("frame")
        for a in ax:
            if not np.isnan(imp):
                a.axvline(imp, color="red", ls="--", lw=1, label=f"suggested impact {imp}")
            if pd.notna(row["impact_frame"]):
                a.axvline(row["impact_frame"], color="k", lw=1, label=f"annotated impact {int(row['impact_frame'])}")
        ax[0].legend(fontsize=7, loc="lower left")
        ax[0].set_title(f"{row['trial_id']} ({row['scenario']}) - verify against the video", fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_name(row['trial_id'])}.png", dpi=110)
        plt.close(fig)
    path = out / "trials_suggested.csv"
    trials.to_csv(path, index=False)
    print(f"wrote {path} and {plot_dir}/ - check each suggestion frame-by-frame in the video, then copy the "
          f"verified values into the impact_frame column of your trials file")


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def plot_hist(res, path, latency_ms):
    f = res[res["trial_type"] == "fall"]
    d = f[f["detected"]]
    if d.empty:
        return
    lead = d["eff_lead_ms"].to_numpy(float)
    step = 1000.0 / C.FPS
    lo, hi = C.LEAD_TARGET_MS
    bins = np.arange(np.floor(lead.min() / step) * step - step / 2, lead.max() + step, step)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(lo, hi, color="tab:green", alpha=0.12, label=f"target {lo}-{hi} ms")
    ax.hist(lead, bins=bins, color="tab:blue", edgecolor="white")
    ax.axvline(0, color="k", ls="--", lw=1, label="impact")
    ax.axvline(np.median(lead), color="tab:red", lw=1.2, label=f"median {np.median(lead):.0f} ms")
    ax.set_xlabel("Warning lead time before impact (ms)" + (f", latency {latency_ms:.0f} ms subtracted" if latency_ms else ""))
    ax.set_ylabel("Number of falls")
    ax.set_title(f"{len(d)}/{len(f)} falls detected, {int(d['pre_impact'].sum())} before impact", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_by_scenario(res, path):
    d = res[(res["trial_type"] == "fall") & res["detected"]]
    if d.empty:
        return
    groups = [(s, g["eff_lead_ms"].to_numpy(float)) for s, g in d.groupby("scenario")]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(groups) + 1.6))
    try:
        ax.boxplot([g for _, g in groups], orientation="horizontal", widths=0.5, showfliers=False)
    except TypeError:  # Matplotlib < 3.10
        ax.boxplot([g for _, g in groups], vert=False, widths=0.5, showfliers=False)
    rng = np.random.default_rng(0)
    for i, (_, g) in enumerate(groups, 1):
        ax.scatter(g, i + rng.uniform(-0.12, 0.12, len(g)), s=12, alpha=0.7, color="tab:blue", zorder=3)
    ax.set_yticks(range(1, len(groups) + 1))
    ax.set_yticklabels([s for s, _ in groups])
    lo, hi = C.LEAD_TARGET_MS
    ax.axvspan(lo, hi, color="tab:green", alpha=0.12)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Warning lead time before impact (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_timeline(pt, row, rule, path, latency_ms=0.0):
    tr = pt.trace
    x = (tr.t - pt.t_impact) * 1000.0
    alert = alert_mask(tr.p, tr.v, rule)
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 6), sharex=True)
    ax[0].plot(x, 1 - tr.hip_y, lw=1.2)
    ax[0].set_ylabel("hip height\n(1 - y_norm)")
    ax[1].plot(x, tr.v, lw=1.0, color="tab:orange")
    ax[1].axhline(rule.v_thresh, color="tab:orange", ls="--", lw=0.8)
    ax[1].set_ylabel(f"descent velocity\n({UNIT})")
    ax[2].plot(x, tr.p, lw=1.0, color="tab:purple")
    ax[2].axhline(rule.p_thresh, color="tab:purple", ls="--", lw=0.8)
    ax[2].fill_between(x, 0, 1, where=alert, color="red", alpha=0.15, step="mid", label="alert condition")
    ax[2].set_ylabel("GRU p(warning)")
    ax[2].set_ylim(-0.02, 1.02)
    ax[2].set_xlabel("time relative to impact (ms)")
    for a in ax:
        a.axvline(0, color="k", lw=1)
        if not np.isnan(pt.t_onset):
            a.axvline((pt.t_onset - pt.t_impact) * 1000, color="gray", ls=":", lw=1)
        if row["detected"]:
            a.axvline(-row["lead_ms"], color="red", lw=1.2)
    lead_txt = "missed"
    if row["detected"]:
        lead_txt = f"lead {row['lead_ms']:.0f} ms" + (f" (effective {row['eff_lead_ms']:.0f} ms)" if latency_ms else "")
    ax[0].set_title(f"{row['trial_id']} ({row['scenario']}): {lead_txt}  "
                    f"[black = impact, red = first alert, gray = onset]", fontsize=9)
    ax[2].legend(fontsize=7, loc="upper left")
    lo = -max(C.MAX_PRE_IMPACT_S * 1000, 1500)
    ax[2].set_xlim(lo, C.MAX_POST_IMPACT_S * 1000)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_examples(res, n):
    f = res[res["trial_type"] == "fall"]
    d = f[f["detected"]].sort_values("lead_ms")
    picks = []
    if len(d):
        q = np.unique(np.round(np.linspace(0, len(d) - 1, max(1, n - 1))).astype(int))
        picks += d.iloc[q]["trial_id"].tolist()
    picks += f[~f["detected"]]["trial_id"].tolist()[:1]
    return picks[:n]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--preds", help="oof_predictions.csv or predict_trials.py output")
    ap.add_argument("--latency-ms", type=float, default=0.0,
                    help="end-to-end processing latency per frame (benchmark_edge.py), subtracted from lead time")
    ap.add_argument("--out", default="results/lead_time")
    ap.add_argument("--suggest-impact", action="store_true")
    ap.add_argument("--n-timelines", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.frames)
    trials = load_trials(args.trials, frames, require_impact=not args.suggest_impact)
    preds = load_predictions(args.preds) if args.preds else None
    traces = build_traces(frames, preds, videos=set(trials[C.VIDEO_COL]))

    if args.suggest_impact:
        run_suggest(trials, traces, out)
        return
    if preds is None:
        ap.error("--preds is required to measure lead time")

    rule = Rule()
    prepared = prepare_trials(trials, traces)
    res = evaluate(prepared, rule, args.latency_ms)
    res.to_csv(out / "per_trial.csv", index=False)
    s = summarize(res)
    s.update(latency_ms=args.latency_ms, velocity_mode=C.VELOCITY_MODE, p_thresh=rule.p_thresh,
             persist=rule.persist, v_thresh=rule.v_thresh, fps=C.FPS)
    save_json(s, out / "summary.json")

    by_sc = []
    for sc, g in res.groupby("scenario"):
        row = summary_row(summarize(g), fa_rate=False)
        row["Configuration"] = sc
        by_sc.append(row)
    by_sc = pd.DataFrame(by_sc).rename(columns={"Configuration": "Scenario"})
    by_sc.to_csv(out / "by_scenario.csv", index=False)

    d = res[(res["trial_type"] == "fall") & res["detected"]]
    lo, hi = C.LEAD_TARGET_MS
    lines = [
        "# Pre-impact warning lead time",
        "",
        f"Decision rule: GRU p(warning) ≥ {rule.p_thresh} for {rule.persist} consecutive frames "
        f"(persistence on '{rule.persist_on}') AND hip descent velocity ≥ {rule.v_thresh} {UNIT}.",
        f"Frame rate {C.FPS:g} fps -> lead-time resolution ±{1000 / C.FPS:.1f} ms (one frame).",
        f"Processing latency subtracted: {args.latency_ms:.1f} ms.",
        "",
        md_table(pd.DataFrame([summary_row(s)])),
        "",
    ]
    if len(d):
        lead = d["eff_lead_ms"]
        lines += [
            f"- Effective lead time over {len(d)} detected falls: mean ± SD = {lead.mean():.0f} ± "
            f"{(lead.std(ddof=1) if len(d) > 1 else 0):.0f} ms, median {lead.median():.0f} ms "
            f"[IQR {lead.quantile(.25):.0f}-{lead.quantile(.75):.0f}], range {lead.min():.0f} to {lead.max():.0f} ms.",
            f"- Falls warned ≥ {lo} ms before impact: {100 * s['rate_lead_ge_target_min']:.1f}% of all falls; "
            f"within {lo}-{hi} ms: {100 * s['rate_lead_in_target']:.1f}%.",
        ]
        if d["delay_from_onset_ms"].notna().any():
            lines.append(f"- Median delay from annotated loss-of-balance onset to alert: "
                         f"{d['delay_from_onset_ms'].median():.0f} ms.")
    lines += ["", "## By scenario", "", md_table(by_sc), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    plot_hist(res, out / "lead_time_hist.png", args.latency_ms)
    plot_by_scenario(res, out / "lead_time_by_scenario.png")
    tl_dir = out / "timelines"
    tl_dir.mkdir(exist_ok=True)
    by_tid = {pt.row["trial_id"]: pt for pt in prepared}
    for tid in pick_examples(res, args.n_timelines):
        row = res[res["trial_id"] == tid].iloc[0]
        plot_timeline(by_tid[tid], row, rule, tl_dir / f"{safe_name(tid)}.png", args.latency_ms)
    print("\n".join(lines[:8]))
    print(f"\noutputs in {out}/")


if __name__ == "__main__":
    main()
