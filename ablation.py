"""
Ablation of the decision rules and threshold sensitivity
(Reviewer 2 #5: contribution of the persistence / velocity rules;
 Reviewer 3: justification of the decision thresholds).

1) Ablation table - same GRU outputs, rules switched on/off:
     GRU only | GRU + persistence | GRU + velocity | GRU + persistence + velocity (proposed)
     | velocity only (threshold baseline, N=1 and N=PERSIST_FRAMES)
   with exact McNemar tests (paired, per trial) against the proposed rule.
2) Sensitivity grid over N (persistence), V (velocity threshold) and P (GRU threshold):
   heatmaps + trade-off plot, so the chosen thresholds can be shown to sit in a stable region.
3) --loso-thresholds: thresholds chosen on all OTHER subjects and applied to the held-out subject,
   i.e. a threshold choice that never sees the test subject.

Usage
  python ablation.py --frames data/frames.csv --preds results/loso/oof_predictions.csv \
                     --trials data/trials.csv --latency-ms 60 --loso-thresholds
Outputs (results/ablation/)
  ablation_table.md/.csv, ablation_per_trial.csv,
  grid_summary.csv, sensitivity_heatmaps.png, tradeoff.png,
  loso_thresholds.md, loso_threshold_picks.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

import config as C  # noqa: E402
from data import load_frames  # noqa: E402
from decision import (Rule, build_traces, evaluate, load_predictions, load_trials,  # noqa: E402
                      prepare_trials, summarize, summary_row)
from utils import md_table, save_json  # noqa: E402

UNIT = C.VEL_UNITS[C.VELOCITY_MODE]
PROPOSED = f"GRU + persistence (N={C.PERSIST_FRAMES}) + velocity (proposed)"


def ablation_rules():
    n = C.PERSIST_FRAMES
    return [
        Rule("GRU only (N=1)", persist=1, use_velocity=False),
        Rule(f"GRU + persistence (N={n})", use_velocity=False),
        Rule("GRU + velocity (N=1)", persist=1),
        Rule(PROPOSED),
        Rule("Velocity only (N=1)", use_gru=False, persist=1, persist_on="combined"),
        Rule(f"Velocity only (N={n})", use_gru=False, persist_on="combined"),
    ]


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar test on paired binary outcomes."""
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return 1.0 if n01 + n10 == 0 else float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def fmt_p(p):
    return "–" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


# ----------------------------------------------------------------------------
def run_ablation(prepared, latency, out):
    per_trial, rows = [], []
    for rule in ablation_rules():
        r = evaluate(prepared, rule, latency)
        per_trial.append(r)
        rows.append(summarize(r))
    allres = pd.concat(per_trial, ignore_index=True)
    allres.to_csv(out / "ablation_per_trial.csv", index=False)

    ref = allres[allres["rule"] == PROPOSED].set_index("trial_id")
    table = []
    for s in rows:
        row = summary_row(s)
        cur = allres[allres["rule"] == s["rule"]].set_index("trial_id").loc[ref.index]
        fall = (ref["trial_type"] == "fall").to_numpy()
        adl = ~fall
        if s["rule"] == PROPOSED:
            row["McNemar p (pre-impact)"] = row["McNemar p (ADL FA)"] = "ref."
        else:
            row["McNemar p (pre-impact)"] = fmt_p(mcnemar_p(cur["pre_impact"].to_numpy()[fall],
                                                            ref["pre_impact"].to_numpy()[fall]))
            row["McNemar p (ADL FA)"] = fmt_p(mcnemar_p(cur["false_alarm"].to_numpy()[adl],
                                                        ref["false_alarm"].to_numpy()[adl]))
        table.append(row)
    table = pd.DataFrame(table)
    pd.DataFrame(rows).to_csv(out / "ablation_summary.csv", index=False)
    table.to_csv(out / "ablation_table.csv", index=False)
    header = (f"# Ablation of the decision rules\n\n"
              f"Same subject-independent GRU outputs for every row. P = {C.P_THRESH}, "
              f"N = {C.PERSIST_FRAMES} frames, V = {C.V_THRESH} {UNIT} ({C.VELOCITY_MODE}), "
              f"latency subtracted = {latency:.0f} ms. McNemar: exact, paired per trial, vs. proposed.\n\n")
    (out / "ablation_table.md").write_text(header + md_table(table) + "\n", encoding="utf-8")
    print(header + md_table(table))


# ----------------------------------------------------------------------------
def run_grid(prepared, latency, out):
    vgrid = list(C.GRID_V[C.VELOCITY_MODE])
    rows, parts = [], []
    for n in C.GRID_PERSIST:
        for v in vgrid:
            for p in C.GRID_P:
                r = evaluate(prepared, Rule(f"N={n}, V={v}, P={p}", persist=n, v_thresh=v, p_thresh=p), latency)
                r["N"], r["V"], r["P"] = n, v, p
                parts.append(r[[C.SUBJECT_COL, C.VIDEO_COL, "trial_id", "trial_type", "pre_impact", "detected",
                                "false_alarm", "eff_lead_ms", "N", "V", "P"]])
                s = summarize(r)
                s.update(N=n, V=v, P=p)
                rows.append(s)
    grid = pd.DataFrame(rows)
    grid["adl_false_alarm_rate"] = 1 - grid["specificity"]
    grid.to_csv(out / "grid_summary.csv", index=False)
    plot_heatmaps(grid, vgrid, out / "sensitivity_heatmaps.png")
    plot_tradeoff(grid, out / "tradeoff.png")
    return pd.concat(parts, ignore_index=True)


def plot_heatmaps(grid, vgrid, path):
    p_used = min(C.GRID_P, key=lambda x: abs(x - C.P_THRESH))
    g = grid[grid["P"] == p_used]
    panels = [("pre_impact_rate", "Falls warned before impact (%)", 100, "Greens"),
              ("adl_false_alarm_rate", "ADL trials with false alarm (%)", 100, "Reds"),
              ("lead_median_ms", "Median lead time (ms)", 1, "Blues")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.45 * len(C.GRID_PERSIST) + 2.4))
    for ax, (col, title, scale, cmap) in zip(axes, panels):
        mat = g.pivot(index="N", columns="V", values=col).reindex(index=C.GRID_PERSIST, columns=vgrid) * scale
        arr = mat.to_numpy(float)
        ax.imshow(arr, cmap=cmap, aspect="auto")
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = arr[i, j]
                dark = not np.isnan(val) and hi > lo and (val - lo) / (hi - lo) > 0.6
                ax.text(j, i, "–" if np.isnan(val) else f"{val:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if dark else "black")
        ax.set_xticks(range(len(vgrid)), [f"{v:g}" for v in vgrid])
        ax.set_yticks(range(len(C.GRID_PERSIST)), C.GRID_PERSIST)
        ax.set_xlabel(f"velocity threshold V ({UNIT})")
        ax.set_ylabel("persistence N (frames)")
        ax.set_title(title, fontsize=10)
        if C.PERSIST_FRAMES in C.GRID_PERSIST and C.V_THRESH in vgrid:
            i, j = C.GRID_PERSIST.index(C.PERSIST_FRAMES), vgrid.index(C.V_THRESH)
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="black", lw=2.2))
    fig.suptitle(f"Threshold sensitivity at GRU threshold P = {p_used} (black box = thresholds used in the paper)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tradeoff(grid, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sc = ax.scatter(100 * grid["adl_false_alarm_rate"], 100 * grid["pre_impact_rate"],
                    c=grid["lead_median_ms"], cmap="viridis", s=18, alpha=0.75)
    fig.colorbar(sc, ax=ax, label="median lead time (ms)")
    prop = grid[(grid["N"] == C.PERSIST_FRAMES) & np.isclose(grid["V"], C.V_THRESH) & np.isclose(grid["P"], C.P_THRESH)]
    if len(prop):
        ax.scatter(100 * prop["adl_false_alarm_rate"], 100 * prop["pre_impact_rate"], marker="*", s=260,
                   color="red", ec="black", label=f"paper: N={C.PERSIST_FRAMES}, V={C.V_THRESH}, P={C.P_THRESH}")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("ADL trials with false alarm (%)")
    ax.set_ylabel("Falls warned before impact (%)")
    ax.set_title("Every (N, V, P) combination of the grid", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
def loso_thresholds(grid_trials, prepared, latency, out):
    """For each subject: pick (N, V, P) maximising J = pre-impact rate - ADL false-alarm rate on the
    other subjects (ties: larger median lead), then apply it to the held-out subject."""
    keys = ["N", "V", "P"]
    chosen, picks = [], []
    for s in sorted(grid_trials[C.SUBJECT_COL].unique()):
        tr = grid_trials[grid_trials[C.SUBJECT_COL] != s]
        f = tr[tr["trial_type"] == "fall"].groupby(keys).agg(pre=("pre_impact", "mean"),
                                                            lead=("eff_lead_ms", "median"))
        a = tr[tr["trial_type"] == "adl"].groupby(keys).agg(fa=("false_alarm", "mean"))
        st = f.join(a, how="outer").fillna({"pre": 0.0, "fa": 0.0, "lead": -np.inf})
        st["J"] = st["pre"] - st["fa"]
        best = st.sort_values(["J", "lead"], ascending=False).index[0]
        picks.append({"held_out_subject": s, "N": best[0], "V": best[1], "P": best[2],
                      "J_on_training_subjects": round(float(st.loc[best, "J"]), 4)})
        rule = Rule("LOSO-selected thresholds", persist=int(best[0]), v_thresh=float(best[1]), p_thresh=float(best[2]))
        chosen.append(evaluate([pt for pt in prepared if pt.trace.subject_id == s], rule, latency))
    picks = pd.DataFrame(picks)
    picks.to_csv(out / "loso_threshold_picks.csv", index=False)
    sel = pd.concat(chosen, ignore_index=True)
    fixed = evaluate(prepared, Rule(PROPOSED), latency)
    table = pd.DataFrame([summary_row(summarize(fixed)), summary_row(summarize(sel))])
    freq = picks.groupby(["N", "V", "P"]).size().rename("times chosen").reset_index() \
        .sort_values("times chosen", ascending=False)
    n_same = int(((picks["N"] == C.PERSIST_FRAMES) & np.isclose(picks["V"], C.V_THRESH)
                  & np.isclose(picks["P"], C.P_THRESH)).sum())
    text = ("# Thresholds selected without the test subject (LOSO)\n\n"
            "For each held-out subject, (N, V, P) maximising J = pre-impact detection rate - ADL false-alarm "
            "rate on the remaining subjects was applied to that subject.\n\n"
            f"{md_table(table)}\n\nThe paper's thresholds were selected in {n_same}/{len(picks)} folds.\n\n"
            f"## How often each combination was chosen\n\n{md_table(freq)}\n\n"
            f"## Per held-out subject\n\n{md_table(picks)}\n")
    (out / "loso_thresholds.md").write_text(text, encoding="utf-8")
    save_json(summarize(sel), out / "loso_thresholds_summary.json")
    print("\n" + text.split("## How often")[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--loso-thresholds", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.frames)
    trials = load_trials(args.trials, frames)
    traces = build_traces(frames, load_predictions(args.preds), videos=set(trials[C.VIDEO_COL]))
    prepared = prepare_trials(trials, traces)
    print(f"{sum(pt.is_fall for pt in prepared)} fall trials, {sum(not pt.is_fall for pt in prepared)} ADL trials")

    run_ablation(prepared, args.latency_ms, out)
    if not args.no_grid or args.loso_thresholds:
        grid_trials = run_grid(prepared, args.latency_ms, out)
        if args.loso_thresholds:
            loso_thresholds(grid_trials, prepared, args.latency_ms, out)
    print(f"\noutputs in {out}/")


if __name__ == "__main__":
    main()
