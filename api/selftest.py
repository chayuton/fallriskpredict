"""
Vercel serverless self-test: runs the decision rule (GRU + persistence + velocity) and the lead-time
evaluation end-to-end on synthetic MediaPipe-like data. No TensorFlow: the "GRU" probability is the
ground-truth loss-of-balance label, so this checks the pipeline, not the model.

GET /api/selftest?subjects=2
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def run(n_subjects: int = 2) -> dict:
    import numpy as np
    import pandas as pd

    import config as C
    import make_demo_data as demo
    from data import hip_descent_velocity
    from decision import Rule, build_traces, evaluate, load_trials, prepare_trials, summarize

    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
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

    # oracle "GRU": probability 1 on loss_of_balance frames (from the 15th frame of each video on)
    preds = fr[["video_id", "frame_idx"]].copy()
    preds["p_pos"] = (fr["label"] == "loss_of_balance").astype(float)
    preds.loc[fr.groupby("video_id").cumcount() < C.WINDOW - 1, "p_pos"] = np.nan

    tr_path = "/tmp/selftest_trials.csv"
    pd.DataFrame(trials).to_csv(tr_path, index=False)
    tr = load_trials(tr_path, fr)
    traces = build_traces(fr, preds.dropna())
    res = evaluate(prepare_trials(tr, traces), Rule())
    s = summarize(res)
    v = np.concatenate([hip_descent_velocity(g) for _, g in fr.groupby("video_id")])
    checks = {
        "frames_generated": int(len(fr)) > 0,
        "velocity_finite": bool(np.isfinite(v).sum() > 0.9 * len(v)),
        "all_falls_evaluated": s["n_falls"] == len(demo.FALLS) * n_subjects,
        "lead_time_computed": bool(np.isfinite(s["lead_median_ms"])) if s["n_detected"] else False,
    }
    return {
        "status": "ok" if all(checks.values()) else "fail",
        "checks": checks,
        "note": "oracle GRU (labels) on synthetic data: tests the code path, not model accuracy",
        "summary": {k: (None if isinstance(x, float) and not np.isfinite(x) else x) for k, x in s.items()},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "runtime_s": round(time.perf_counter() - t0, 2),
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        try:
            n = max(1, min(int(q.get("subjects", ["2"])[0]), 4))
            body, code = run(n), 200
        except Exception as e:  # noqa: BLE001
            import traceback
            body, code = {"status": "error", "error": repr(e), "trace": traceback.format_exc()[-2000:]}, 500
        data = json.dumps(body, ensure_ascii=False, indent=2, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)
