"""Quick checks of the web app back end:  python -m pytest tests  (or python tests/test_webapp.py)"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.analysis import analyze, apply_settings, demo_inputs, mcnemar_p  # noqa: E402

import numpy as np  # noqa: E402


def test_demo_analysis():
    fr, tr, pr = demo_inputs(2)
    out = analyze(fr, tr, pr, {"latency_ms": 30})
    assert out["meta"]["fall_trials"] == 8 and out["meta"]["adl_trials"] == 12
    assert out["summary"]["n_falls"] == 8
    assert len(out["ablation"]) == 6 and out["ablation"][3]["McNemar p (ADL FA)"] == "ref."
    assert out["grid"]["rows"] and out["loso_thresholds"]["folds"] == 2


def test_settings_reset_between_requests():
    apply_settings({"v_thresh": 9.9, "velocity_mode": "torso"})
    used = apply_settings({})
    assert used["v_thresh"] == 1.5 and used["velocity_mode"] == "meters"


def test_bad_setting_rejected():
    try:
        apply_settings({"p_thresh": 3})
    except ValueError:
        return
    raise AssertionError("p_thresh=3 should be rejected")


def test_mcnemar():
    a = np.array([True] * 10 + [False] * 10)
    assert mcnemar_p(a, a) == 1.0
    assert mcnemar_p(np.ones(10, bool), np.zeros(10, bool)) < 0.01


def test_csv_input():
    fr, tr, pr = demo_inputs(1)
    out = analyze(io.StringIO(fr.to_csv(index=False)), tr, io.StringIO(pr.to_csv(index=False)), include_grid=False)
    assert "grid" not in out and out["summary"]["n_falls"] == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
