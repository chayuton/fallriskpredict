"""
Per-frame latency, throughput (FPS), memory and CPU of the full pipeline on the target device
(Reviewer 1 #2: FPS / memory on Raspberry Pi 5 or Jetson; Reviewer 3: processing latency).
Run it ON the device, with the same MediaPipe settings, resolution and model as the real system.

Stages timed per frame:  read/decode -> pose (MediaPipe) -> features + buffer -> GRU -> decision rule
The p95 of the processing latency is a conservative value for  lead_time.py --latency-ms.

Usage
  python benchmark_edge.py --video test.mp4 --model models/loso/S01.keras --pose-model pose_landmarker_full.task
  python benchmark_edge.py --camera 0 --model my_model.keras --n-frames 900            # live camera
  python benchmark_edge.py --model my_model.keras --dummy-pose                         # GRU + rule only
Pose backend: --pose-model <.task> uses the MediaPipe Tasks API (needed for mediapipe >= 0.10.30, which has
no mp.solutions); without it the legacy mp.solutions.pose API is used if your MediaPipe still has it.
Outputs (results/benchmark/): <tag>_frames.csv, <tag>_summary.json, <tag>_summary.md
"""
from __future__ import annotations

import argparse
import glob
import os
import platform
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

import config as C  # noqa: E402
from utils import md_table, safe_name, save_json  # noqa: E402

N_LM = 33


# ----------------------------------------------------------------------------
# pose backends
# ----------------------------------------------------------------------------
class Pose:
    def __init__(self, args):
        self.kind = "dummy"
        if args.dummy_pose:
            self.rng = np.random.default_rng(0)
            return
        import mediapipe as mp
        self.mp = mp
        if args.pose_model:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            opts = vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=args.pose_model),
                                                running_mode=vision.RunningMode.VIDEO, num_poses=1)
            self.landmarker = vision.PoseLandmarker.create_from_options(opts)
            self.kind = f"tasks:{Path(args.pose_model).name}"
        elif hasattr(mp, "solutions"):
            self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=args.model_complexity,
                                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.kind = f"legacy:model_complexity={args.model_complexity}"
        else:
            raise SystemExit("This MediaPipe has no mp.solutions: pass --pose-model pose_landmarker_*.task "
                             "(download from the MediaPipe Pose Landmarker page) or use --dummy-pose")

    def close(self):
        for attr in ("landmarker", "pose"):
            if hasattr(self, attr):
                getattr(self, attr).close()

    def __call__(self, frame_bgr, ts_ms):
        """Returns (33, 4) array of x, y, z, visibility, or None if no person was found."""
        if self.kind == "dummy":
            lms = np.c_[self.rng.uniform(0.3, 0.7, (N_LM, 3)), np.full(N_LM, 0.95)]
            return lms
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.kind.startswith("tasks"):
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect_for_video(img, int(ts_ms))
            if not res.pose_landmarks:
                return None
            return np.array([(p.x, p.y, p.z, 1.0 if p.visibility is None else p.visibility)
                             for p in res.pose_landmarks[0]], np.float32)
        res = self.pose.process(rgb)
        if res.pose_landmarks is None:
            return None
        return np.array([(p.x, p.y, p.z, p.visibility) for p in res.pose_landmarks.landmark], np.float32)


def features_from_landmarks(lms):
    """>>> EDIT so the feature vector matches training exactly (order: lm0_x, lm0_y, lm0_z, lm0_v, lm1_x, ...)."""
    if lms is None:
        return np.zeros(N_LM * 4, np.float32)   # same convention as your extraction code?
    return lms.reshape(-1).astype(np.float32)


# ----------------------------------------------------------------------------
# GRU backends
# ----------------------------------------------------------------------------
class GRU:
    def __init__(self, path, scaler, use_predict):
        self.scaler = np.load(scaler) if scaler else None
        if path.endswith(".tflite"):
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite import Interpreter
            self.it = Interpreter(model_path=path, num_threads=os.cpu_count())
            self.it.allocate_tensors()
            self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]
            self.kind = "tflite"
        else:
            import tensorflow as tf
            from tensorflow import keras
            self.model = keras.models.load_model(path, compile=False)
            # a traced tf.function is much faster per frame than eager model(x) or model.predict(x)
            self.fn = tf.function(lambda x: self.model(x, training=False), reduce_retracing=True)
            self.kind = "keras.predict" if use_predict else "keras+tf.function"

    def __call__(self, x):
        if self.scaler is not None:
            x = (x - self.scaler["mean"]) / self.scaler["std"]
        x = x[None].astype(np.float32)
        if self.kind == "tflite":
            self.it.set_tensor(self.inp["index"], x)
            self.it.invoke()
            return self.it.get_tensor(self.out["index"])[0]
        if self.kind == "keras.predict":
            return self.model.predict(x, verbose=0)[0]
        return np.asarray(self.fn(x))[0]


# ----------------------------------------------------------------------------
# streaming version of the decision rule (same logic as decision.py)
# ----------------------------------------------------------------------------
class StreamingRule:
    def __init__(self):
        self.y = deque(maxlen=C.VEL_LAG_FRAMES + 1)
        self.t = deque(maxlen=C.VEL_LAG_FRAMES + 1)
        self.v = deque(maxlen=max(1, C.VEL_WINDOW))
        self.torso = deque(maxlen=C.TORSO_REF_WIN + C.TORSO_REF_LAG)
        self.run = 0
        self.pos = [C.CLASS_NAMES.index(c) for c in C.POSITIVE_CLASSES]

    def step(self, lms, t, prob):
        y = np.nan
        if lms is not None and min(lms[23, 3], lms[24, 3]) >= C.MIN_VISIBILITY:
            y = float(lms[23, 1] + lms[24, 1]) / 2
        self.y.append(y)
        self.t.append(t)
        v = np.nan
        if len(self.y) == self.y.maxlen:
            v = (self.y[-1] - self.y[0]) / (self.t[-1] - self.t[0])
        if C.VELOCITY_MODE == "meters":
            v *= C.FRAME_H_METERS
        elif C.VELOCITY_MODE == "pixels":
            v *= C.FRAME_H
        else:
            if lms is not None:
                sx, sy = (lms[11, 0] + lms[12, 0]) / 2, (lms[11, 1] + lms[12, 1]) / 2
                hx, hy = (lms[23, 0] + lms[24, 0]) / 2, (lms[23, 1] + lms[24, 1]) / 2
                self.torso.append(np.hypot((sx - hx) * C.FRAME_W, (sy - hy) * C.FRAME_H))
            hist = list(self.torso)
            ref_part = hist[:len(hist) - C.TORSO_REF_LAG] if len(hist) > C.TORSO_REF_LAG else hist
            ref = np.median(ref_part[-C.TORSO_REF_WIN:]) if ref_part else np.nan
            v = v * C.FRAME_H / ref if ref and ref > 1e-6 else np.nan
        self.v.append(v)
        vel_ok = np.nanmax(self.v) >= C.V_THRESH if np.isfinite(self.v).any() else False
        g = prob is not None and float(np.sum(prob[self.pos])) >= C.P_THRESH
        if C.PERSIST_ON == "combined":
            self.run = self.run + 1 if (g and vel_ok) else 0
            return self.run >= C.PERSIST_FRAMES
        self.run = self.run + 1 if g else 0
        return self.run >= C.PERSIST_FRAMES and vel_ok


# ----------------------------------------------------------------------------
def device_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "ram_total_mb": round(psutil.virtual_memory().total / 2 ** 20)}
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        if os.path.exists(p):
            info["board"] = Path(p).read_text(errors="ignore").strip("\x00\n ")
            break
    for mod in ("tensorflow", "mediapipe", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return info


def max_temp_c():
    vals = []
    for f in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            vals.append(int(Path(f).read_text().strip()) / 1000.0)
        except Exception:  # noqa: BLE001
            pass
    return max(vals) if vals else None


def stats(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {"mean": x.mean(), "median": np.median(x), "p95": np.percentile(x, 95), "max": x.max()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--model", required=True, help=".keras / .h5 / .tflite")
    ap.add_argument("--scaler", help=".npz with mean/std if the model was trained on standardised features")
    ap.add_argument("--pose-model", help="MediaPipe Tasks pose_landmarker_{lite,full,heavy}.task")
    ap.add_argument("--model-complexity", type=int, default=1, help="legacy API only: 0 lite, 1 full, 2 heavy")
    ap.add_argument("--dummy-pose", action="store_true", help="skip MediaPipe (measures GRU + rule only)")
    ap.add_argument("--keras-predict", action="store_true",
                    help="use model.predict() per frame (slower; use it if your system does)")
    ap.add_argument("--n-frames", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--tag", default=None, help="name for the output files, e.g. rpi5")
    ap.add_argument("--out", default="results/benchmark")
    args = ap.parse_args()

    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 2 ** 20
    pose = Pose(args)
    gru = GRU(args.model, args.scaler, args.keras_predict)
    rss_loaded = proc.memory_info().rss / 2 ** 20

    cap = None
    if args.video is not None or args.camera is not None:
        import cv2
        cap = cv2.VideoCapture(args.video if args.video is not None else args.camera)
        if not cap.isOpened():
            raise SystemExit("cannot open video source")
        fps = cap.get(cv2.CAP_PROP_FPS) or C.FPS
    elif not args.dummy_pose:
        raise SystemExit("give --video or --camera (or --dummy-pose)")
    else:
        fps = C.FPS
    blank = np.zeros((C.FRAME_H, C.FRAME_W, 3), np.uint8)

    rule = StreamingRule()
    buf = deque(maxlen=C.WINDOW)
    rows, temp_start = [], max_temp_c()
    proc.cpu_percent(None)
    psutil.cpu_percent(None)
    t_start = time.perf_counter()
    i = 0
    while i < args.n_frames + args.warmup:
        t0 = time.perf_counter()
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                if args.video is not None and i < args.n_frames + args.warmup:
                    cap.set(1, 0)   # loop the file
                    ok, frame = cap.read()
                if not ok:
                    break
        else:
            frame = blank
        t1 = time.perf_counter()
        lms = pose(frame, i * 1000.0 / fps)
        t2 = time.perf_counter()
        buf.append(features_from_landmarks(lms))
        x = np.stack(buf) if len(buf) == C.WINDOW else None
        t3 = time.perf_counter()
        prob = gru(x) if x is not None else None
        t4 = time.perf_counter()
        alert = rule.step(lms, i / fps, prob)
        t5 = time.perf_counter()
        rows.append({"frame": i, "warmup": i < args.warmup, "read_ms": (t1 - t0) * 1e3, "pose_ms": (t2 - t1) * 1e3,
                     "features_ms": (t3 - t2) * 1e3, "gru_ms": (t4 - t3) * 1e3 if x is not None else np.nan,
                     "decision_ms": (t5 - t4) * 1e3, "processing_ms": (t5 - t1) * 1e3,
                     "frame_total_ms": (t5 - t0) * 1e3, "person_found": lms is not None, "alert": bool(alert),
                     "rss_mb": proc.memory_info().rss / 2 ** 20})
        i += 1
    wall = time.perf_counter() - t_start
    cpu_proc, cpu_sys = proc.cpu_percent(None), psutil.cpu_percent(None)

    df = pd.DataFrame(rows)
    m = df[~df["warmup"] & df["gru_ms"].notna()]
    tag = safe_name(args.tag or platform.node() or "device")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{tag}_frames.csv", index=False)
    summary = {
        "device": device_info(), "pose_backend": pose.kind, "gru_backend": gru.kind,
        "frames_measured": len(m), "video_fps": fps,
        "throughput_fps": len(m) / (m["frame_total_ms"].sum() / 1e3) if len(m) else None,
        "processing_only_fps": 1e3 / m["processing_ms"].mean() if len(m) else None,
        "overall_fps_including_warmup": len(df) / wall,
        "latency_ms": {k: stats(m[k]) for k in ["read_ms", "pose_ms", "features_ms", "gru_ms",
                                                "decision_ms", "processing_ms", "frame_total_ms"]},
        "rss_mb": {"before_models": rss0, "after_loading_models": rss_loaded,
                   "mean_during_run": m["rss_mb"].mean() if len(m) else None,
                   "peak_during_run": df["rss_mb"].max()},
        "cpu_percent_process_mean": cpu_proc, "cpu_percent_system_mean": cpu_sys,
        "temperature_c": {"start": temp_start, "end": max_temp_c()},
        "person_found_rate": float(m["person_found"].mean()) if len(m) else None,
    }
    save_json(summary, out / f"{tag}_summary.json")
    lat = summary["latency_ms"]
    table = pd.DataFrame([{"Stage": k.replace("_ms", ""), "Mean (ms)": f"{v['mean']:.2f}",
                           "Median (ms)": f"{v['median']:.2f}", "p95 (ms)": f"{v['p95']:.2f}",
                           "Max (ms)": f"{v['max']:.2f}"} for k, v in lat.items() if v["mean"] is not None])
    dev = summary["device"]
    text = (f"# Edge benchmark: {dev.get('board', dev['platform'])}\n\n"
            f"Pose: {pose.kind}; GRU: {gru.kind}; frames measured: {len(m)} (after {args.warmup} warm-up)\n\n"
            f"{md_table(table)}\n\n"
            f"- Throughput: {summary['throughput_fps']:.1f} FPS (processing only: {summary['processing_only_fps']:.1f} FPS)\n"
            f"- Memory (RSS): {rss_loaded:.0f} MB after loading models, peak {summary['rss_mb']['peak_during_run']:.0f} MB\n"
            f"- CPU: process {cpu_proc:.0f}% (100% = one core), system {cpu_sys:.0f}%\n"
            f"- Temperature: {summary['temperature_c']['start']} -> {summary['temperature_c']['end']} °C\n\n"
            f"For lead_time.py use --latency-ms {lat['processing_ms']['p95']:.0f} (p95 processing latency; "
            f"camera capture delay and alarm/LINE delivery are not included).\n")
    (out / f"{tag}_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
