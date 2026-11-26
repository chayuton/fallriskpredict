"""
Convert open fall datasets into the frames.csv / trials.csv format used by loso.py, lead_time.py and ablation.py.

Label sources
  omnifall : OmniFall label CSVs (labels/<dataset>.csv of huggingface.co/datasets/simplexsigil2/omnifall),
             one harmonised label set with subject ids for CMDFall, UP-Fall, Le2i, GMDCSA24, CAUCAFall,
             EDF, OCCU, MCFD and OOPS-Fall. The videos themselves come from each original dataset.
  urfd     : UR Fall Detection per-frame labels (urfall-cam0-falls.csv, urfall-cam0-adls.csv)
  other    : write segments.csv + manifest.csv yourself (columns below) and start at "extract"

Steps
  1) labels  -> <out>/manifest.csv (video_id, subject_id, dataset, media) + <out>/segments.csv
       python prepare_open_data.py omnifall --labels labels/up_fall.csv --video-root /data/UP-Fall --out data/upfall
       python prepare_open_data.py urfd --falls urfall-cam0-falls.csv --adls urfall-cam0-adls.csv \
                                        --video-root /data/URFD --out data/urfd
  2) MediaPipe landmarks for every video (resumable: one file per video in <out>/parts/), then frames.csv
       python prepare_open_data.py extract --dir data/upfall --pose-model pose_landmarker_full.task
  3) (optional) resample to the frame rate the GRU was trained on (GRU window = 15 frames)
       python prepare_open_data.py resample --frames data/upfall/frames.csv --fps 30
  4) trials.csv: fall onset from the labels, impact suggested from the hip trajectory (CHECK BY EYE)
       python prepare_open_data.py trials --dir data/upfall --frames data/upfall/frames_30fps.csv --mode video

segments.csv : video_id, subject_id, label, start_s, end_s   (label = source label, e.g. fall, fallen, walk)
manifest.csv : video_id, subject_id, dataset, media [, fps, crop]  (media = video file, image folder or .zip
               of images; crop = "x,y,w,h" in pixels, e.g. one view of a split-screen video)
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

OMNIFALL_CLASSES = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up", "standing",
                    "other", "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump"]
# source label -> class of YOUR model (anything not listed -> --other-label). Edit with --label-map.
DEFAULT_LABEL_MAP = {"fall": "loss_of_balance", "fallen": "fallen"}
DYNAMIC_ADL = ["walk", "sit_down", "lie_down", "stand_up", "kneel_down", "squat_down", "crawl", "jump", "other"]
UPFALL_ACTIVITIES = {1: "fall_forward_hands", 2: "fall_forward_knees", 3: "fall_backward", 4: "fall_sideward",
                     5: "fall_sitting_empty_chair", 6: "walking", 7: "standing", 8: "sitting",
                     9: "picking_up_object", 10: "jumping", 11: "laying"}
VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".mpg", ".mpeg", ".m4v")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[T_ ](\d{2})[_:\-](\d{2})[_:\-](\d{2}(?:\.\d+)?)")
LM_COLS = [f"lm{i}_{a}" for i in range(33) for a in "xyzv"]
EPS = 1e-4   # s; label times and frame timestamps are rounded differently - far below one frame period


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def find_media(root: Path, rel: str, dataset: str | None = None):
    """Video file, image folder or zip for a relative path (extension optional)."""
    bases = [root / rel] + ([root / dataset / rel] if dataset else [])
    for b in bases:
        if b.is_file():
            return b
        for ext in VIDEO_EXT + (".zip",):
            for e in (ext, ext.upper()):
                c = Path(str(b) + e)
                if c.is_file():
                    return c
        if b.is_dir() and any(f.suffix.lower() in IMG_EXT for f in b.iterdir()):
            return b
    return None


def parse_roots(values) -> dict:
    """--video-root /data/all   or   --video-root up_fall=/data/UP-Fall le2i=/data/Le2i"""
    roots = {}
    for v in values or []:
        key, _, path = v.partition("=") if "=" in v else ("*", "", v)
        roots.setdefault(key, []).append(Path(path))
    return roots


def write_outputs(out: Path, manifest: pd.DataFrame, segments: pd.DataFrame):
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "manifest.csv", index=False)
    segments.to_csv(out / "segments.csv", index=False)
    found = int(manifest["media"].notna().sum())
    n_fall_videos = segments[segments["label"] == "fall"]["video_id"].nunique()
    print(f"{len(manifest)} videos ({found} found on disk), {manifest['subject_id'].nunique()} subjects, "
          f"{n_fall_videos} videos with a fall -> {out}/manifest.csv, segments.csv")
    if found < len(manifest):
        print("  missing media are listed with an empty 'media' column; check --video-root")


# ----------------------------------------------------------------------------
# 1a) OmniFall labels
# ----------------------------------------------------------------------------
def cmd_omnifall(args):
    df = pd.concat([pd.read_csv(p) for p in args.labels], ignore_index=True)
    need = {"path", "label", "start", "end", "subject", "dataset"}
    if need - set(df.columns):
        raise SystemExit(f"OmniFall labels need columns {sorted(need)}; found {list(df.columns)}")
    if args.split:
        keep = set(pd.concat([pd.read_csv(p) for p in args.split])["path"])
        df = df[df["path"].isin(keep)]
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)]
    if pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(int).map(lambda i: OMNIFALL_CLASSES[i])
    df["video_id"] = df["dataset"].astype(str) + "/" + df["path"].astype(str)
    sub = df["subject"].astype(int)
    df["subject_id"] = np.where(sub >= 0, df["dataset"].astype(str) + "_s" + sub.astype(str),
                                df["video_id"])            # unknown subject (e.g. OOPS): one group per video
    seg = df.rename(columns={"start": "start_s", "end": "end_s"})[
        ["video_id", "subject_id", "label", "start_s", "end_s", "dataset"]].sort_values(["video_id", "start_s"])
    man = df.groupby("video_id").agg(subject_id=("subject_id", "first"), dataset=("dataset", "first"),
                                     path=("path", "first")).reset_index()
    if "cam" in df.columns:
        man["cam"] = df.groupby("video_id")["cam"].first().reindex(man["video_id"]).to_numpy()
    roots = parse_roots(args.video_root)
    media = []
    for p, d in zip(man["path"], man["dataset"]):
        hit = None
        for r in roots.get(d, []) + roots.get("*", []):
            if (hit := find_media(r, p, d)) is not None:
                break
        media.append(str(hit) if hit is not None else None)
    man["media"] = media
    write_outputs(Path(args.out), man.drop(columns=["path"]), seg)


# ----------------------------------------------------------------------------
# 1b) UR Fall Detection labels
# ----------------------------------------------------------------------------
def _read_urfd(path):
    df = pd.read_csv(path, header=None)
    if not re.fullmatch(r"-?\d+", str(df.iloc[0, 1]).strip()):
        df = pd.read_csv(path)                                     # file has a header row
    df = df.iloc[:, :3]
    df.columns = ["seq", "frame", "label"]
    return df.astype({"frame": int, "label": int})


def cmd_urfd(args):
    """URFD labels: -1 not lying, 0 falling (temporary pose), 1 lying on the ground.
    Frame numbers in the CSV start at --frame-offset (1 by default) and the camera runs at ~30 fps."""
    rows, man = [], []
    for path, kind in ((args.falls, "fall"), (args.adls, "adl")):
        if not path:
            continue
        df = _read_urfd(path)
        for seq, g in df.groupby("seq"):
            g = g.sort_values("frame")
            names = {-1: "other", 0: "fall", 1: "fallen" if kind == "fall" else "lying"}
            run = (g["label"] != g["label"].shift()).cumsum()
            for _, r in g.groupby(run):
                rows.append({"video_id": f"{seq}-cam0", "subject_id": "urfd", "label": names[int(r["label"].iloc[0])],
                             "start_s": (r["frame"].iloc[0] - args.frame_offset) / args.fps,
                             "end_s": (r["frame"].iloc[-1] - args.frame_offset + 1) / args.fps, "dataset": "urfd"})
            media = None
            if args.video_root:
                root = Path(args.video_root)
                for name in (f"{seq}-cam0.mp4", f"{seq}-cam0-rgb.zip", f"{seq}-cam0-rgb"):
                    hit = next(iter(root.rglob(name)), None)
                    if hit is not None and (hit.is_file() or hit.is_dir()):
                        media = str(hit)
                        break
            man.append({"video_id": f"{seq}-cam0", "subject_id": "urfd", "dataset": "urfd", "media": media,
                        "fps": args.fps})
    if not man:
        raise SystemExit("give --falls and/or --adls")
    write_outputs(Path(args.out), pd.DataFrame(man), pd.DataFrame(rows))
    print("  URFD has no per-sequence subject ids: use it as an external test set, not for LOSO")


# ----------------------------------------------------------------------------
# 2) MediaPipe extraction
# ----------------------------------------------------------------------------
def _times_from_names(names, fps):
    stamps = []
    for n in names:
        m = TS_RE.search(Path(n).name)
        if not m:
            return np.arange(len(names)) / fps
        stamps.append(datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}").timestamp())
    stamps = np.array(stamps)
    return stamps - stamps[0]


def iter_media(media: str, fps_hint: float):
    """Yields (frame_idx, time_s, BGR image) from a video file, an image folder or a zip of images."""
    import cv2
    p = Path(media)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(IMG_EXT))
            for i, (n, t) in enumerate(zip(names, _times_from_names(names, fps_hint))):
                yield i, float(t), cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        for i, (f, t) in enumerate(zip(files, _times_from_names([f.name for f in files], fps_hint))):
            yield i, float(t), cv2.imread(str(f))
    else:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 < fps < 241 else fps_hint
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield i, i / fps, img
            i += 1
        cap.release()


def extract_video(row, args, part_path: Path):
    import cv2

    from benchmark_edge import Pose
    pose = Pose(args)                       # new instance per video (MediaPipe VIDEO mode needs rising timestamps)
    crop = None
    if isinstance(row.get("crop"), str) and row["crop"].strip():
        crop = [int(float(v)) for v in row["crop"].split(",")]
    fps_hint = float(row["fps"]) if pd.notna(row.get("fps")) else args.fps
    recs, last_ms = [], -1
    for i, t, img in iter_media(row["media"], fps_hint):
        if img is None:
            continue
        if crop:
            x, y, w, h = crop
            img = img[y:y + h, x:x + w]
        if args.max_side and max(img.shape[:2]) > args.max_side:
            s = args.max_side / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        last_ms = max(int(t * 1000.0), last_ms + 1)
        lms = pose(img, last_ms)
        vals = np.zeros(33 * 4, np.float32) if lms is None else lms.reshape(-1)
        recs.append([i, t, img.shape[1], img.shape[0], int(lms is not None), *vals.tolist()])
    pose.close()
    df = pd.DataFrame(recs, columns=[C.FRAME_COL, C.TIME_COL, "frame_w", "frame_h", "detected"] + LM_COLS)
    df.insert(0, C.VIDEO_COL, row["video_id"])
    df.insert(0, C.SUBJECT_COL, row["subject_id"])
    if "dataset" in row and pd.notna(row["dataset"]):
        df.insert(2, "dataset", row["dataset"])
    tmp = part_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, float_format="%.5f")
    tmp.replace(part_path)                  # atomic: an interrupted run never leaves half a file
    return len(df), float(df["detected"].mean()) if len(df) else 0.0


def label_frames(frames: pd.DataFrame, segments: pd.DataFrame, label_map: dict, other: str) -> pd.Series:
    """Per-frame class of your model from time segments (segment containing the frame's timestamp)."""
    out = pd.Series(other, index=frames.index, dtype=object)
    groups = frames.groupby(C.VIDEO_COL).indices
    for vid, seg in segments.groupby("video_id"):
        if vid not in groups:
            continue
        idx = frames.index[groups[vid]]
        seg = seg.sort_values("start_s")
        t = frames.loc[idx, C.TIME_COL].to_numpy(float)
        k = np.searchsorted(seg["start_s"].to_numpy(float), t + EPS, side="right") - 1
        inside = (k >= 0) & (t + EPS < seg["end_s"].to_numpy(float)[np.clip(k, 0, None)])
        lab = np.where(inside, seg["label"].to_numpy()[np.clip(k, 0, None)], None)
        out.loc[idx] = [label_map.get(x, other) if x is not None else other for x in lab]
    return out


def cmd_extract(args):
    d = Path(args.dir)
    man = pd.read_csv(args.manifest or d / "manifest.csv")
    man = man[man["media"].notna()]
    parts = d / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    if not args.merge_only:
        for n, row in enumerate(man.to_dict("records"), 1):
            part = parts / (re.sub(r"[^A-Za-z0-9_.-]", "_", row["video_id"]) + ".csv")
            if part.exists() and not args.overwrite:
                continue
            try:
                nf, det = extract_video(row, args, part)
                print(f"[{n}/{len(man)}] {row['video_id']}: {nf} frames, person found in {100 * det:.0f}%", flush=True)
            except Exception as e:  # noqa: BLE001 - keep going, report at the end
                warnings.warn(f"{row['video_id']}: {e}")
    files = [parts / (re.sub(r"[^A-Za-z0-9_.-]", "_", v) + ".csv") for v in man["video_id"]]
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit("no extracted videos yet")
    frames = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    seg_path = Path(args.segments) if args.segments else d / "segments.csv"
    if seg_path.exists():
        label_map = {**DEFAULT_LABEL_MAP, **json.loads(args.label_map)} if args.label_map else DEFAULT_LABEL_MAP
        bad = set(label_map.values()) - set(C.CLASS_NAMES) | ({args.other_label} - set(C.CLASS_NAMES))
        if bad:
            raise SystemExit(f"labels {sorted(bad)} are not in CLASS_NAMES of config.py")
        labels = label_frames(frames, pd.read_csv(seg_path), label_map, args.other_label)
        frames = frames.copy()
        frames.insert(frames.columns.get_loc(C.TIME_COL) + 1, C.LABEL_COL, labels)
    out = Path(args.frames_out) if args.frames_out else d / "frames.csv"
    frames.to_csv(out, index=False, float_format="%.5f")
    print(f"wrote {out}: {len(frames)} frames from {len(files)}/{len(man)} videos")
    if C.LABEL_COL in frames.columns:
        print(frames[C.LABEL_COL].value_counts().to_string())


# ----------------------------------------------------------------------------
# 3) resample to a fixed frame rate
# ----------------------------------------------------------------------------
def cmd_resample(args):
    df = pd.read_csv(args.frames)
    lm = [c for c in df.columns if c.startswith("lm")]
    out_parts = []
    for vid, g in df.groupby(C.VIDEO_COL, sort=False):
        g = g.sort_values(C.FRAME_COL)
        t = g[C.TIME_COL].to_numpy(float)
        tn = np.arange(t[0], t[-1] + 1e-9, 1.0 / args.fps)
        valid = g["detected"].to_numpy(bool) if "detected" in g else np.ones(len(g), bool)
        lm_vals = {c: np.zeros(len(tn)) for c in lm}
        detected = np.zeros(len(tn), int)
        if valid.sum() >= 2:
            tv = t[valid]
            # no interpolation across long gaps without a detected person, nor beyond the first/last detection
            j = np.searchsorted(tv, tn, side="right")
            left, right = tv[np.clip(j - 1, 0, len(tv) - 1)], tv[np.clip(j, 0, len(tv) - 1)]
            exact = np.abs(tn - left) < 1e-6
            outside = (tn < tv[0] - 1e-6) | (tn > tv[-1] + 1e-6)
            gap = ~exact & (outside | ((right - left) > args.max_gap))
            for c in lm:
                lm_vals[c] = np.where(gap, 0.0, np.interp(tn, tv, g[c].to_numpy(float)[valid]))
            detected = (~gap).astype(int)
        prev = np.clip(np.searchsorted(t, tn, side="right") - 1, 0, len(t) - 1)
        meta = {c: g[c].to_numpy()[prev] for c in [C.SUBJECT_COL, C.VIDEO_COL, "dataset", C.LABEL_COL,
                                                      "frame_w", "frame_h"] if c in g.columns}
        res = pd.DataFrame({**meta, C.FRAME_COL: np.arange(len(tn)), C.TIME_COL: tn, "detected": detected, **lm_vals})
        out_parts.append(res)
    out = pd.concat(out_parts, ignore_index=True)
    front = [c for c in [C.SUBJECT_COL, C.VIDEO_COL, "dataset", C.FRAME_COL, C.TIME_COL, C.LABEL_COL,
                         "frame_w", "frame_h", "detected"] if c in out.columns]
    out = out[front + lm]
    path = args.out or str(Path(args.frames).with_name(Path(args.frames).stem + f"_{args.fps:g}fps.csv"))
    out.to_csv(path, index=False, float_format="%.5f")
    print(f"wrote {path}: {len(out)} frames at {args.fps:g} fps (was {len(df)})")


# ----------------------------------------------------------------------------
# 4) trials
# ----------------------------------------------------------------------------
def scenario_name(video_id, dataset, seg, ttype, fall_start=None):
    if dataset == "up_fall":
        m = re.search(r"Activity(\d+)", video_id)
        if m and int(m.group(1)) in UPFALL_ACTIVITIES:
            return UPFALL_ACTIVITIES[int(m.group(1))]
    if ttype == "fall":
        before = seg[seg["end_s"] <= fall_start + 1e-6]
        return f"{before['label'].iloc[-1]}->fall" if len(before) else "fall"
    dyn = seg[seg["label"].isin(DYNAMIC_ADL)]
    if len(dyn):
        return str((dyn["end_s"] - dyn["start_s"]).groupby(dyn["label"]).sum().idxmax())
    return str(seg["label"].iloc[0]) if len(seg) else "adl"


def cmd_trials(args):
    from data import load_frames
    from decision import build_traces, slice_trace
    from lead_time import suggest_impact

    d = Path(args.dir)
    frames = load_frames(args.frames or d / "frames.csv")
    segs = pd.read_csv(args.segments or d / "segments.csv")
    segs["video_id"] = segs["video_id"].astype(str)
    traces = build_traces(frames, None)
    ds = frames.groupby(C.VIDEO_COL)["dataset"].first() if "dataset" in frames.columns else {}
    fall_labels = set(args.fall_labels)
    rows = []

    def first_after(tr, t):
        k = np.searchsorted(tr.t, t - EPS, side="left")
        return int(tr.frame[min(k, len(tr.frame) - 1)])

    def last_before(tr, t):
        k = np.searchsorted(tr.t, t + EPS, side="right") - 1
        return int(tr.frame[max(k, 0)])

    def last_inside(tr, t_end):        # segment ends are exclusive: last frame before t_end
        k = np.searchsorted(tr.t, t_end - EPS, side="left") - 1
        return int(tr.frame[max(k, 0)])

    def impact_for(tr, on_f, end_f):
        if args.impact == "segment_end":
            return end_f, "label_segment_end (last frame of the fall label: at/after impact)"
        if args.impact == "suggest":
            fps = (len(tr.t) - 1) / max(tr.t[-1] - tr.t[0], 1e-6)
            imp = suggest_impact(slice_trace(tr, on_f, end_f + int(round(0.5 * fps))))
            return (imp, "suggested_from_hip (VERIFY)") if pd.notna(imp) else (np.nan, "none")
        return np.nan, "none"

    for vid, tr in traces.items():
        seg = segs[segs["video_id"] == vid].sort_values("start_s")
        dataset = ds.get(vid, "") if len(ds) else ""
        falls = seg[seg["label"].isin(fall_labels)]
        base = {"video_id": vid, "subject_id": tr.subject_id, "dataset": dataset}
        if args.mode == "video":
            if len(falls):
                if len(falls) > 1:
                    warnings.warn(f"{vid}: {len(falls)} falls in one video - use --mode events")
                f0 = falls.iloc[0]
                on_f, end_f = first_after(tr, f0["start_s"]), last_inside(tr, f0["end_s"])
                imp, src = impact_for(tr, on_f, end_f)
                rows.append({**base, "trial_id": vid, "trial_type": "fall",
                             "scenario": scenario_name(vid, dataset, seg, "fall", f0["start_s"]),
                             "onset_frame": on_f, "impact_frame": imp, "impact_source": src, "fall_end_frame": end_f})
            else:
                rows.append({**base, "trial_id": vid, "trial_type": "adl",
                             "scenario": scenario_name(vid, dataset, seg, "adl")})
        else:   # events: one trial per fall / per dynamic ADL segment, each with its own frame window
            windows = []
            for k, f in enumerate(falls.itertuples(), 1):
                on_f, end_f = first_after(tr, f.start_s), last_inside(tr, f.end_s)
                imp, src = impact_for(tr, on_f, end_f)
                w0, w1 = first_after(tr, f.start_s - args.pre_s), last_before(tr, f.end_s + args.post_s)
                windows.append((f.start_s - args.pre_s, f.end_s + args.post_s))
                rows.append({**base, "trial_id": f"{vid}#fall{k}", "trial_type": "fall",
                             "scenario": scenario_name(vid, dataset, seg, "fall", f.start_s),
                             "onset_frame": on_f, "impact_frame": imp, "impact_source": src,
                             "fall_end_frame": end_f, "start_frame": w0, "end_frame": w1})
            for k, a in enumerate(seg[seg["label"].isin(args.adl_labels)].itertuples(), 1):
                if any(a.start_s < b1 and a.end_s > b0 for b0, b1 in windows):
                    continue                    # the activity itself overlaps a fall window
                # context before/after the activity, clipped so it never reaches into a fall window
                a0 = max([a.start_s - args.pre_s] + [b1 for _, b1 in windows if b1 <= a.start_s])
                a1 = min([a.end_s + args.post_s] + [b0 for b0, _ in windows if b0 >= a.end_s])
                rows.append({**base, "trial_id": f"{vid}#{a.label}{k}", "trial_type": "adl", "scenario": a.label,
                             "start_frame": first_after(tr, a0), "end_frame": last_before(tr, a1)})
    out = pd.DataFrame(rows)
    cols = ["trial_id", "video_id", "subject_id", "dataset", "trial_type", "scenario", "onset_frame",
            "impact_frame", "impact_source", "fall_end_frame", "start_frame", "end_frame"]
    out = out[[c for c in cols if c in out.columns]]
    path = Path(args.out) if args.out else d / "trials.csv"
    out.to_csv(path, index=False)
    n_f = int((out["trial_type"] == "fall").sum())
    print(f"wrote {path}: {n_f} fall trials, {len(out) - n_f} ADL trials")
    if n_f and args.impact == "suggest":
        print("  impact_frame is a SUGGESTION from the hip trajectory: check every fall frame-by-frame "
              "(lead_time.py --suggest-impact draws the plots) before reporting lead time")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("omnifall", help="OmniFall label CSVs -> manifest + segments")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--split", nargs="*", help="optional split CSVs (keep only their 'path's)")
    p.add_argument("--datasets", nargs="*", help="keep only these dataset names, e.g. up_fall le2i")
    p.add_argument("--video-root", nargs="*", help="folder(s) with the original videos: PATH or DATASET=PATH")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_omnifall)

    p = sp.add_parser("urfd", help="UR Fall Detection label CSVs -> manifest + segments")
    p.add_argument("--falls")
    p.add_argument("--adls")
    p.add_argument("--video-root")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--frame-offset", type=int, default=1, help="frame number of the first frame in the CSV")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_urfd)

    p = sp.add_parser("extract", help="MediaPipe landmarks for every video in the manifest -> frames.csv")
    p.add_argument("--dir", required=True, help="output folder of step 1")
    p.add_argument("--manifest")
    p.add_argument("--segments")
    p.add_argument("--pose-model", help="pose_landmarker_{lite,full,heavy}.task (MediaPipe Tasks API)")
    p.add_argument("--model-complexity", type=int, default=1, help="legacy mp.solutions API only")
    p.add_argument("--dummy-pose", action="store_true", help="random landmarks, for testing the pipeline")
    p.add_argument("--fps", type=float, default=30.0, help="frame rate of image sequences without timestamps")
    p.add_argument("--max-side", type=int, default=0, help="downscale frames whose longer side exceeds this")
    p.add_argument("--label-map", help='JSON, e.g. \'{"fall": "loss_of_balance", "fallen": "fallen"}\'')
    p.add_argument("--other-label", default="normal")
    p.add_argument("--frames-out")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--merge-only", action="store_true", help="skip extraction, rebuild frames.csv (e.g. new label map)")
    p.set_defaults(func=cmd_extract)

    p = sp.add_parser("resample", help="resample landmarks to a fixed frame rate")
    p.add_argument("--frames", required=True)
    p.add_argument("--fps", type=float, default=C.FPS)
    p.add_argument("--max-gap", type=float, default=0.25, help="s; longer gaps without a person stay empty")
    p.add_argument("--out")
    p.set_defaults(func=cmd_resample)

    p = sp.add_parser("trials", help="segments + frames -> trials.csv for lead_time.py / ablation.py")
    p.add_argument("--dir", required=True)
    p.add_argument("--frames")
    p.add_argument("--segments")
    p.add_argument("--mode", choices=["video", "events"], default="video",
                   help="video: one trial per video (clip datasets); events: one per fall/ADL segment (long recordings)")
    p.add_argument("--impact", choices=["suggest", "segment_end", "none"], default="suggest")
    p.add_argument("--fall-labels", nargs="+", default=["fall"])
    p.add_argument("--adl-labels", nargs="+", default=DYNAMIC_ADL)
    p.add_argument("--pre-s", type=float, default=2.0)
    p.add_argument("--post-s", type=float, default=1.5)
    p.add_argument("--out")
    p.set_defaults(func=cmd_trials)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
