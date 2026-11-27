#!/usr/bin/env python3
"""
GMDCSA24 annotations -> manifest.csv + segments.csv for prepare_open_data.py.

GMDCSA24 (Alam et al., Data in Brief 2024; MIT licence) keeps one annotation file per subject and type,
`Subject N/Fall.csv` and `Subject N/ADL.csv`, whose `Classes` column lists activities with time ranges in
seconds, e.g. "Walking[0 to 3.4]; Falling (SW)[3.4 to 6]". This script turns them into

  manifest.csv  video_id, subject_id, dataset, media (path of the mp4), description, classes
  segments.csv  video_id, subject_id, label, start_s, end_s, dataset, source_label, fall_direction

Labels: every "Falling (..)" range becomes `fall` (direction SW/FW/BW -> sideways/forward/backward);
other activities keep their name in lower case (walking, sitting, ...). Two irregular rows are repaired and
reported: a fall written without a time range (taken from the end of the previous activity to the end of
the video) and one fall written as consecutive phases (merged into one range).

  git clone https://github.com/ekramalam/gmdcsa24-a-dataset-for-human-fall-detection-in-videos data/raw/gmdcsa24
  python datasets/gmdcsa24/gmdcsa24_labels.py data/raw/gmdcsa24 data/gmdcsa24
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SEG = re.compile(r"([A-Za-z][A-Za-z \-/()]*?)\s*\[\s*([\d.]+)\s*(?:to)?\s*([\d.]+)\s*\]")   # "[2.3 6]" typo tolerated
DIRECTIONS = {"SW": "sideways", "FW": "forward", "BW": "backward"}


def video_duration_s(media: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(media))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    return dur


def parse(root: Path):
    man, segs, problems = [], [], []
    for sdir in sorted(root.glob("Subject *")):
        s = sdir.name.split()[-1]
        for kind in ("Fall", "ADL"):
            ann = pd.read_csv(sdir / f"{kind}.csv", skipinitialspace=True, dtype=str)
            ann.columns = [c.strip() for c in ann.columns]
            for d in ann.to_dict("records"):
                fname = str(d["File Name"]).strip()
                media = sdir / kind / fname
                if not media.exists():
                    problems.append(f"missing video {media}")
                    continue
                vid = f"gmdcsa24/S{s}/{kind}/{Path(fname).stem}"
                man.append({"video_id": vid, "subject_id": f"gmdcsa24_s{s}", "dataset": "gmdcsa24",
                            "media": str(media), "description": str(d.get("Description", "")).strip(),
                            "classes": str(d.get("Classes", "")).strip()})
                text = str(d.get("Classes", ""))
                found = SEG.findall(text)
                if kind == "Fall" and not any(n.strip().lower().startswith("fall") for n, _, _ in found) \
                        and "fall" in text.lower():
                    # e.g. "Falling (FW); Sitting[0 to 1.7]": no range -> end of the last activity .. end of video
                    dur = video_duration_s(media)
                    m = re.search(r"(Falling\s*\(\w+\))", text)
                    start = max([float(b) for _, _, b in found] or [0.0])
                    found.append((m.group(1) if m else "Falling", str(start), f"{dur:.2f}"))
                    problems.append(f"inferred fall interval for {vid}: [{start}, {dur:.2f}] from '{text}'")
                if not found:
                    problems.append(f"no segments parsed for {vid}: {d.get('Classes')}")
                for name, a, b in found:
                    name = name.strip()
                    low = name.lower()
                    if low.startswith("fall"):
                        m = re.search(r"\((\w+)\)", name)
                        label, direction = "fall", DIRECTIONS.get(m.group(1) if m else "", "unspecified")
                    else:
                        label, direction = re.sub(r"\W+", "_", low).strip("_"), ""
                    segs.append({"video_id": vid, "subject_id": f"gmdcsa24_s{s}", "label": label,
                                 "start_s": float(a), "end_s": float(b), "dataset": "gmdcsa24",
                                 "source_label": name, "fall_direction": direction})
    seg = pd.DataFrame(segs).sort_values(["video_id", "start_s"]).reset_index(drop=True)
    # one fall described in phases, e.g. "Falling (BW)[2 to 4]; Falling (SW)[4 to 11]" -> one fall segment
    merged = []
    for r in seg.to_dict("records"):
        if merged and r["label"] == "fall" and merged[-1]["label"] == "fall" \
                and merged[-1]["video_id"] == r["video_id"] and r["start_s"] - merged[-1]["end_s"] <= 0.5:
            merged[-1]["end_s"] = max(merged[-1]["end_s"], r["end_s"])
            merged[-1]["source_label"] += " + " + r["source_label"]
            problems.append(f"merged consecutive fall phases in {r['video_id']}")
        else:
            merged.append(dict(r))
    return pd.DataFrame(man), pd.DataFrame(merged), problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="GMDCSA24 folder that contains 'Subject 1' ... 'Subject 4'")
    ap.add_argument("out", help="output folder for manifest.csv and segments.csv (e.g. data/gmdcsa24)")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    if not any(root.glob("Subject *")):
        raise SystemExit(f"no 'Subject N' folders in {root}")
    out.mkdir(parents=True, exist_ok=True)
    man, seg, problems = parse(root)
    man.to_csv(out / "manifest.csv", index=False)
    seg.to_csv(out / "segments.csv", index=False)
    print(f"{len(man)} videos, {len(seg)} segments, {seg[seg.label == 'fall'].video_id.nunique()} videos with a fall")
    print("labels:", seg.label.value_counts().to_dict())
    print("\n".join(problems[:20]) or "no parsing problems")


if __name__ == "__main__":
    main()
