#!/usr/bin/env python3
"""
Download the public fall data that needs no request form, into ./public_data, then print the commands that
turn it into frames.csv / trials.csv for fallriskpredict (https://github.com/chayuton/fallriskpredict).

  1) URFD (UR Fall Detection): camera-0 videos of 30 falls + 40 ADL and the per-frame labels
     licence CC BY-NC-SA 4.0 (non-commercial research only), https://fenix.ur.edu.pl/~mkepski/ds/uf.html
  2) OmniFall label files (Hugging Face: simplexsigil2/omnifall), CC BY-NC(-SA) 4.0.
     Labels only: the videos must come from each source dataset (UP-Fall, CMDFall, Le2i, CAUCAFall, ...).

  python get_public_data.py                 # both
  python get_public_data.py --only urfd     # or: --only omnifall
  python get_public_data.py --out D:/falls  # another folder

Re-running skips files that are already complete.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

URFD_BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
URFD_FILES = (["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]
              + [f"fall-{i:02d}-cam0.mp4" for i in range(1, 31)]
              + [f"adl-{i:02d}-cam0.mp4" for i in range(1, 41)])


def fetch(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "already there"
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fallriskpredict-data-helper"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)
    return f"{dest.stat().st_size / 1e6:.1f} MB"


def get_urfd(root: Path) -> Path:
    d = root / "urfd"
    d.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, name in enumerate(URFD_FILES, 1):
        try:
            status = fetch(URFD_BASE + name, d / name)
        except Exception as e:  # noqa: BLE001 - report and keep going
            status = f"FAILED: {e}"
            failed.append(name)
        print(f"  [{i}/{len(URFD_FILES)}] urfd/{name}: {status}", flush=True)
    if failed:
        print(f"  {len(failed)} file(s) failed - run the script again to retry them")
    return d


def get_omnifall(root: Path) -> Path | None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  OmniFall needs huggingface_hub:  pip install huggingface_hub   (then run this script again)")
        return None
    d = root / "omnifall"
    try:
        snapshot_download("simplexsigil2/omnifall", repo_type="dataset",
                          allow_patterns=["labels/*", "splits/*", "README.md"], local_dir=str(d))
    except Exception as e:  # noqa: BLE001
        print(f"  OmniFall download failed: {e}\n  If the dataset page asks you to accept its terms, accept them on "
              "https://huggingface.co/datasets/simplexsigil2/omnifall, run `huggingface-cli login`, then retry.")
        return None
    print(f"  OmniFall labels in {d}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="public_data")
    ap.add_argument("--only", choices=["urfd", "omnifall"])
    a = ap.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "urfd"):
        print("URFD (30 falls + 40 ADL, camera 0, about 30 fps):")
        u = get_urfd(root)
        print(f"""
  next, inside the fallriskpredict folder:
    python prepare_open_data.py urfd --falls {u}/urfall-cam0-falls.csv --adls {u}/urfall-cam0-adls.csv --video-root {u} --out data/urfd
    python prepare_open_data.py extract  --dir data/urfd
    python prepare_open_data.py resample --frames data/urfd/frames.csv --fps 30
    python prepare_open_data.py trials   --dir data/urfd --frames data/urfd/frames_30fps.csv --mode video --impact segment_end
  URFD has no per-sequence subject ids: use it as an external test set (predict_trials.py with a model trained elsewhere).
""")
    if a.only in (None, "omnifall"):
        print("OmniFall labels:")
        o = get_omnifall(root)
        if o:
            csvs = sorted(p for p in o.rglob("*.csv") if "split" not in str(p).lower())
            print("  label files:", ", ".join(str(p.relative_to(o)) for p in csvs[:12]) or "none found")
            up = next((p for p in csvs if p.stem.lower().replace("-", "_") == "up_fall"), o / "labels" / "up_fall.csv")
            print(f"""
  next, after downloading the videos of a source dataset (e.g. UP-Fall camera videos):
    python prepare_open_data.py omnifall --labels {up} --video-root up_fall=/path/to/UP-Fall --out data/upfall
    python prepare_open_data.py extract  --dir data/upfall
    python prepare_open_data.py resample --frames data/upfall/frames.csv --fps 30
    python prepare_open_data.py trials   --dir data/upfall --frames data/upfall/frames_30fps.csv --mode video --impact suggest
  (CMDFall and other long recordings: --mode events)
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
