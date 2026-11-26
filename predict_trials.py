"""
Per-frame GRU probabilities for trial recordings that were NOT in the LOSO training data
(e.g. separately recorded scenario trials and ADL trials). Output has the same format as
results/loso/oof_predictions.csv, so it can be fed to lead_time.py and ablation.py.

Each video is scored by the LOSO model in which the video's subject was held out, so no
subject is scored by a model that saw that subject during training. Videos of subjects
that never appeared in the training data are scored with --model.

Usage
  python predict_trials.py --frames data/trial_frames.csv --loso-dir models/loso \
                           --out results/trial_predictions.csv
  python predict_trials.py --frames data/trial_frames.csv --model my_model.keras \
                           --out results/trial_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

import config as C  # noqa: E402
from data import feature_columns, load_frames, make_windows, prob_columns  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--loso-dir", help="models/loso (contains fold_info.json)")
    ap.add_argument("--model", help="model for subjects without a LOSO fold (.keras/.h5)")
    ap.add_argument("--scaler", help="optional .npz with mean/std for --model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.loso_dir and not args.model:
        ap.error("give --loso-dir and/or --model")

    from tensorflow import keras

    df = load_frames(args.frames)
    feats = feature_columns(df)
    with_labels = C.LABEL_COL in df.columns
    X, y, meta = make_windows(df, feats, with_labels=with_labels)

    subject_to_key = {}
    if args.loso_dir:
        d = Path(args.loso_dir)
        info = json.loads((d / "fold_info.json").read_text(encoding="utf-8"))
        for fold in info["folds"].values():
            key = (str(d / fold["model"]), str(d / fold["scaler"]) if fold.get("scaler") else "")
            for s in fold["test_subjects"]:
                subject_to_key[str(s)] = key
    default_key = (args.model, args.scaler or "") if args.model else None

    keys = meta[C.SUBJECT_COL].map(lambda s: subject_to_key.get(str(s), default_key))
    if keys.isna().any():
        missing = sorted(meta.loc[keys.isna(), C.SUBJECT_COL].unique().tolist())
        raise SystemExit(f"No model for subjects {missing}: pass --model for subjects outside the LOSO folds")

    K = len(C.CLASS_NAMES)
    prob = np.zeros((len(X), K), np.float32)
    used = np.empty(len(X), dtype=object)
    for key in sorted(set(keys)):
        idx = np.flatnonzero((keys == key).to_numpy())
        model = keras.models.load_model(key[0], compile=False)
        Xi = X[idx]
        if key[1]:
            sc = np.load(key[1])
            Xi = (Xi - sc["mean"]) / sc["std"]
        p = model.predict(Xi, batch_size=1024, verbose=0)
        if p.shape[1] != K:
            raise SystemExit(f"{key[0]} outputs {p.shape[1]} classes but CLASS_NAMES has {K}")
        prob[idx] = p
        used[idx] = Path(key[0]).name
        print(f"{Path(key[0]).name}: {len(idx)} windows")

    out = meta.copy()
    out["fold"] = used
    if with_labels:
        out["y_true"] = y
    out["y_pred"] = prob.argmax(1)
    for k, col in enumerate(prob_columns()):
        out[col] = prob[:, k]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
