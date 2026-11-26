"""
Subject-independent evaluation of the GRU classifier.
(Reviewer 1 #1, Reviewer 2 #1, Reviewer 3: "train and test frames come from the same recordings")

Protocols
  loso   : Leave-One-Subject-Out - one fold per subject, the held-out subject is never seen in training
  video  : video-independent GroupKFold - no window of the same recording in both train and test
  random : random 80/20 split of windows - reproduces the original protocol (leaky), for comparison
  all    : run the three protocols and write a comparison table

Early stopping uses a validation split taken from the TRAINING subjects only (never the test subject),
and feature standardisation (if enabled) is fitted on the training fold only.

Usage
  python loso.py --frames data/frames.csv --mode all
  python loso.py --frames data/frames.csv --mode loso --epochs 60

Outputs (results/<mode>/)
  fold_metrics.csv      one row per fold (per held-out subject for LOSO)
  oof_predictions.csv   out-of-fold window probabilities -> input of lead_time.py / ablation.py
  summary.json / .md    mean ± SD over folds and pooled metrics
  confusion_matrix.png  pooled, row-normalised
results/split_comparison.md (mode all)
models/<mode>/*.keras + fold_info.json (used by predict_trials.py)
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_fscore_support, precision_score,
                             recall_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, LeaveOneGroupOut,  # noqa: E402
                                     train_test_split)
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

import config as C  # noqa: E402
from data import feature_columns, load_frames, make_windows, positive_ids, prob_columns  # noqa: E402
from utils import md_table, mean_sd_pct, pct, safe_name, save_json  # noqa: E402

METRICS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
           "f1_weighted", "sensitivity_warning", "specificity_warning"]
PROTOCOL_NAMES = {
    "random": "Random window split 80/20 (original, leaky)",
    "video": "Video-independent GroupKFold",
    "loso": "Leave-One-Subject-Out (LOSO)",
}


# ----------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------
def iter_folds(mode, y, subj, vid, seed):
    idx = np.arange(len(y))
    if mode == "loso":
        for tr, te in LeaveOneGroupOut().split(idx, groups=subj):
            yield str(subj[te][0]), tr, te
    elif mode == "video":
        k = min(C.VIDEO_FOLDS, len(np.unique(vid)))
        for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(idx, groups=vid)):
            yield f"fold{i + 1}", tr, te
    elif mode == "random":
        try:
            tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        except ValueError:
            tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
        yield "random80-20", tr, te
    else:
        raise ValueError(mode)


def inner_split(mode, tr, y, subj, vid, seed):
    """Validation set for early stopping, drawn from the training fold only."""
    if mode == "random":
        try:
            return train_test_split(tr, test_size=C.VAL_FRACTION, stratify=y[tr], random_state=seed)
        except ValueError:
            return train_test_split(tr, test_size=C.VAL_FRACTION, random_state=seed)
    use_subjects = mode == "loso" and len(np.unique(subj[tr])) >= 3
    groups = subj[tr] if use_subjects else vid[tr]
    gss = GroupShuffleSplit(n_splits=1, test_size=C.VAL_FRACTION, random_state=seed)
    a, b = next(gss.split(tr, groups=groups))
    return tr[a], tr[b]


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def fit_scaler(X):
    flat = X.reshape(-1, X.shape[2])
    mean, std = flat.mean(0), flat.std(0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def train_fold(X_fit, y_fit, X_val, y_val, n_classes, seed, epochs, verbose, batch_size=C.BATCH_SIZE):
    from tensorflow import keras

    from model import build_gru

    class BestWeights(keras.callbacks.Callback):
        """Keeps the weights of the epoch with the lowest val_loss (works on Keras 2 and 3)."""

        def __init__(self):
            super().__init__()
            self.best, self.weights, self.best_epoch = np.inf, None, 0

        def on_epoch_end(self, epoch, logs=None):
            v = (logs or {}).get("val_loss")
            if v is not None and v < self.best:
                self.best, self.weights, self.best_epoch = v, self.model.get_weights(), epoch + 1

        def on_train_end(self, logs=None):
            if self.weights is not None:
                self.model.set_weights(self.weights)

    keras.utils.set_random_seed(seed)
    model = build_gru(X_fit.shape[1], X_fit.shape[2], n_classes)
    class_weight = None
    if C.CLASS_WEIGHT:
        present = np.unique(y_fit)
        w = compute_class_weight("balanced", classes=present, y=y_fit)
        class_weight = {i: 1.0 for i in range(n_classes)}
        class_weight.update({int(c): float(wi) for c, wi in zip(present, w)})
    best = BestWeights()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE)
    hist = model.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=epochs,
                     batch_size=batch_size, class_weight=class_weight,
                     callbacks=[best, stop], verbose=verbose)
    return model, best.best_epoch, len(hist.history["loss"])


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    K = len(C.CLASS_NAMES)
    pos = positive_ids()
    out = {"n_windows": int(len(y_true))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["accuracy"] = accuracy_score(y_true, y_pred)
        out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        # macro over the classes present in this fold (absent classes would add a spurious 0)
        out["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        out["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        out["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        out["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        _, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(K)), zero_division=0)
    present = set(np.unique(y_true).tolist())
    for k, name in enumerate(C.CLASS_NAMES):
        out[f"recall_{name}"] = float(r[k]) if k in present else np.nan
        out[f"f1_{name}"] = float(f[k]) if k in present else np.nan
    t_pos, p_pos = np.isin(y_true, pos), np.isin(y_pred, pos)
    tp, fn = np.sum(t_pos & p_pos), np.sum(t_pos & ~p_pos)
    tn, fp = np.sum(~t_pos & ~p_pos), np.sum(~t_pos & p_pos)
    out["sensitivity_warning"] = tp / (tp + fn) if tp + fn else np.nan
    out["specificity_warning"] = tn / (tn + fp) if tn + fp else np.nan
    return out


def plot_confusion(y_true, y_pred, path, title):
    K = len(C.CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    with np.errstate(invalid="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(1.6 * K + 2.2, 1.4 * K + 1.8))
    ax.imshow(np.nan_to_num(cmn), cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            val = cmn[i, j]
            ax.text(j, i, f"{cm[i, j]}\n({100 * val:.1f}%)" if not np.isnan(val) else "0",
                    ha="center", va="center", fontsize=9,
                    color="white" if (not np.isnan(val) and val > 0.6) else "black")
    ax.set_xticks(range(K), C.CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(K), C.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# one protocol
# ----------------------------------------------------------------------------
def run_protocol(mode, X, y, meta, args):
    out_dir = Path(args.out) / mode
    model_dir = Path(args.models) / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    subj = meta[C.SUBJECT_COL].to_numpy()
    vid = meta[C.VIDEO_COL].to_numpy()
    K = len(C.CLASS_NAMES)
    folds = list(iter_folds(mode, y, subj, vid, args.seed))
    fold_rows, oof_parts, fold_info = [], [], {}

    for i, (fold, tr, te) in enumerate(folds, 1):
        t0 = time.time()
        fit_idx, val_idx = inner_split(mode, tr, y, subj, vid, args.seed)
        X_fit, X_val, X_te = X[fit_idx], X[val_idx], X[te]
        scaler = None
        if C.STANDARDIZE:
            scaler = fit_scaler(X_fit)
            X_fit, X_val, X_te = [(a - scaler[0]) / scaler[1] for a in (X_fit, X_val, X_te)]
        model, best_epoch, n_epochs = train_fold(X_fit, y[fit_idx], X_val, y[val_idx], K,
                                                 args.seed, args.epochs, args.verbose, args.batch_size)
        prob = model.predict(X_te, batch_size=1024, verbose=0)
        pred = prob.argmax(1)
        m = compute_metrics(y[te], pred)
        test_subjects = sorted(np.unique(subj[te]).tolist())
        row = {"fold": fold, "test_subjects": ";".join(test_subjects),
               "n_train_windows": len(fit_idx), "n_val_windows": len(val_idx),
               "best_epoch": best_epoch, "epochs_run": n_epochs,
               "train_time_s": round(time.time() - t0, 1), **m}
        fold_rows.append(row)

        part = meta.iloc[te].reset_index(drop=True).copy()
        part["fold"] = fold
        part["y_true"] = y[te]
        part["y_pred"] = pred
        for k, col in enumerate(prob_columns()):
            part[col] = prob[:, k]
        oof_parts.append(part)

        if not args.no_save_models:
            model_dir.mkdir(parents=True, exist_ok=True)
            mpath = model_dir / f"{safe_name(fold)}.keras"
            model.save(mpath)
            spath = None
            if scaler is not None:
                spath = model_dir / f"{safe_name(fold)}_scaler.npz"
                np.savez(spath, mean=scaler[0], std=scaler[1])
            fold_info[fold] = {"test_subjects": test_subjects, "model": mpath.name,
                               "scaler": spath.name if spath else None, "best_epoch": best_epoch}
        print(f"[{mode}] fold {i}/{len(folds)} {fold:>12s}: acc={m['accuracy']:.4f} "
              f"F1macro={m['f1_macro']:.4f} (best epoch {best_epoch}/{n_epochs}, {row['train_time_s']} s)",
              flush=True)

    fold_df = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_parts, ignore_index=True)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    if fold_info:
        save_json({"mode": mode, "window": C.WINDOW, "class_names": C.CLASS_NAMES, "folds": fold_info},
                  model_dir / "fold_info.json")

    pooled = compute_metrics(oof["y_true"], oof["y_pred"])
    summary = {"mode": mode, "protocol": PROTOCOL_NAMES[mode], "n_folds": len(fold_df),
               "n_subjects": int(len(np.unique(subj))), "n_videos": int(len(np.unique(vid))),
               "n_windows": int(len(y))}
    for k in METRICS:
        summary[f"{k}_mean"] = float(fold_df[k].mean())
        summary[f"{k}_sd"] = float(fold_df[k].std(ddof=1)) if len(fold_df) > 1 else np.nan
        summary[f"{k}_pooled"] = float(pooled[k])
    for name in C.CLASS_NAMES:
        summary[f"recall_{name}_pooled"] = pooled[f"recall_{name}"]
    save_json(summary, out_dir / "summary.json")

    rows = [{"Metric": k, "Mean ± SD over folds (%)": mean_sd_pct(summary[f"{k}_mean"], summary[f"{k}_sd"]),
             "Pooled (%)": pct(summary[f"{k}_pooled"])} for k in METRICS]
    per_fold = fold_df[["fold", "test_subjects", "n_windows", "accuracy", "f1_macro", "best_epoch"]].copy()
    per_fold["accuracy"] = per_fold["accuracy"].map(pct)
    per_fold["f1_macro"] = per_fold["f1_macro"].map(pct)
    text = (f"# {PROTOCOL_NAMES[mode]}\n\nSubjects: {summary['n_subjects']}, videos: {summary['n_videos']}, "
            f"windows: {summary['n_windows']}, folds: {summary['n_folds']}\n\n"
            f"{md_table(pd.DataFrame(rows))}\n\n## Per fold\n\n{md_table(per_fold)}\n")
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    plot_confusion(oof["y_true"], oof["y_pred"], out_dir / "confusion_matrix.png",
                   f"{PROTOCOL_NAMES[mode]} (pooled)")
    return summary


def write_comparison(summaries, out):
    rows = []
    for s in summaries:
        rows.append({
            "Protocol": s["protocol"], "Folds": s["n_folds"],
            "Accuracy (%) mean ± SD": mean_sd_pct(s["accuracy_mean"], s["accuracy_sd"]),
            "Accuracy (%) pooled": pct(s["accuracy_pooled"]),
            "Macro-F1 (%) mean ± SD": mean_sd_pct(s["f1_macro_mean"], s["f1_macro_sd"]),
            "Macro-F1 (%) pooled": pct(s["f1_macro_pooled"]),
            "Weighted-F1 (%) pooled": pct(s["f1_weighted_pooled"]),
            "Warning-class sensitivity (%) pooled": pct(s["sensitivity_warning_pooled"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / "split_comparison.csv", index=False)
    (Path(out) / "split_comparison.md").write_text(
        "# Effect of the evaluation protocol\n\n" + md_table(df) + "\n", encoding="utf-8")
    print("\n" + md_table(df))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="frame-level CSV/Parquet")
    ap.add_argument("--mode", default="loso", choices=["loso", "video", "random", "all"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--no-save-models", action="store_true")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    df = load_frames(args.frames)
    feats = feature_columns(df)
    X, y, meta = make_windows(df, feats, with_labels=True)
    print(f"{len(df)} frames, {df[C.SUBJECT_COL].nunique()} subjects, {df[C.VIDEO_COL].nunique()} videos "
          f"-> {len(X)} windows of {C.WINDOW} x {len(feats)} features")
    counts = np.bincount(y, minlength=len(C.CLASS_NAMES))
    print("window labels:", dict(zip(C.CLASS_NAMES, counts.tolist())))

    modes = ["random", "video", "loso"] if args.mode == "all" else [args.mode]
    summaries = [run_protocol(m, X, y, meta, args) for m in modes]
    if len(summaries) > 1:
        write_comparison(summaries, args.out)


if __name__ == "__main__":
    main()
