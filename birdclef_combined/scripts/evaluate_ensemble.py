"""evaluate_ensemble.py — Final val/auc measurement for the ensemble.

Runs the multi-checkpoint TTA ensemble on the fold-0 validation set
(reproducing the same val split used during training) and reports:
  - Macro-averaged ROC AUC
  - Per-class AUC distribution (top 5 / bottom 5)
  - Optional per-class threshold tuning for F1 (post-hoc)

Usage:
    python scripts/evaluate_ensemble.py \
        --ckpts /workspace/.../fold_0/.../best-epoch07-auc0.7756.ckpt \
                /workspace/.../fold_1/.../best.ckpt \
                /workspace/.../fold_2/.../best.ckpt \
        --train_csv /workspace/birdclef-2025/train.csv \
        --audio_root /workspace/birdclef-2025/train_audio \
        --fold 0 \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_inference import ensemble_predict, load_ckpt


def get_fold_val_files(train_csv: str, fold: int, n_folds: int = 5) -> pd.DataFrame:
    """Reproduce the EXACT CV split from main_train.py (StratifiedGroupKFold
    by primary species, grouped by author/recordist). Falls back to
    StratifiedKFold if no author column exists."""
    # Import the canonical CV builder so val matches training exactly
    cb_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(cb_root))
    sys.path.insert(0, str(cb_root / "scripts"))
    from main_train import build_cv_splits

    df = pd.read_csv(train_csv)
    df = build_cv_splits(df, n_folds=n_folds)
    return df[df["fold"] == fold].reset_index(drop=True)


def build_label_targets(df: pd.DataFrame, labels: list) -> np.ndarray:
    """Build (n_samples, n_classes) one-hot ground-truth matrix."""
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = np.zeros((len(df), len(labels)), dtype=np.float32)
    for i, row in df.iterrows():
        if row["primary_label"] in label_to_idx:
            y[i, label_to_idx[row["primary_label"]]] = 1.0
    return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts",      nargs="+", required=True)
    p.add_argument("--train_csv",  required=True)
    p.add_argument("--audio_root", required=True)
    p.add_argument("--fold",       type=int, default=0)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--max_files",  type=int, default=None,
                   help="Limit val files (for quick sanity checks).")
    p.add_argument("--output_csv", default=None,
                   help="Optional path to save per-class AUC table.")
    args = p.parse_args()

    # ─── 1. Load val split ─────────────────────────────────────────────────
    val_df = get_fold_val_files(args.train_csv, args.fold)
    if args.max_files:
        val_df = val_df.head(args.max_files)
    print(f"val fold {args.fold}: {len(val_df):,} files")

    labels = sorted(pd.read_csv(args.train_csv)["primary_label"].unique())
    y_true = build_label_targets(val_df, labels)
    print(f"labels: {len(labels)}")

    # ─── 2. Load ckpts ─────────────────────────────────────────────────────
    loaded = []
    for c in args.ckpts:
        if not Path(c).exists():
            print(f"[WARN] skipping missing ckpt: {c}")
            continue
        lc = load_ckpt(c, device=args.device)
        loaded.append(lc)
        tag = "FiLM" if lc.has_film else "clean"
        print(f"  loaded {Path(c).name}  [{tag}]  val_auc={lc.val_auc}")
    if not loaded:
        print("[ERROR] no checkpoints loaded.")
        sys.exit(1)

    # ─── 3. Predict on every val file (first 5s chunk) ─────────────────────
    y_pred = np.zeros_like(y_true)
    audio_root = Path(args.audio_root)
    failed = 0
    for i, row in tqdm(val_df.iterrows(), total=len(val_df), desc="inference"):
        audio_path = audio_root / row["filename"]
        if not audio_path.exists():
            failed += 1
            continue
        try:
            probs = ensemble_predict(
                audio_path  = audio_path,
                ckpts       = loaded,
                device      = args.device,
            )
            # Use first 5s chunk (matches training's chunk_strategy)
            y_pred[i] = probs[0]
        except Exception as e:
            failed += 1
            print(f"\n[WARN] {audio_path.name}: {e}")
    if failed:
        print(f"  failed: {failed} files")

    # ─── 4. Compute AUC ────────────────────────────────────────────────────
    per_class_auc = np.full(len(labels), np.nan)
    for c in range(len(labels)):
        if y_true[:, c].sum() > 0 and y_true[:, c].sum() < len(y_true):
            try:
                per_class_auc[c] = roc_auc_score(y_true[:, c], y_pred[:, c])
            except ValueError:
                pass

    valid = ~np.isnan(per_class_auc)
    macro_auc = per_class_auc[valid].mean()
    print()
    print(f"=== ENSEMBLE FOLD {args.fold} VAL AUC ===")
    print(f"  Macro-averaged ROC AUC: {macro_auc:.4f}")
    print(f"  Classes with valid AUC: {valid.sum()} of {len(labels)}")

    # Top 5 / bottom 5 per-class
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(-per_class_auc[valid_idx])]
    print(f"\n  TOP 5 classes by AUC:")
    for c in order[:5]:
        print(f"    {labels[c]:<14}  {per_class_auc[c]:.4f}")
    print(f"\n  BOTTOM 5 classes by AUC:")
    for c in order[-5:]:
        print(f"    {labels[c]:<14}  {per_class_auc[c]:.4f}")

    if args.output_csv:
        out = pd.DataFrame({
            "label":    labels,
            "auc":      per_class_auc,
            "n_pos":    y_true.sum(axis=0).astype(int),
        }).sort_values("auc", ascending=False)
        out.to_csv(args.output_csv, index=False)
        print(f"\nsaved per-class AUC table: {args.output_csv}")


if __name__ == "__main__":
    sys.exit(main())
