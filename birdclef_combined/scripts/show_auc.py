"""
show_auc.py — Show validation AUC from a trained checkpoint
============================================================
Loads the best checkpoint from fold_0, runs it on the validation
set, and prints a clean AUC report for presentation.

Usage (run from birdclef_combined/ directory):
    python scripts/show_auc.py

Options:
    --checkpoint   Path to .ckpt file (auto-detected if omitted)
    --data_root    Dataset root (default: D:/birdclef_data/birdclef-2025)
    --n_samples    Max validation samples to run (default: all)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code_base.models.train_module import BirdCLEFModule
from code_base.utils.postprocessing import padded_auc_score


# ─── Auto-detect best checkpoint ──────────────────────────────────────────

def find_best_checkpoint(logdir: str = "logdir") -> Path:
    """Find the checkpoint with highest AUC in any fold."""
    logdir = Path(logdir)
    candidates = list(logdir.rglob("best-epoch*-auc*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best-*.ckpt found under {logdir}. "
            "Train at least 1 fold first:\n"
            "  python scripts/main_train.py configs/train/selected_eca.py --fold 0"
        )

    def auc_from_name(p: Path) -> float:
        # best-epoch09-auc0.7301.ckpt → 0.7301
        try:
            return float(p.stem.split("auc")[-1])
        except Exception:
            return 0.0

    best = max(candidates, key=auc_from_name)
    return best


# ─── Load metadata and validation set ─────────────────────────────────────

def load_val_dataset(cfg: dict, fold: int = 0):
    """Build the validation DataLoader for a given fold."""
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    from code_base.datasets.audio_dataset import BirdCLEFDataset

    dataset_root = cfg.get("data_root", "D:/birdclef_data/birdclef-2025")
    data_root    = str(Path(dataset_root) / "train_audio")
    meta_path    = Path(dataset_root) / "train.csv"
    meta = pd.read_csv(meta_path)

    # Rebuild same fold split used during training
    logdir_meta = list(Path("logdir").rglob("metadata_with_folds.csv"))
    if logdir_meta:
        meta = pd.read_csv(logdir_meta[0])
        print(f"  [fold split] Loaded from {logdir_meta[0]}")
    else:
        # Recompute
        meta["fold"] = -1
        splitter = StratifiedGroupKFold(n_splits=cfg.get("n_folds", 5), shuffle=True, random_state=42)
        groups  = meta.get("author", meta.index).astype(str)
        labels  = meta["primary_label"].astype("category").cat.codes
        for i, (_, val_idx) in enumerate(splitter.split(meta, labels, groups)):
            meta.loc[val_idx, "fold"] = i

    val_meta  = meta[meta["fold"] == fold].reset_index(drop=True)

    # Load label map
    label_files = list(Path("logdir").rglob("label_to_idx.json"))
    if label_files:
        with open(label_files[0]) as f:
            label_to_idx = json.load(f)
    else:
        labels_sorted = sorted(meta["primary_label"].unique())
        label_to_idx  = {l: i for i, l in enumerate(labels_sorted)}

    n_classes = len(label_to_idx)

    dataset = BirdCLEFDataset(
        df            = val_meta,
        audio_root    = data_root,
        label_to_idx  = label_to_idx,
        n_classes     = n_classes,
        chunk_duration = cfg.get("clip_duration", 5.0),
        sr            = cfg.get("sr", 32_000),
        augment       = False,
        is_train      = False,
    )
    return dataset, n_classes, label_to_idx


# ─── Main evaluation ──────────────────────────────────────────────────────

def run(args):
    print()
    print("=" * 65)
    print("  BirdCLEF 2025 — Validation AUC Report")
    print("=" * 65)

    # ── Find checkpoint ───────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_best_checkpoint("logdir")

    fold = 0
    for part in ckpt_path.parts:
        if part.startswith("fold_"):
            fold = int(part.split("_")[1])
            break

    ckpt_auc = 0.0
    try:
        ckpt_auc = float(ckpt_path.stem.split("auc")[-1])
    except Exception:
        pass

    print(f"\n  Checkpoint  : {ckpt_path.name}")
    print(f"  Fold        : {fold}")
    print(f"  Saved AUC   : {ckpt_auc:.4f}  (from training logs)")

    # ── Load model ────────────────────────────────────────────────────────
    print("\n  Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device      : {device}")

    module = BirdCLEFModule.load_from_checkpoint(
        str(ckpt_path), map_location=device, strict=False
    )
    module.eval()
    cfg = module.cfg

    # ── Load validation data ───────────────────────────────────────────────
    print(f"\n  Building validation set (fold {fold})...")
    dataset, n_classes, label_to_idx = load_val_dataset(cfg, fold=fold)

    if args.n_samples and args.n_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.n_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"  Subset      : {args.n_samples} samples (--n_samples)")
    else:
        print(f"  Val samples : {len(dataset):,}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = 32,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = device.type == "cuda",
    )

    # ── Run inference ─────────────────────────────────────────────────────
    print(f"\n  Running inference on {len(dataset):,} clips...")
    all_probs   = []
    all_targets = []
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            audio  = batch["audio"].to(device)
            labels = batch["label"].to(device)

            mel    = module.mel(audio)
            probs  = module._inference_model().get_probabilities(mel)

            all_probs.append(probs.float().cpu().numpy())
            all_targets.append(labels.float().cpu().numpy())

            if (i + 1) % 20 == 0:
                done = (i + 1) * 32
                print(f"    {done}/{len(dataset)} clips processed...", end="\r")

    elapsed = time.time() - t0
    probs_np   = np.concatenate(all_probs,   axis=0)  # (N, C)
    targets_np = np.concatenate(all_targets, axis=0)  # (N, C)

    # ── Compute AUC ───────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score

    # Macro AUC (competition metric)
    auc_macro = padded_auc_score(targets_np, probs_np)

    # Per-class AUC
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    per_class_aucs = []
    for c in range(n_classes):
        pos = targets_np[:, c].sum()
        if pos >= 2:
            try:
                a = roc_auc_score(targets_np[:, c], probs_np[:, c])
                per_class_aucs.append((idx_to_label.get(c, f"species_{c}"), a, int(pos)))
            except Exception:
                pass

    per_class_aucs.sort(key=lambda x: x[1])

    # ── Print results ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  VALIDATION AUC (Macro): {auc_macro:.4f}")
    print("=" * 65)
    print()
    print(f"  Clips evaluated   : {len(probs_np):,}")
    print(f"  Classes evaluated : {len(per_class_aucs)} / {n_classes}")
    print(f"  Inference time    : {elapsed:.1f}s  ({len(probs_np)/elapsed:.0f} clips/sec)")
    print()
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Macro AUC (competition metric)':<40} {auc_macro:>10.4f}")
    if per_class_aucs:
        print(f"  {'Mean per-class AUC':<40} {np.mean([x[1] for x in per_class_aucs]):>10.4f}")
        print(f"  {'Median per-class AUC':<40} {np.median([x[1] for x in per_class_aucs]):>10.4f}")
        print(f"  {'Min per-class AUC':<40} {per_class_aucs[0][1]:>10.4f}  ({per_class_aucs[0][0]})")
        print(f"  {'Max per-class AUC':<40} {per_class_aucs[-1][1]:>10.4f}  ({per_class_aucs[-1][0]})")
    else:
        print(f"  (Not enough positive samples in subset for per-class AUC)")

    if per_class_aucs:
        # Bottom 10 classes (worst)
        print()
        print("  Hardest classes (lowest AUC — rare species):")
        print(f"  {'Species':<30} {'AUC':>8} {'Train clips':>12}")
        print(f"  {'-'*54}")
        for name, auc, cnt in per_class_aucs[:10]:
            bar = "░" * int(auc * 20)
            print(f"  {name:<30} {auc:>8.4f} {cnt:>12,}  {bar}")

        # Top 10 classes (best)
        print()
        print("  Easiest classes (highest AUC — common species):")
        print(f"  {'Species':<30} {'AUC':>8} {'Train clips':>12}")
        print(f"  {'-'*54}")
        for name, auc, cnt in per_class_aucs[-10:][::-1]:
            bar = "█" * int(auc * 20)
            print(f"  {name:<30} {auc:>8.4f} {cnt:>12,}  {bar}")

    print()
    print("=" * 65)
    print("  CONTEXT: Where does 0.73 sit on the leaderboard?")
    print("=" * 65)
    print(f"  Current AUC    : {auc_macro:.4f}  (1 fold, ~10 epochs)")
    print(f"  Target AUC     : 0.9300  (full pipeline, 5 folds, 50 epochs)")
    print(f"  1st place 2025 : 0.9300")
    print(f"  Gap            : {0.93 - auc_macro:+.4f}")
    print()
    print("  Remaining steps to reach 0.93:")
    print("    1. Train all 5 folds × 50 epochs ECA-NFNet-L0   +0.04 AUC")
    print("    2. Train EfficientNetV2-S (5 folds)              +0.02 AUC")
    print("    3. 3× pseudo-label noisy student iterations      +0.06 AUC")
    print("    4. Novel #1 DANN + #5 FiLM noise conditioning    +0.02 AUC")
    print("    5. Novel #2 GCN + #3 calibration + #4 prototypes +0.02 AUC")
    print()
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show validation AUC from trained BirdCLEF checkpoint"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .ckpt file (auto-detected if omitted)")
    parser.add_argument("--data_root",  type=str,
                        default="D:/birdclef_data/birdclef-2025")
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Limit val samples for speed (default: all)")
    args = parser.parse_args()
    run(args)
