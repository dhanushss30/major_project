"""predict_cli.py — Per-5-second species prediction CLI for demos.

Loads the project's standard ensemble (configured via --manifest or default
hard-coded fold 0 + fold 1 + fold 2 + 0.7841 FiLM ckpt), predicts on a single
audio file, prints a human-readable per-chunk breakdown with confidence bars
and the open-set "not a bird" verdict.

Usage:
    python scripts/predict_cli.py --audio /path/to/audio.wav
    python scripts/predict_cli.py --audio /path/to/audio.wav --top_k 5
    python scripts/predict_cli.py --audio /path/to/audio.wav --no_bird_thresh 0.30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_inference import ensemble_predict, load_ckpt


# ============================================================================
# Defaults — point to the actual checkpoints we trained
# ============================================================================
import glob

_V3_DIR = "/workspace/major_project/birdclef_combined/logdir/_v3_eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/fold_0/checkpoints"
_F1_DIR = "/workspace/major_project/birdclef_combined/logdir/eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/fold_1/checkpoints"


def _best_fold_ckpt(ckpt_dir: str) -> str | None:
    """Return the highest-AUC 'best-epoch*-aucX.YYYY.ckpt' in a fold dir."""
    candidates = glob.glob(f"{ckpt_dir}/best-epoch*-auc*.ckpt")
    if not candidates:
        return None
    import re
    def auc_from(path):
        m = re.search(r"auc([\d.]+)\.ckpt$", path)
        return float(m.group(1)) if m else 0.0
    return max(candidates, key=auc_from)


# Build the default ensemble dynamically — handles missing ckpts gracefully
DEFAULT_CKPTS = [c for c in [
    # 1. v3 fold 0 clean (0.7756)
    f"{_V3_DIR}/best-epoch07-auc0.7756.ckpt",
    # 2. v3 fold 0 FiLM-enabled (0.7841) — included for ensemble diversity
    f"{_V3_DIR}/best-epoch06-auc0.7841.ckpt",
    # 3. ESC-50 BG-trained peak (0.6814) — noise-augmented training run
    "/workspace/esc50_bg_peak.ckpt",
    # 4. New fold 1 clean (best ckpt, auto-detected)
    _best_fold_ckpt(_F1_DIR),
] if c]

DEFAULT_TRAIN_CSV = "/workspace/birdclef-2025/train.csv"


def load_label_names(train_csv: str) -> list:
    """Load 206-class label list, sorted alphabetically (matches training)."""
    df = pd.read_csv(train_csv)
    return sorted(df["primary_label"].unique())


def confidence_bar(p: float, width: int = 30) -> str:
    """Render a probability as an ASCII bar."""
    filled = int(round(p * width))
    return "█" * filled + "░" * (width - filled)


def main():
    p = argparse.ArgumentParser(description="Per-5-sec bird species prediction.")
    p.add_argument("--audio",          required=True, help="Audio file (any format).")
    p.add_argument("--ckpts",          nargs="+", default=None,
                   help="Override checkpoint paths (default: project ensemble).")
    p.add_argument("--train_csv",      default=DEFAULT_TRAIN_CSV)
    p.add_argument("--top_k",          type=int, default=3)
    p.add_argument("--no_bird_thresh", type=float, default=0.10,
                   help="If max-class probability < threshold, mark chunk as 'no bird'. "
                        "0.10 calibrated for this model (BCE+label-smoothing produces conservative probs).")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--tta_hop",        type=float, default=2.5)
    p.add_argument("--noise_preprocess", action="store_true", default=True,
                   help="Apply spectral noise gating + bandpass + pre-emphasis before inference.")
    p.add_argument("--no_noise_preprocess", action="store_false", dest="noise_preprocess",
                   help="Disable noise preprocessing (raw audio to model).")
    p.add_argument("--consistency_boost", type=float, default=0.05,
                   help="Boost confidence on classes appearing in adjacent chunks (0 = off).")
    p.add_argument("--quiet",          action="store_true",
                   help="Suppress ckpt-load messages.")
    args = p.parse_args()

    ckpt_paths = args.ckpts or DEFAULT_CKPTS
    ckpt_paths = [c for c in ckpt_paths if Path(c).exists()]
    if not ckpt_paths:
        print("[ERROR] no checkpoints found. Specify --ckpts or train folds first.")
        sys.exit(1)

    labels = load_label_names(args.train_csv)
    assert len(labels) == 206, f"Expected 206 labels in train.csv, got {len(labels)}"

    if not args.quiet:
        print(f"loading {len(ckpt_paths)} ckpt(s):")
    loaded = []
    for c in ckpt_paths:
        lc = load_ckpt(c, device=args.device)
        loaded.append(lc)
        if not args.quiet:
            tag = "FiLM" if lc.has_film else "clean"
            print(f"  - {Path(c).name}  [{tag}]  val_auc={lc.val_auc}")
    if not args.quiet:
        print()

    print(f"audio: {args.audio}")
    print(f"noise preprocess: {args.noise_preprocess}  consistency boost: {args.consistency_boost}")
    probs = ensemble_predict(
        audio_path             = args.audio,
        ckpts                  = loaded,
        device                 = args.device,
        tta_hop_sec            = args.tta_hop,
        noise_preprocess       = args.noise_preprocess,
        consistency_vote_boost = args.consistency_boost,
    )
    n_chunks, n_classes = probs.shape
    print(f"duration: ~{n_chunks * 5}s ({n_chunks} × 5-sec chunk(s))")
    print()

    n_no_bird = 0
    for i in range(n_chunks):
        order = np.argsort(-probs[i])[:args.top_k]
        max_p = probs[i, order[0]]
        if max_p < args.no_bird_thresh:
            n_no_bird += 1
            verdict = f"\033[33m[no bird detected]\033[0m  (max prob = {max_p:.3f})"
            print(f"  chunk {i:>2} ({i*5:>3}-{(i+1)*5:>3}s):  {verdict}")
        else:
            print(f"  chunk {i:>2} ({i*5:>3}-{(i+1)*5:>3}s):")
            for c in order:
                lbl = labels[c]
                pr  = probs[i, c]
                bar = confidence_bar(pr)
                print(f"      {lbl:<14}  {bar}  {pr:.3f}")
        if i < n_chunks - 1:
            print()

    print()
    print(f"summary: {n_chunks - n_no_bird} chunk(s) with bird detected, "
          f"{n_no_bird} chunk(s) 'no bird' (threshold {args.no_bird_thresh})")


if __name__ == "__main__":
    sys.exit(main())
