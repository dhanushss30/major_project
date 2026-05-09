#!/usr/bin/env python3
"""
generate_pseudo_labels.py — Pseudo-Label Generation Pipeline
=============================================================
Implements exact 2nd place pipeline (§5.4) with 1st place power transform.

Iteration schedule (from paper Table 4):
  Iter 1: threshold=0.5, class_thresh=0.1, power=1.0 → 19,405 chunks (4,430 files)
  Iter 2: threshold=0.5, class_thresh=0.1, power=1.0 →  3,750 chunks (1,483 files)
  Iter 3: threshold=0.5, class_thresh=0.1, power=1.0 →  4,108 chunks (1,437 files)

1st place power transform (applied BEFORE selection):
  Iter 1: alpha=0.6
  Iter 2: alpha=0.5
  Iter 3: alpha=0.4

Modes:
  --mode full : All soundscapes predicted by full ensemble (default)
  --mode oof  : OOF mode — each fold's soundscapes predicted by other folds' models

Usage:
  python scripts/generate_pseudo_labels.py \\
      --checkpoint_manifest logdir/selected_eca/checkpoint_manifest.json \\
      --soundscape_dir /data/unlabeled_soundscapes \\
      --output_dir /data/pseudo \\
      --iteration 1 --power_alpha 0.6 \\
      --backbone eca_nfnet_l0 --n_classes 206

  python scripts/generate_pseudo_labels.py \\
      --checkpoint_manifest logdir/selected_eca/checkpoint_manifest.json \\
      --soundscape_dir /data/unlabeled_soundscapes \\
      --output_dir /data/pseudo --mode oof \\
      --iteration 2 --power_alpha 0.5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code_base.models.backbone import BirdCLEFModel
from code_base.augmentations.nnaudio_mel import MelExtractor
from code_base.datasets.audio_dataset import SoundscapeDataset
from code_base.utils.pseudo_labels import PseudoLabelGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_fold_models(
    manifest_path: str,
    backbone:      str,
    n_classes:     int,
    device:        torch.device,
) -> List[BirdCLEFModel]:
    """Load all fold models from a checkpoint manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    models = []
    for fold_name, ckpt_path in manifest.items():
        if not ckpt_path or not Path(ckpt_path).exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}, skipping {fold_name}")
            continue

        model = BirdCLEFModel(
            backbone_name = backbone,
            n_classes     = n_classes,
            pretrained    = False,
        )

        # Load from Lightning checkpoint
        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
            # Strip "model." prefix added by Lightning
            state = {k.replace("model.", "", 1): v for k, v in state.items()
                     if k.startswith("model.")}

        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()
        models.append(model)
        logger.info(f"  Loaded {fold_name}: {ckpt_path}")

    return models


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint_manifest", required=True,
                   help="Path to checkpoint_manifest.json from training")
    p.add_argument("--soundscape_dir",      required=True)
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--backbone",            default="eca_nfnet_l0")
    p.add_argument("--n_classes",           type=int,   default=206)
    p.add_argument("--iteration",           type=int,   default=1)
    p.add_argument("--max_prob_threshold",  type=float, default=0.5,
                   help="Discard chunks with max prob below this (paper: 0.5)")
    p.add_argument("--class_prob_threshold",type=float, default=0.1,
                   help="Zero out class probs below this (paper: 0.1)")
    p.add_argument("--power_alpha",         type=float, default=1.0,
                   help="Power transform exponent (1st place: 0.6/0.5/0.4)")
    p.add_argument("--mode",                choices=["full", "oof"], default="full")
    p.add_argument("--n_folds",             type=int,   default=5)
    p.add_argument("--batch_size",          type=int,   default=64)
    p.add_argument("--num_workers",         type=int,   default=0,
                   help="0 is safest on Windows; use 2-4 on Linux")
    p.add_argument("--sr",                  type=int,   default=32_000)
    # Novel #9: MC Dropout Uncertainty
    p.add_argument("--use_mc_dropout",      action="store_true",
                   help="Enable MC Dropout uncertainty filtering (Novel #9)")
    p.add_argument("--n_mc_samples",        type=int,   default=8,
                   help="Number of MC forward passes per model (T)")
    p.add_argument("--mc_min_confidence",   type=float, default=0.3,
                   help="Reject pseudo-labels below this confidence score")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    logger.info(f"Loading models from {args.checkpoint_manifest}...")
    models = load_fold_models(
        args.checkpoint_manifest, args.backbone, args.n_classes, device
    )
    if not models:
        logger.error("No models loaded! Check checkpoint_manifest.json.")
        sys.exit(1)
    logger.info(f"Loaded {len(models)} fold models")

    # Mel extractor (exact paper params)
    mel = MelExtractor(
        sr         = args.sr,
        n_mels     = 128,
        fmin       = 20.0,
        n_fft      = 2_048,
        hop_length = 512,
    )

    # Soundscape dataset
    logger.info(f"Loading soundscapes from {args.soundscape_dir}...")
    soundscape_ds = SoundscapeDataset(
        soundscape_dir = args.soundscape_dir,
        sr             = args.sr,
        n_folds        = args.n_folds,
    )
    logger.info(f"Soundscape chunks: {len(soundscape_ds):,}")

    # Generate
    generator = PseudoLabelGenerator(
        models       = models,
        mel_extractor = mel,
        soundscape_dataset = soundscape_ds,
        device       = device,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        n_classes    = args.n_classes,
    )

    output_path = output_dir / f"pseudo_labels_iter{args.iteration}_{args.mode}.parquet"

    if args.mode == "full":
        pseudo_df = generator.generate(
            iteration            = args.iteration,
            max_prob_threshold   = args.max_prob_threshold,
            class_prob_threshold = args.class_prob_threshold,
            power_alpha          = args.power_alpha,
            output_path          = output_path,
            use_mc_dropout       = args.use_mc_dropout,
            n_mc_samples         = args.n_mc_samples,
            mc_min_confidence    = args.mc_min_confidence,
        )
    else:  # OOF
        # Build fold→models mapping
        with open(args.checkpoint_manifest) as f:
            manifest = json.load(f)

        fold_models = {}
        for fold_idx, (fold_name, ckpt_path) in enumerate(manifest.items()):
            if not ckpt_path or not Path(ckpt_path).exists():
                continue
            fold_models[fold_idx] = [models[fold_idx % len(models)]]

        pseudo_df = generator.generate_oof(
            fold_models          = fold_models,
            iteration            = args.iteration,
            max_prob_threshold   = args.max_prob_threshold,
            class_prob_threshold = args.class_prob_threshold,
            power_alpha          = args.power_alpha,
            output_path          = output_path,
        )

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Pseudo-label generation complete (Iteration {args.iteration}, {args.mode})")
    logger.info(f"  Selected chunks  : {len(pseudo_df):,}")
    logger.info(f"  Unique files     : {pseudo_df['filename'].nunique():,}")
    logger.info(f"  Power alpha      : {args.power_alpha}")
    logger.info(f"  Max prob thresh  : {args.max_prob_threshold}")
    logger.info(f"  Class thresh     : {args.class_prob_threshold}")
    logger.info(f"  Saved to         : {output_path}")
    logger.info(f"{'='*50}")

    # Save summary
    summary = {
        "iteration":            args.iteration,
        "mode":                 args.mode,
        "backbone":             args.backbone,
        "n_models":             len(models),
        "n_soundscape_chunks":  len(soundscape_ds),
        "n_selected":           len(pseudo_df),
        "n_unique_files":       int(pseudo_df["filename"].nunique()),
        "power_alpha":          args.power_alpha,
        "max_prob_threshold":   args.max_prob_threshold,
        "class_prob_threshold": args.class_prob_threshold,
        "use_mc_dropout":       args.use_mc_dropout,
        "n_mc_samples":         args.n_mc_samples if args.use_mc_dropout else None,
        "mc_min_confidence":    args.mc_min_confidence if args.use_mc_dropout else None,
        "output_path":          str(output_path),
    }
    with open(output_dir / f"pseudo_summary_iter{args.iteration}_{args.mode}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
