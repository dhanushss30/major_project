"""
train_novel_components.py — Novel Contributions Pipeline
=========================================================
Trains and saves the three novel components added to the combined solution:

  1. Per-Class Calibration (Contribution #3)
     → Fit temperature scaler + adaptive thresholds on OOF predictions
     → Used in all subsequent pseudo-label generation

  2. Species Co-occurrence GCN (Contribution #2)
     → Build co-occurrence matrix from soundscape OOF predictions
     → Train 2-layer GCN on OOF data
     → Applied at inference time to refine final predictions

  NOTE: DANN (Contribution #1) is activated via config flags during
        main_train.py — set "use_dann": True in your train config.

Usage:
    # After supervised training completes all 5 folds:
    python scripts/train_novel_components.py \\
        --manifest logdir/eca_nfnet_l0_.../checkpoint_manifest.json \\
        --data_root D:/birdclef_data/birdclef-2025 \\
        --soundscape_root D:/birdclef_data/birdclef-2025/train_soundscapes \\
        --output_dir logdir/novel_components \\
        --backbone eca_nfnet_l0

Run order:
    1. python scripts/main_train.py configs/train/selected_eca.py  (all 5 folds)
    2. python scripts/train_novel_components.py  ← THIS SCRIPT
    3. python scripts/generate_pseudo_labels.py --use_calibration  (improved pseudo-labels)
    4. python scripts/main_train.py configs/train/pseudo_iter1_eca.py
    5. python scripts/main_inference.py --use_gcn  (improved final predictions)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code_base.models.backbone import BirdCLEFModel
from code_base.augmentations.nnaudio_mel import MelExtractor
from code_base.datasets.audio_dataset import BirdCLEFDataset, SoundscapeDataset
from code_base.utils.postprocessing import padded_auc_score
from code_base.utils.calibration import (
    run_calibration_pipeline,
    PerClassTemperatureScaler,
    AdaptiveThresholdSelector,
)
from code_base.models.cooccurrence_graph import (
    build_cooccurrence_matrix,
    SpeciesCooccurrenceGCN,
    train_cooccurrence_gcn,
    save_gcn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_fold_models(manifest_path: str, backbone: str, n_classes: int = 206):
    """Load all fold checkpoints from a training manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    models = {}
    for fold_key, ckpt_path in manifest.items():
        if not Path(ckpt_path).exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}, skipping")
            continue

        model = BirdCLEFModel(
            backbone_name = backbone,
            n_classes     = n_classes,
            pretrained    = False,
        )

        state = torch.load(ckpt_path, map_location="cpu")
        # Unwrap Lightning checkpoint
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("model.", ""): v for k, v in state.items()
                 if k.startswith("model.")}
        model.load_state_dict(state, strict=False)
        model.eval()

        fold_idx = int(fold_key.split("_")[-1])
        models[fold_idx] = model
        logger.info(f"Loaded {fold_key}: {ckpt_path}")

    return models


@torch.no_grad()
def get_oof_predictions(
    models:       dict,           # {fold_idx: model}
    df:           pd.DataFrame,   # metadata with 'fold' column
    label_to_idx: dict,
    cfg:          dict,
    device:       torch.device,
) -> tuple:
    """
    Get out-of-fold predictions: for each fold, use the fold's model
    to predict only the validation samples of that fold.

    Returns:
        oof_logits  : (N_total, N_classes) — raw logits
        oof_probs   : (N_total, N_classes) — sigmoid probabilities
        oof_targets : (N_total, N_classes) — ground truth labels
        oof_df      : DataFrame with filename, fold, etc.
    """
    mel_extractor = MelExtractor(
        sr         = cfg.get("sr", 32_000),
        n_mels     = cfg.get("n_mels", 128),
        fmin       = cfg.get("fmin", 20.0),
        n_fft      = cfg.get("n_fft", 2_048),
        hop_length = cfg.get("hop_length", 512),
        top_db     = cfg.get("top_db", 80.0),
        amin       = cfg.get("amin", 1e-10),
    ).to(device)

    all_logits  = []
    all_targets = []
    all_rows    = []

    for fold_idx, model in models.items():
        model = model.to(device)
        val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

        val_ds = BirdCLEFDataset(
            df             = val_df,
            hdf5_root      = cfg.get("hdf5_root"),
            audio_root     = cfg.get("audio_root"),
            label_to_idx   = label_to_idx,
            n_classes      = cfg.get("n_classes", 206),
            sr             = cfg.get("sr", 32_000),
            chunk_strategy = "random",
            augment        = False,
            is_train       = False,
            label_smoothing= 0.0,
        )

        loader = torch.utils.data.DataLoader(
            val_ds, batch_size=32, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        fold_logits  = []
        fold_targets = []

        for batch in loader:
            audio  = batch["audio"].to(device)
            labels = batch["label"]

            mel    = mel_extractor(audio)
            logits = model.get_logits(mel)

            fold_logits.append(logits.cpu().float().numpy())
            fold_targets.append(labels.float().numpy())

        fold_logits  = np.concatenate(fold_logits,  axis=0)
        fold_targets = np.concatenate(fold_targets, axis=0)

        all_logits.append(fold_logits)
        all_targets.append(fold_targets)
        all_rows.append(val_df)

        auc = padded_auc_score(fold_targets, 1 / (1 + np.exp(-fold_logits)))
        logger.info(f"Fold {fold_idx} OOF AUC: {auc:.4f}")

        model.cpu()

    oof_logits  = np.concatenate(all_logits,  axis=0)
    oof_targets = np.concatenate(all_targets, axis=0)
    oof_df      = pd.concat(all_rows,         ignore_index=True)

    overall_auc = padded_auc_score(oof_targets, 1 / (1 + np.exp(-oof_logits)))
    logger.info(f"Overall OOF AUC (raw): {overall_auc:.4f}")

    return oof_logits, 1 / (1 + np.exp(-oof_logits)), oof_targets, oof_df


@torch.no_grad()
def get_soundscape_oof_predictions(
    models:       dict,
    soundscape_dir: str,
    cfg:          dict,
    device:       torch.device,
) -> tuple:
    """
    Get predictions on soundscape files for building co-occurrence matrix.
    Uses OOF strategy: for file f, use models NOT trained on fold(f).
    """
    mel_extractor = MelExtractor(
        sr         = cfg.get("sr", 32_000),
        n_mels     = cfg.get("n_mels", 128),
        fmin       = cfg.get("fmin", 20.0),
        n_fft      = cfg.get("n_fft", 2_048),
        hop_length = cfg.get("hop_length", 512),
        top_db     = cfg.get("top_db", 80.0),
        amin       = cfg.get("amin", 1e-10),
    ).to(device)

    all_probs    = []
    all_file_ids = []

    n_folds = len(models)

    for fold_idx, model in models.items():
        model = model.to(device)

        # Load soundscape files assigned to this fold's validation set
        sc_ds = SoundscapeDataset(
            soundscape_dir = soundscape_dir,
            sr             = cfg.get("sr", 32_000),
            fold_id        = fold_idx,
            n_folds        = n_folds,
        )

        if len(sc_ds) == 0:
            logger.warning(f"No soundscape chunks for fold {fold_idx}")
            continue

        loader = torch.utils.data.DataLoader(
            sc_ds, batch_size=32, shuffle=False, num_workers=0,
        )

        for batch in loader:
            audio    = batch["audio"].to(device)
            file_ids = batch["file_fold"]

            mel   = mel_extractor(audio)
            probs = model.get_probabilities(mel)

            all_probs.append(probs.cpu().float().numpy())
            all_file_ids.extend(file_ids.numpy().tolist())

        model.cpu()

    if not all_probs:
        raise RuntimeError("No soundscape predictions generated")

    sc_probs    = np.concatenate(all_probs, axis=0)
    sc_file_ids = np.array(all_file_ids)

    logger.info(f"Soundscape OOF predictions: {len(sc_probs)} chunks, "
               f"{sc_file_ids.max()+1} files")

    return sc_probs, sc_file_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",        required=True,
                        help="Path to checkpoint_manifest.json from main_train.py")
    parser.add_argument("--data_root",       required=True)
    parser.add_argument("--soundscape_root", required=True)
    parser.add_argument("--output_dir",      default="logdir/novel_components")
    parser.add_argument("--backbone",        default="eca_nfnet_l0")
    parser.add_argument("--metadata_file",   default="train.csv")
    parser.add_argument("--audio_root",      default=None)
    parser.add_argument("--hdf5_root",       default=None)
    parser.add_argument("--n_classes",       type=int, default=206)
    parser.add_argument("--target_precision",type=float, default=0.80,
                        help="Target precision for adaptive threshold selection")
    parser.add_argument("--gcn_epochs",      type=int, default=30)
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load metadata ─────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    df = pd.read_csv(data_root / args.metadata_file)

    # Load folds from logdir (saved by main_train.py)
    manifest_dir = Path(args.manifest).parent
    df_folds_path = manifest_dir / "metadata_with_folds.csv"
    if df_folds_path.exists():
        df = pd.read_csv(df_folds_path)
        logger.info(f"Loaded metadata with folds: {len(df)} samples")
    else:
        raise FileNotFoundError(f"metadata_with_folds.csv not found in {manifest_dir}. "
                               "Run main_train.py first.")

    # Label mapping
    with open(manifest_dir / "label_to_idx.json") as f:
        label_to_idx = json.load(f)

    cfg = {
        "sr": 32_000, "n_mels": 128, "fmin": 20.0,
        "n_fft": 2_048, "hop_length": 512, "top_db": 80.0, "amin": 1e-10,
        "n_classes":  args.n_classes,
        "audio_root": args.audio_root,
        "hdf5_root":  args.hdf5_root,
    }

    # ── Load trained fold models ──────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("Loading trained fold models...")
    models = load_fold_models(args.manifest, args.backbone, args.n_classes)

    if not models:
        raise RuntimeError("No models loaded. Check checkpoint paths in manifest.")

    # ── Step 1: OOF predictions on validation data ────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Computing OOF predictions on labeled validation data")
    logger.info("="*60)

    oof_logits, oof_probs, oof_targets, oof_df = get_oof_predictions(
        models, df, label_to_idx, cfg, device
    )

    np.save(str(output_dir / "oof_logits.npy"),  oof_logits)
    np.save(str(output_dir / "oof_probs.npy"),   oof_probs)
    np.save(str(output_dir / "oof_targets.npy"), oof_targets)
    logger.info(f"OOF predictions saved to {output_dir}")

    # ── Step 2: Per-class calibration (Contribution #3) ──────────────────
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Per-class calibration + adaptive thresholds")
    logger.info("="*60)

    scaler, selector = run_calibration_pipeline(
        oof_logits       = oof_logits,
        oof_targets      = oof_targets,
        save_dir         = str(output_dir / "calibration"),
        target_precision = args.target_precision,
        n_temp_iters     = 500,
    )

    # Measure AUC improvement from calibration alone
    calibrated_probs = scaler(
        torch.tensor(oof_logits, dtype=torch.float32)
    ).numpy()
    raw_auc  = padded_auc_score(oof_targets, oof_probs)
    cal_auc  = padded_auc_score(oof_targets, calibrated_probs)
    logger.info(f"AUC before calibration: {raw_auc:.4f}")
    logger.info(f"AUC after calibration:  {cal_auc:.4f}  (delta: {cal_auc-raw_auc:+.4f})")

    # ── Step 3: Soundscape OOF predictions for co-occurrence ─────────────
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Soundscape OOF predictions for co-occurrence matrix")
    logger.info("="*60)

    sc_probs, sc_file_ids = get_soundscape_oof_predictions(
        models, args.soundscape_root, cfg, device
    )

    np.save(str(output_dir / "sc_probs.npy"),    sc_probs)
    np.save(str(output_dir / "sc_file_ids.npy"), sc_file_ids)

    # ── Step 4: Build co-occurrence matrix ───────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Building species co-occurrence matrix")
    logger.info("="*60)

    A = build_cooccurrence_matrix(
        predictions = sc_probs,
        file_ids    = sc_file_ids,
        threshold   = 0.3,
        smoothing   = 1.0,
        normalize   = True,
    )
    torch.save(A, str(output_dir / "cooccurrence_matrix.pt"))
    logger.info(f"Co-occurrence matrix saved: {A.shape}")

    # ── Step 5: Train Species GCN (Contribution #2) ──────────────────────
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Training Species Co-occurrence GCN")
    logger.info("="*60)

    gcn = SpeciesCooccurrenceGCN(n_classes=args.n_classes, hidden_dim=128)
    gcn.set_adjacency(A)

    gcn = train_cooccurrence_gcn(
        gcn         = gcn,
        oof_probs   = oof_probs,        # Use raw probs for GCN training
        oof_targets = oof_targets,
        n_epochs    = args.gcn_epochs,
        lr          = 1e-3,
        device      = args.device,
    )

    save_gcn(gcn, str(output_dir / "species_gcn.pt"))

    # Final AUC with GCN
    gcn.eval().to(device)
    with torch.no_grad():
        refined_probs = gcn(
            torch.tensor(oof_probs, dtype=torch.float32).to(device)
        ).cpu().numpy()

    gcn_auc = padded_auc_score(oof_targets, refined_probs)
    logger.info(f"\n{'='*60}")
    logger.info("NOVEL COMPONENTS TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Raw ensemble AUC:           {raw_auc:.4f}")
    logger.info(f"  + Calibration:              {cal_auc:.4f}  ({cal_auc-raw_auc:+.4f})")
    logger.info(f"  + GCN refinement:           {gcn_auc:.4f}  ({gcn_auc-raw_auc:+.4f})")
    logger.info(f"\nArtifacts saved to: {output_dir}")
    logger.info("Next: python scripts/generate_pseudo_labels.py --use_calibration")


if __name__ == "__main__":
    main()
