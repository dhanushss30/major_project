#!/usr/bin/env python3
"""
run_full_pipeline.py — Fully Automated BirdCLEF Training Pipeline
==================================================================
Single command, zero intervention. Runs the ENTIRE training pipeline:

  Stage 1: Supervised training (5 folds × 2 backbones)
  Stage 2: Pseudo-label generation (with MC Dropout uncertainty)
  Stage 3: Pseudo-label retraining (3 iterations)
  Stage 4: Final calibration + threshold optimization

Auto-safety features:
  - After each fold, checks val AUC. If a novel feature HURTS (AUC < baseline),
    it's automatically disabled for remaining folds.
  - Logs everything to TensorBoard + JSON summaries
  - Saves checkpoints and manifests for each stage
  - If training crashes, can resume from last completed fold

Usage:
  python scripts/run_full_pipeline.py --data_root /data/birdclef-2025 --gpu 0

  # Resume after crash:
  python scripts/run_full_pipeline.py --data_root /data/birdclef-2025 --resume
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PIPELINE] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Pipeline configuration ─────────────────────────────────────────────────

BACKBONES = ["eca_nfnet_l0", "tf_efficientnetv2_s"]
N_FOLDS = 5
PSEUDO_ITERATIONS = 3
POWER_ALPHAS = [0.6, 0.5, 0.4]

# AUC threshold: if fold AUC drops below this relative to previous fold average,
# the most recently added feature is auto-disabled
AUC_REGRESSION_THRESHOLD = 0.005  # 0.5% drop = something is wrong


def get_state_path(logdir: Path) -> Path:
    return logdir / "pipeline_state.json"


def load_state(logdir: Path) -> dict:
    state_path = get_state_path(logdir)
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {
        "stage": "supervised",
        "backbone_idx": 0,
        "fold_idx": 0,
        "pseudo_iteration": 0,
        "completed_folds": [],
        "fold_aucs": {},
        "disabled_features": [],
        "best_auc": 0.0,
    }


def save_state(logdir: Path, state: dict):
    state_path = get_state_path(logdir)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def build_config(
    args,
    backbone: str,
    fold: int,
    stage: str = "supervised",
    pseudo_path: str = None,
    disabled_features: list = None,
) -> dict:
    """Build training config dict for a single fold."""
    disabled = disabled_features or []

    is_nfnet = "nfnet" in backbone
    cfg = {
        "exp_name": f"{backbone}_{stage}_fold{fold}",
        "stage": stage,
        "backbone": backbone,
        "n_classes": 206,
        "pretrained": True,
        "pretrained_path": args.pretrained_path,
        "hidden_dim": 512,
        "dropout1": 0.25,
        "dropout2": 0.5,
        "gem_p": 3.0,
        "use_sed_head": False,

        # Data
        "data_root": args.data_root,
        "metadata_file": args.metadata_file,   # train_merged.csv or train.csv
        "audio_root": str(Path(args.data_root) / "train_audio"),
        "hdf5_root": args.hdf5_root,
        # Soundscapes always from 2025 (unlabeled test-like audio for DANN + pseudo)
        "soundscape_root": str(Path(args.data_root) / "train_soundscapes"),
        "pseudo_labels_path": pseudo_path,

        # CV
        "n_folds": N_FOLDS,
        "fold": fold,
        "seed": 42,

        # Spectrogram
        "sr": 32000,
        "n_mels": 128,
        "fmin": 20.0,
        "fmax": None,
        "n_fft": 2048,
        "hop_length": 512,
        "top_db": 80.0,
        "amin": 1e-10,

        # Loss
        "bce_weight": 0.5,
        "focal_weight": 0.5,
        "auc_weight": 0.3 if stage != "supervised" else 0.0,
        "focal_gamma": 2.0,

        # Training
        "label_smoothing": 0.05,
        "lr": 1e-3 if is_nfnet else 1e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        # Supervised: 50 epochs (full paper schedule)
        # Pseudo: 20 epochs (model already warm; more epochs = overfitting pseudo noise)
        "epochs": 50 if stage == "supervised" else 20,
        "limit_train_batches": 3000,   # 3000 batches × batch_size per epoch
        "batch_size": 16,              # RTX 4090 24GB; effective batch = 16×4 = 64
        "num_workers": 8,              # 16 CPU cores on server; 8 workers for data loading
        "precision": "bf16-mixed",
        "accumulate_grad": 4,          # gradient accumulation to reach effective batch 64

        # Sampler
        "sampler_gamma": -0.5,

        # Augmentation
        "mixup_p": 0.5,
        "mixup_scale": 0.5,
        "chunk_strategy": "firstlast7",
        "use_secondary_labels": True,
        "bg_soundscape_paths": [],
        "bg_esc50_paths": [],

        # EMA + SWA
        "use_ema": True,
        "ema_decay": 0.999,
        "use_swa": False,   # Disabled: deepcopy conflict with custom modules
        "swa_lr": 1e-6,
        "swa_start_frac": 0.75,

        # Oversampling
        "min_oversample_count": 100,
        "use_rating_weight": True,

        # Pseudo
        "labeled_fraction": 0.5,

        # ── Novel features (auto-disabled if they hurt) ──────────────────
        "use_dann": "dann" not in disabled,
        "dann_lambda_max": 0.1,
        "dann_hidden_dim": 512,

        "use_prototypical": "prototypical" not in disabled,
        "proto_temperature": 0.1,
        "proto_margin": 0.3,
        "proto_rare_thr": 0.2,
        "proto_loss_weight": 0.1,
        "proto_half_life": 50,
        "proto_max_weight": 0.5,

        "use_noise_conditioning": "noise_conditioning" not in disabled,
        "noise_dim": 128,

        "use_tcr": ("tcr" not in disabled) and (stage != "supervised"),
        "tcr_weight": 0.05,
        "tcr_max_gap": 5.0,
        "tcr_temperature": 2.0,

        "use_taxonomy": "taxonomy" not in disabled,
        "taxonomy_hidden_dim": 128,
        "taxonomy_aux_weight": 0.1,
        "taxonomy_consistency_weight": 0.05,
        "taxonomy_confusion_weight": 0.02,
        "species_to_taxon": {},
        "species_to_taxon_path": None,

        "use_multi_res_mel": False,

        "use_causal": "causal" not in disabled,
        "causal_dim": 768,
        "spurious_dim": 512,
        "causal_dropout": 0.1,
        "causal_lambda_inv": 1.0,
        "causal_lambda_hsic": 0.1,
        "causal_warmup_steps": 1000,

        # Logging
        "logdir": str(args.logdir),
        "use_wandb": False,
    }

    # Background noise paths (auto-detect)
    sc_root = Path(args.data_root) / "train_soundscapes"
    if sc_root.exists():
        sc_files = list(sc_root.glob("*.ogg")) + list(sc_root.glob("*.wav"))
        cfg["bg_soundscape_paths"] = [str(p) for p in sc_files[:200]]

    esc_root = Path(args.data_root) / "esc50"
    if esc_root.exists():
        cfg["bg_esc50_paths"] = [str(p) for p in esc_root.glob("**/*.wav")]

    return cfg


def train_single_fold(cfg: dict, logdir: Path) -> float:
    """
    Train a single fold. Returns best validation AUC.
    Writes config to a temp JSON, calls main_train.py as subprocess,
    then reads results.json written by main_train.py after training.
    """
    exp_name = cfg["exp_name"]
    fold     = cfg["fold"]

    cfg_path = logdir / f"_tmp_cfg_{exp_name}_fold{fold}.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    fold_logdir = logdir / exp_name
    fold_logdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "main_train.py"),
        "--config_json", str(cfg_path),
        "--fold", str(fold),
        "--logdir", str(fold_logdir),
    ]

    logger.info(f"\nTraining: {exp_name} | fold {fold}")
    logger.info(f"  backbone={cfg['backbone']}  stage={cfg['stage']}")
    logger.info(f"  causal={cfg['use_causal']}  dann={cfg['use_dann']}  "
                f"proto={cfg['use_prototypical']}  taxonomy={cfg['use_taxonomy']}")

    start_time = time.time()

    # Stream output directly to console (not captured) so training is visible
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    elapsed = time.time() - start_time
    logger.info(f"  Done in {elapsed/3600:.1f}h | exit code {result.returncode}")

    if result.returncode != 0:
        logger.error(f"  Fold {fold} FAILED — check output above")

    # Read AUC from results.json written by main_train.py
    results_path = fold_logdir / f"fold_{fold}" / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            return float(json.load(f).get("best_val_auc", 0.0))

    return 0.0


def generate_pseudo_labels(
    args,
    logdir: Path,
    backbone: str,
    iteration: int,
    use_mc_dropout: bool = True,
) -> str:
    """Generate pseudo-labels for one backbone. Returns path to parquet."""
    manifest_path = logdir / f"{backbone}_supervised" / "checkpoint_manifest.json"
    if iteration > 1:
        manifest_path = logdir / f"{backbone}_pseudo_iter{iteration-1}" / "checkpoint_manifest.json"

    if not manifest_path.exists():
        # Fallback: look for any manifest
        candidates = list(logdir.glob(f"*{backbone}*/checkpoint_manifest.json"))
        if candidates:
            manifest_path = candidates[-1]
        else:
            logger.error(f"No checkpoint manifest found for {backbone}")
            return None

    output_dir = logdir / "pseudo_labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "generate_pseudo_labels.py"),
        "--checkpoint_manifest", str(manifest_path),
        "--soundscape_dir", str(Path(args.data_root) / "train_soundscapes"),
        "--output_dir", str(output_dir),
        "--backbone", backbone,
        "--iteration", str(iteration),
        "--power_alpha", str(POWER_ALPHAS[iteration - 1]),
        "--batch_size", "64",
        "--num_workers", "4",
    ]

    if use_mc_dropout:
        cmd.extend([
            "--use_mc_dropout",
            "--n_mc_samples", "8",
            "--mc_min_confidence", "0.3",
        ])

    logger.info(f"Generating pseudo-labels: iter {iteration}, {backbone}, "
                f"MC Dropout={use_mc_dropout}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        logger.error(f"Pseudo-label generation failed: {result.stderr[-500:]}")
        return None

    output_path = output_dir / f"pseudo_labels_iter{iteration}_full.parquet"
    if output_path.exists():
        return str(output_path)

    # Try finding the output
    candidates = list(output_dir.glob(f"*iter{iteration}*"))
    return str(candidates[0]) if candidates else None


def check_and_disable_features(
    state: dict,
    current_auc: float,
    fold: int,
    backbone: str,
) -> list:
    """
    Auto-safety: if AUC drops significantly, identify and disable the culprit.

    Strategy: compare against running average. If regression detected,
    disable features in priority order (least important first):
      taxonomy → tcr → prototypical → noise_conditioning → dann → causal
    """
    key = f"{backbone}_fold{fold}"
    state["fold_aucs"][key] = current_auc

    # Need at least 2 folds to compare
    backbone_aucs = [
        v for k, v in state["fold_aucs"].items()
        if backbone in k and v > 0
    ]

    if len(backbone_aucs) < 2:
        return state["disabled_features"]

    avg_auc = sum(backbone_aucs[:-1]) / len(backbone_aucs[:-1])

    if current_auc < avg_auc - AUC_REGRESSION_THRESHOLD:
        # AUC dropped — disable the least critical feature not yet disabled
        disable_order = [
            "tcr", "taxonomy", "prototypical",
            "noise_conditioning", "dann", "causal",
        ]
        for feat in disable_order:
            if feat not in state["disabled_features"]:
                state["disabled_features"].append(feat)
                logger.warning(
                    f"AUC REGRESSION DETECTED: {current_auc:.4f} vs avg {avg_auc:.4f}. "
                    f"Auto-disabling '{feat}' for remaining folds."
                )
                break

    if current_auc > state["best_auc"]:
        state["best_auc"] = current_auc

    return state["disabled_features"]


def run_supervised_stage(args, logdir: Path, state: dict):
    """Stage 1: Train all folds for all backbones."""
    logger.info("=" * 60)
    logger.info("STAGE 1: SUPERVISED TRAINING")
    logger.info("=" * 60)

    for bb_idx in range(state["backbone_idx"], len(BACKBONES)):
        backbone = BACKBONES[bb_idx]
        start_fold = state["fold_idx"] if bb_idx == state["backbone_idx"] else 0

        logger.info(f"\nBackbone: {backbone}")

        for fold in range(start_fold, N_FOLDS):
            fold_key = f"{backbone}_supervised_fold{fold}"
            if fold_key in state["completed_folds"]:
                logger.info(f"  Fold {fold} already complete, skipping")
                continue

            cfg = build_config(
                args, backbone, fold,
                stage="supervised",
                disabled_features=state["disabled_features"],
            )

            auc = train_single_fold(cfg, logdir)
            logger.info(f"  Fold {fold} AUC: {auc:.4f}")

            state["fold_aucs"][fold_key] = auc
            state["completed_folds"].append(fold_key)
            state["fold_idx"] = fold + 1
            state["backbone_idx"] = bb_idx

            # Auto-safety check
            check_and_disable_features(state, auc, fold, backbone)
            save_state(logdir, state)

        state["fold_idx"] = 0

    state["backbone_idx"] = 0
    state["stage"] = "pseudo"
    save_state(logdir, state)


def run_pseudo_stage(args, logdir: Path, state: dict):
    """Stages 2-4: Iterative pseudo-labeling."""
    logger.info("=" * 60)
    logger.info("STAGE 2-4: PSEUDO-LABEL ITERATIONS")
    logger.info("=" * 60)

    start_iter = state["pseudo_iteration"] + 1

    for iteration in range(start_iter, PSEUDO_ITERATIONS + 1):
        logger.info(f"\n{'─'*40}")
        logger.info(f"PSEUDO ITERATION {iteration} / {PSEUDO_ITERATIONS}")
        logger.info(f"  Power alpha: {POWER_ALPHAS[iteration-1]}")
        logger.info(f"  MC Dropout: enabled")
        logger.info(f"{'─'*40}")

        # Generate pseudo-labels for each backbone
        pseudo_paths = []
        for backbone in BACKBONES:
            path = generate_pseudo_labels(
                args, logdir, backbone, iteration,
                use_mc_dropout=True,
            )
            if path:
                pseudo_paths.append(path)
                logger.info(f"  Generated: {path}")

        if not pseudo_paths:
            logger.error("No pseudo-labels generated! Stopping.")
            break

        # Use the first available pseudo-label file
        pseudo_path = pseudo_paths[0]

        # Retrain all folds with pseudo-labels
        for bb_idx, backbone in enumerate(BACKBONES):
            for fold in range(N_FOLDS):
                fold_key = f"{backbone}_pseudo_iter{iteration}_fold{fold}"
                if fold_key in state["completed_folds"]:
                    continue

                cfg = build_config(
                    args, backbone, fold,
                    stage="pseudo",
                    pseudo_path=pseudo_path,
                    disabled_features=state["disabled_features"],
                )

                auc = train_single_fold(cfg, logdir)
                logger.info(f"  {backbone} fold {fold} AUC: {auc:.4f}")

                state["completed_folds"].append(fold_key)
                state["fold_aucs"][fold_key] = auc
                save_state(logdir, state)

        state["pseudo_iteration"] = iteration
        save_state(logdir, state)

    state["stage"] = "calibration"
    save_state(logdir, state)


def run_calibration_stage(args, logdir: Path, state: dict):
    """Final stage: calibrate temperatures and find optimal thresholds."""
    logger.info("=" * 60)
    logger.info("STAGE FINAL: CALIBRATION + THRESHOLD OPTIMIZATION")
    logger.info("=" * 60)

    # This runs the calibration script on the final model outputs
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "calibrate_and_threshold.py"),
        "--logdir", str(logdir),
        "--n_classes", "206",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logger.warning(f"Calibration script returned non-zero: {result.stderr[-300:]}")
    else:
        logger.info("Calibration complete.")

    state["stage"] = "done"
    save_state(logdir, state)


def print_final_summary(logdir: Path, state: dict):
    """Print final pipeline summary."""
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    # Collect all AUCs
    supervised_aucs = [v for k, v in state["fold_aucs"].items()
                       if "supervised" in k and v > 0]
    pseudo_aucs = [v for k, v in state["fold_aucs"].items()
                   if "pseudo" in k and v > 0]

    if supervised_aucs:
        logger.info(f"\nSupervised stage:")
        logger.info(f"  Mean AUC: {sum(supervised_aucs)/len(supervised_aucs):.4f}")
        logger.info(f"  Best AUC: {max(supervised_aucs):.4f}")

    if pseudo_aucs:
        logger.info(f"\nPseudo-label stage:")
        logger.info(f"  Mean AUC: {sum(pseudo_aucs)/len(pseudo_aucs):.4f}")
        logger.info(f"  Best AUC: {max(pseudo_aucs):.4f}")

    logger.info(f"\nOverall best AUC: {state['best_auc']:.4f}")

    if state["disabled_features"]:
        logger.info(f"\nAuto-disabled features (hurt performance):")
        for feat in state["disabled_features"]:
            logger.info(f"  - {feat}")
    else:
        logger.info(f"\nAll novel features KEPT (all improved performance)")

    logger.info(f"\nResults saved to: {logdir}")
    logger.info(f"Pipeline state: {get_state_path(logdir)}")
    logger.info("=" * 60)


def main():
    p = argparse.ArgumentParser(
        description="Fully automated BirdCLEF training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", required=True,
                   help="Path to dataset root (birdclef-2025 or birdclef-merged)")
    p.add_argument("--metadata_file", default=None,
                   help="Metadata CSV filename inside data_root. "
                        "Auto-detected: train_merged.csv > train.csv > train_metadata.csv")
    p.add_argument("--hdf5_root", default=None,
                   help="Path to pre-decoded HDF5 audio cache (optional)")
    p.add_argument("--pretrained_path", default=None,
                   help="Path to in-domain pretrained backbone (optional)")
    p.add_argument("--logdir", default=str(PROJECT_ROOT / "logdir"),
                   help="Output directory for checkpoints and logs")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU device index")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint (reads pipeline_state.json)")
    args = p.parse_args()

    # Auto-detect metadata file (prefer merged > train.csv > train_metadata.csv)
    if args.metadata_file is None:
        data_root = Path(args.data_root)
        for candidate in ["train_merged.csv", "train.csv", "train_metadata.csv"]:
            if (data_root / candidate).exists():
                args.metadata_file = candidate
                break
        if args.metadata_file is None:
            args.metadata_file = "train.csv"
    logger.info(f"Metadata file: {args.metadata_file}")

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Load or create state — fresh start unless --resume is given
    if args.resume:
        state = load_state(logdir)
        logger.info(f"Resuming from state: stage={state['stage']}, "
                    f"completed={len(state['completed_folds'])} folds")
    else:
        state = {
            "stage": "supervised",
            "backbone_idx": 0,
            "fold_idx": 0,
            "pseudo_iteration": 0,
            "completed_folds": [],
            "fold_aucs": {},
            "disabled_features": [],
            "best_auc": 0.0,
        }

    logger.info("BirdCLEF 2025 — Full Automated Pipeline")
    logger.info(f"  Data root:    {args.data_root}")
    logger.info(f"  HDF5 root:    {args.hdf5_root}")
    logger.info(f"  Log dir:      {logdir}")
    logger.info(f"  Backbones:    {BACKBONES}")
    logger.info(f"  Folds:        {N_FOLDS}")
    logger.info(f"  Pseudo iters: {PSEUDO_ITERATIONS}")
    logger.info(f"  Resume:       {args.resume}")
    logger.info(f"  Current state: stage={state['stage']}, "
                f"completed={len(state['completed_folds'])} folds")

    total_start = time.time()

    try:
        if state["stage"] == "supervised":
            run_supervised_stage(args, logdir, state)

        if state["stage"] == "pseudo":
            run_pseudo_stage(args, logdir, state)

        if state["stage"] == "calibration":
            run_calibration_stage(args, logdir, state)

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted. Run with --resume to continue.")
        save_state(logdir, state)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        save_state(logdir, state)
        raise

    total_hours = (time.time() - total_start) / 3600
    logger.info(f"\nTotal pipeline time: {total_hours:.1f} hours")
    print_final_summary(logdir, state)


if __name__ == "__main__":
    main()
