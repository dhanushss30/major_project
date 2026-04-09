"""
run_pipeline.py — Master Orchestration Script
==============================================
Runs the complete BirdCLEF 2025 pipeline from start to finish.
Combines the 1st & 2nd place solutions with 3 novel contributions.

Target: ~0.95 AUC (vs 1st place 0.930, 2nd place 0.928)

Novel Contributions (not in either winning solution):
  #1 — DANN Domain Adaptation (backbone learns domain-invariant features)
  #2 — Species Co-occurrence GCN (ecological graph prediction refinement)
  #3 — Per-class Calibration + Adaptive Pseudo-label Thresholds

Usage:
    # Run from birdclef_combined\ directory
    # First edit CONFIG section below with YOUR paths

    python scripts/run_pipeline.py --stage all          # Full pipeline
    python scripts/run_pipeline.py --stage supervised   # Only Stage 1
    python scripts/run_pipeline.py --stage pseudo1      # Only Iter 1
    python scripts/run_pipeline.py --stage novel        # Only calibration+GCN
    python scripts/run_pipeline.py --stage submit       # Only final inference

    # List all stages
    python scripts/run_pipeline.py --list
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ]
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG — EDIT THESE PATHS BEFORE RUNNING
# ════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Your data paths ───────────────────────────────────────────────────
    "DATA_ROOT"       : "D:/birdclef_data/birdclef-2025",
    "AUDIO_ROOT"      : "D:/birdclef_data/birdclef-2025/train_audio",
    "SOUNDSCAPE_ROOT" : "D:/birdclef_data/birdclef-2025/train_soundscapes",
    "HDF5_ROOT"       : "D:/birdclef_data/birdclef_hdf5",       # Optional speedup
    "PSEUDO_DIR"      : "D:/birdclef_data/pseudo_labels",
    "LOGDIR"          : "logdir",

    # ── Background noise (download from links below before training) ───────
    # ESC-50: https://github.com/karolpiczak/ESC-50/releases/download/v2.0.0/ESC-50-master.zip
    # Prior soundscapes: download from Kaggle BirdCLEF 2024 dataset
    "BG_ESC50_PATHS"       : [],   # e.g. ["D:/esc50/audio/1-100032-A-0.wav", ...]
    "BG_SOUNDSCAPE_PATHS"  : [],   # e.g. ["D:/birdclef2024/soundscapes/XC1.ogg", ...]

    # ── Backbone names (do not change unless adding new backbones) ────────
    "ECA_BACKBONE"  : "eca_nfnet_l0",
    "EBS_BACKBONE"  : "tf_efficientnetv2_s",
    "N_CLASSES"     : 206,

    # ── Python executable ─────────────────────────────────────────────────
    "PYTHON"        : sys.executable,
}

# ════════════════════════════════════════════════════════════════════════════
#  STAGE TRACKING  (saves progress so you can resume)
# ════════════════════════════════════════════════════════════════════════════

PROGRESS_FILE = Path("pipeline_progress.json")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(stage: str, status: str = "done"):
    p = load_progress()
    p[stage] = {"status": status, "time": datetime.now().isoformat()}
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)
    logger.info(f"Progress saved: {stage} → {status}")


def is_done(stage: str) -> bool:
    return load_progress().get(stage, {}).get("status") == "done"


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def run(cmd: list, desc: str = ""):
    """Run a subprocess command, stream output, raise on failure."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  {desc}")
    logger.info(f"  CMD: {' '.join(str(c) for c in cmd)}")
    logger.info(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = timedelta(seconds=int(time.time() - t0))
    logger.info(f"  Completed in {elapsed}")
    return result


def find_manifest(logdir: str, pattern: str) -> Path:
    """Find checkpoint_manifest.json matching a name pattern."""
    matches = list(Path(logdir).glob(f"**/{pattern}*/checkpoint_manifest.json"))
    if not matches:
        raise FileNotFoundError(
            f"No manifest found matching '{pattern}' in {logdir}.\n"
            f"Did the training stage complete successfully?"
        )
    # Return most recently modified
    return sorted(matches, key=lambda p: p.stat().st_mtime)[-1]


def update_config_paths():
    """
    Inject BG paths from CONFIG into the train configs.
    Writes updated values directly so you don't have to edit configs manually.
    """
    import importlib.util, ast

    for cfg_file in [
        "configs/train/selected_eca.py",
        "configs/train/selected_ebs.py",
    ]:
        path = Path(cfg_file)
        text = path.read_text()

        # Replace empty bg paths if CONFIG has values
        if CONFIG["BG_ESC50_PATHS"]:
            esc_repr = repr(CONFIG["BG_ESC50_PATHS"])
            text = text.replace('"bg_esc50_paths":      []', f'"bg_esc50_paths":      {esc_repr}')
            text = text.replace('"bg_esc50_paths":  []',    f'"bg_esc50_paths":  {esc_repr}')

        if CONFIG["BG_SOUNDSCAPE_PATHS"]:
            sc_repr = repr(CONFIG["BG_SOUNDSCAPE_PATHS"])
            text = text.replace('"bg_soundscape_paths": []', f'"bg_soundscape_paths": {sc_repr}')
            text = text.replace('"bg_soundscape_paths":  []', f'"bg_soundscape_paths":  {sc_repr}')

        path.write_text(text)

    logger.info("Config paths updated with background noise paths from CONFIG.")


def set_pretrained_path_in_config(config_file: str, ckpt_path: str):
    """Update pretrained_path in a pseudo config."""
    path  = Path(config_file)
    text  = path.read_text()
    old   = '"pretrained_path":    None'
    new   = f'"pretrained_path":    "{ckpt_path}"'
    if old not in text:
        old = '"pretrained_path":      None'
        new = f'"pretrained_path":      "{ckpt_path}"'
    path.write_text(text.replace(old, new, 1))
    logger.info(f"Set pretrained_path in {config_file}: {ckpt_path}")


def get_best_checkpoint(manifest_path: Path) -> str:
    """Get the best checkpoint path from a manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    # Return fold_0 best checkpoint (or first available)
    for key in ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]:
        if key in manifest and manifest[key]:
            return manifest[key]
    raise ValueError(f"No valid checkpoint in {manifest_path}")


# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGES
# ════════════════════════════════════════════════════════════════════════════

def stage_precompute_hdf5():
    """
    Optional but highly recommended: Convert audio to HDF5 for fast loading.
    Converts all train_audio/*.ogg files to HDF5 format.
    Only run if you have the HDF5_ROOT path set in CONFIG.
    Skippable — training will fall back to raw audio loading.
    """
    if not CONFIG["HDF5_ROOT"]:
        logger.info("HDF5_ROOT not set — skipping HDF5 precompute (slower training)")
        return

    if is_done("hdf5"):
        logger.info("HDF5 precompute already done — skipping")
        return

    run([
        CONFIG["PYTHON"], "scripts/precompute_features.py",
        "--src_dir",   CONFIG["AUDIO_ROOT"],
        "--dst_dir",   CONFIG["HDF5_ROOT"],
        "--n_cores",   "4",
    ], desc="HDF5 Precompute: Convert all audio to fast-access HDF5 format")

    save_progress("hdf5")


def stage_supervised_eca():
    """
    Stage 1A: Train ECA-NFNet-L0, all 5 folds, 50 epochs.
    Novel: DANN domain adaptation enabled in config.
    Time: ~2.5–3 days on RTX 3050 Ti.
    """
    if is_done("supervised_eca"):
        logger.info("supervised_eca already done — skipping")
        return

    update_config_paths()

    run([
        CONFIG["PYTHON"], "scripts/main_train.py",
        "configs/train/selected_eca.py",
    ], desc="Stage 1A: ECA-NFNet-L0 supervised training (5 folds, 50 epochs, DANN)")

    save_progress("supervised_eca")


def stage_supervised_ebs():
    """
    Stage 1B: Train EfficientNetV2-S, all 5 folds, 50 epochs.
    Novel: DANN domain adaptation enabled in config.
    Time: ~2.5–3 days on RTX 3050 Ti.
    """
    if is_done("supervised_ebs"):
        logger.info("supervised_ebs already done — skipping")
        return

    run([
        CONFIG["PYTHON"], "scripts/main_train.py",
        "configs/train/selected_ebs.py",
    ], desc="Stage 1B: EfficientNetV2-S supervised training (5 folds, 50 epochs, DANN)")

    save_progress("supervised_ebs")


def stage_novel_components():
    """
    Stage 2: Novel Contributions #2 and #3.
      - Per-class temperature calibration
      - Adaptive pseudo-label thresholds
      - Species co-occurrence GCN training
    Requires: supervised_eca done.
    Time: ~30 minutes.
    """
    if is_done("novel_components"):
        logger.info("novel_components already done — skipping")
        return

    eca_manifest = find_manifest(CONFIG["LOGDIR"], "eca_nfnet_l0_noamp")

    run([
        CONFIG["PYTHON"], "scripts/train_novel_components.py",
        "--manifest",        str(eca_manifest),
        "--data_root",       CONFIG["DATA_ROOT"],
        "--soundscape_root", CONFIG["SOUNDSCAPE_ROOT"],
        "--audio_root",      CONFIG["AUDIO_ROOT"],
        "--hdf5_root",       CONFIG["HDF5_ROOT"] or "",
        "--output_dir",      f"{CONFIG['LOGDIR']}/novel_components",
        "--backbone",        CONFIG["ECA_BACKBONE"],
        "--gcn_epochs",      "30",
        "--target_precision","0.80",
    ], desc="Stage 2: Novel Components — Calibration + Species GCN (Contribution #2 & #3)")

    save_progress("novel_components")


def stage_pseudo_iter1():
    """
    Stage 3: Pseudo-labeling Iteration 1.
    Teacher: Stage 1A ECA ensemble (5 folds).
    Power alpha=0.6 (mild compression of overconfidence).
    Expected: ~19,000 soundscape chunks selected.
    """
    if is_done("pseudo_iter1_generate"):
        logger.info("pseudo_iter1_generate already done — skipping")
    else:
        eca_manifest = find_manifest(CONFIG["LOGDIR"], "eca_nfnet_l0_noamp")

        run([
            CONFIG["PYTHON"], "scripts/generate_pseudo_labels.py",
            "--checkpoint_manifest", str(eca_manifest),
            "--soundscape_dir",      CONFIG["SOUNDSCAPE_ROOT"],
            "--output_dir",          CONFIG["PSEUDO_DIR"],
            "--iteration",           "1",
            "--power_alpha",         "0.6",
            "--backbone",            CONFIG["ECA_BACKBONE"],
            "--mode",                "full",
            "--num_workers",         "0",
        ], desc="Stage 3A: Generate Pseudo-Labels Iteration 1 (power_alpha=0.6, ~19K chunks)")

        save_progress("pseudo_iter1_generate")

    if is_done("pseudo_iter1_train"):
        logger.info("pseudo_iter1_train already done — skipping")
        return

    # Set pretrained_path to Stage 1 best checkpoint
    eca_manifest = find_manifest(CONFIG["LOGDIR"], "eca_nfnet_l0_noamp")
    best_ckpt    = get_best_checkpoint(eca_manifest)
    set_pretrained_path_in_config("configs/train/pseudo_iter1_eca.py", best_ckpt)

    # Update pseudo labels path
    pseudo_path = f"{CONFIG['PSEUDO_DIR']}/pseudo_labels_iter1_full.parquet"
    cfg_text    = Path("configs/train/pseudo_iter1_eca.py").read_text()
    cfg_text    = cfg_text.replace(
        "D:/birdclef_data/pseudo_labels/pseudo_labels_iter1_full.parquet",
        pseudo_path
    )
    Path("configs/train/pseudo_iter1_eca.py").write_text(cfg_text)

    run([
        CONFIG["PYTHON"], "scripts/main_train.py",
        "configs/train/pseudo_iter1_eca.py",
    ], desc="Stage 3B: Pseudo Training Iteration 1 — ECA student (SoftAUC enabled)")

    save_progress("pseudo_iter1_train")


def stage_pseudo_iter2():
    """
    Stage 4: Pseudo-labeling Iteration 2.
    Teacher: Stage 3 Iter1 ECA student.
    Power alpha=0.5. Mode: full + OOF merged.
    EfficientNetV2-S as student (backbone diversity).
    """
    if is_done("pseudo_iter2_generate"):
        logger.info("pseudo_iter2_generate already done — skipping")
    else:
        pseudo1_manifest = find_manifest(CONFIG["LOGDIR"], "PseudoF2PT05MT01P06I1")

        for mode in ["full", "oof"]:
            run([
                CONFIG["PYTHON"], "scripts/generate_pseudo_labels.py",
                "--checkpoint_manifest", str(pseudo1_manifest),
                "--soundscape_dir",      CONFIG["SOUNDSCAPE_ROOT"],
                "--output_dir",          CONFIG["PSEUDO_DIR"],
                "--iteration",           "2",
                "--power_alpha",         "0.5",
                "--backbone",            CONFIG["ECA_BACKBONE"],
                "--mode",                mode,
                "--num_workers",         "0",
            ], desc=f"Stage 4A: Generate Pseudo-Labels Iteration 2 ({mode} mode)")

        # Merge full + oof
        run([
            CONFIG["PYTHON"], "scripts/merge_pseudo_labels.py",
            "--inputs",
            f"{CONFIG['PSEUDO_DIR']}/pseudo_labels_iter2_full.parquet",
            f"{CONFIG['PSEUDO_DIR']}/pseudo_labels_iter2_oof.parquet",
            "--output",
            f"{CONFIG['PSEUDO_DIR']}/pseudo_labels_iter2_combined.parquet",
        ], desc="Stage 4B: Merge Iteration 2 full + OOF pseudo-labels")

        save_progress("pseudo_iter2_generate")

    if is_done("pseudo_iter2_train"):
        logger.info("pseudo_iter2_train already done — skipping")
        return

    # Set pretrained_path to Iter1 best checkpoint
    pseudo1_manifest = find_manifest(CONFIG["LOGDIR"], "PseudoF2PT05MT01P06I1")
    best_ckpt        = get_best_checkpoint(pseudo1_manifest)
    set_pretrained_path_in_config("configs/train/pseudo_iter2_ebs.py", best_ckpt)

    pseudo_path = f"{CONFIG['PSEUDO_DIR']}/pseudo_labels_iter2_combined.parquet"
    cfg_text    = Path("configs/train/pseudo_iter2_ebs.py").read_text()
    cfg_text    = cfg_text.replace(
        "D:/birdclef_data/pseudo_labels/pseudo_labels_iter2_combined.parquet",
        pseudo_path
    )
    Path("configs/train/pseudo_iter2_ebs.py").write_text(cfg_text)

    run([
        CONFIG["PYTHON"], "scripts/main_train.py",
        "configs/train/pseudo_iter2_ebs.py",
    ], desc="Stage 4C: Pseudo Training Iteration 2 — EfficientNetV2-S student")

    save_progress("pseudo_iter2_train")


def stage_pseudo_iter3():
    """
    Stage 5: Pseudo-labeling Iteration 3.
    Teacher: Stage 4 EBS student (best teacher yet).
    Power alpha=0.4 (most aggressive compression).
    Merge ALL 3 iterations for final training.
    """
    if is_done("pseudo_iter3_generate"):
        logger.info("pseudo_iter3_generate already done — skipping")
    else:
        pseudo2_manifest = find_manifest(CONFIG["LOGDIR"], "PseudoF2PT05MT01P05I2")

        run([
            CONFIG["PYTHON"], "scripts/generate_pseudo_labels.py",
            "--checkpoint_manifest", str(pseudo2_manifest),
            "--soundscape_dir",      CONFIG["SOUNDSCAPE_ROOT"],
            "--output_dir",          CONFIG["PSEUDO_DIR"],
            "--iteration",           "3",
            "--power_alpha",         "0.4",
            "--backbone",            CONFIG["EBS_BACKBONE"],
            "--mode",                "full",
            "--num_workers",         "0",
        ], desc="Stage 5A: Generate Pseudo-Labels Iteration 3 (power_alpha=0.4, ~4K chunks)")

        # Merge ALL 3 iterations
        run([
            CONFIG["PYTHON"], "scripts/merge_pseudo_labels.py",
            "--inputs",
            f"{CONFIG['PSEUDO_DIR']}/pseudo_labels_iter1_full.parquet",
            f"{CONFIG['PSEUDO_DIR']}/pseudo_labels_iter2_combined.parquet",
            f"{CONFIG['PSEUDO_DIR']}/pseudo_labels_iter3_full.parquet",
            "--output",
            f"{CONFIG['PSEUDO_DIR']}/pseudo_labels_all_iters_merged.parquet",
        ], desc="Stage 5B: Merge ALL 3 iterations (total ~27,000 pseudo-labeled chunks)")

        save_progress("pseudo_iter3_generate")

    if is_done("pseudo_iter3_train"):
        logger.info("pseudo_iter3_train already done — skipping")
        return

    # Set pretrained_path to Iter2 EBS best checkpoint
    pseudo2_manifest = find_manifest(CONFIG["LOGDIR"], "PseudoF2PT05MT01P05I2")
    best_ckpt        = get_best_checkpoint(pseudo2_manifest)
    set_pretrained_path_in_config("configs/train/pseudo_iter3_eca.py", best_ckpt)

    pseudo_path = f"{CONFIG['PSEUDO_DIR']}/pseudo_labels_all_iters_merged.parquet"
    cfg_text    = Path("configs/train/pseudo_iter3_eca.py").read_text()
    cfg_text    = cfg_text.replace(
        "D:/birdclef_data/pseudo_labels/pseudo_labels_all_iters_merged.parquet",
        pseudo_path
    )
    Path("configs/train/pseudo_iter3_eca.py").write_text(cfg_text)

    run([
        CONFIG["PYTHON"], "scripts/main_train.py",
        "configs/train/pseudo_iter3_eca.py",
    ], desc="Stage 5C: Pseudo Training Iteration 3 — Final ECA student (trained on 27K pseudo chunks)")

    save_progress("pseudo_iter3_train")


def stage_specialist():
    """
    Stage 6: Specialist model for insects, amphibians, mammals.
    These 51 rare species get their own dedicated model (+0.003 AUC).
    Time: ~1 day on RTX 3050 Ti.
    """
    if is_done("specialist"):
        logger.info("specialist already done — skipping")
        return

    run([
        CONFIG["PYTHON"], "scripts/main_train.py",
        "configs/train/insects_amphibians.py",
    ], desc="Stage 6: Specialist model — Insects + Amphibians + Mammals (51 species)")

    save_progress("specialist")


def stage_submit():
    """
    Stage 7: Final ensemble inference + GCN refinement → submission.csv
    Uses all trained models: 15-model ensemble + TTA + GCN + TopN.
    """
    if is_done("submit"):
        logger.info("submit already done — skipping")
        return

    run([
        CONFIG["PYTHON"], "scripts/main_inference.py",
        "configs/inference/ensemble_final.py",
        "--mode", "submit",
    ], desc="Stage 7: Final inference — 15-model ensemble + TTA + GCN + TopN → submission.csv")

    save_progress("submit")


# ════════════════════════════════════════════════════════════════════════════
#  STAGE REGISTRY
# ════════════════════════════════════════════════════════════════════════════

STAGES = {
    "hdf5"        : (stage_precompute_hdf5,  "Optional: Convert audio to HDF5 (faster loading)"),
    "supervised"  : (None,                   "Run supervised_eca + supervised_ebs"),
    "supervised_eca" : (stage_supervised_eca,"Stage 1A: ECA-NFNet-L0 (5 folds, DANN)"),
    "supervised_ebs" : (stage_supervised_ebs,"Stage 1B: EfficientNetV2-S (5 folds, DANN)"),
    "novel"       : (stage_novel_components, "Stage 2: Calibration + GCN (Novel #2 & #3)"),
    "pseudo1"     : (stage_pseudo_iter1,     "Stage 3: Pseudo iteration 1 (power=0.6)"),
    "pseudo2"     : (stage_pseudo_iter2,     "Stage 4: Pseudo iteration 2 (power=0.5)"),
    "pseudo3"     : (stage_pseudo_iter3,     "Stage 5: Pseudo iteration 3 (power=0.4)"),
    "specialist"  : (stage_specialist,       "Stage 6: Specialist insects/amphibians model"),
    "submit"      : (stage_submit,           "Stage 7: Final ensemble → submission.csv"),
    "all"         : (None,                   "Run complete pipeline end-to-end"),
}


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BirdCLEF 2025 Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --stage all          # Full pipeline
  python scripts/run_pipeline.py --stage supervised   # Stages 1A+1B only
  python scripts/run_pipeline.py --stage pseudo1      # Pseudo iter 1 only
  python scripts/run_pipeline.py --stage submit       # Final inference only
  python scripts/run_pipeline.py --list               # Show all stages
  python scripts/run_pipeline.py --status             # Show progress
        """
    )
    parser.add_argument("--stage",  default="all", choices=list(STAGES.keys()))
    parser.add_argument("--list",   action="store_true", help="List all stages")
    parser.add_argument("--status", action="store_true", help="Show pipeline progress")
    parser.add_argument("--reset",  type=str, default=None,
                        help="Reset a stage (mark as not done) so it reruns")
    args = parser.parse_args()

    if args.list:
        print("\n  BirdCLEF 2025 Pipeline Stages:")
        print("  " + "─"*55)
        for name, (fn, desc) in STAGES.items():
            status = "✓" if is_done(name) else " "
            print(f"  [{status}] {name:<18} {desc}")
        print()
        return

    if args.status:
        progress = load_progress()
        print("\n  Pipeline Progress:")
        print("  " + "─"*55)
        for name, (fn, desc) in STAGES.items():
            if fn is None:
                continue
            info = progress.get(name, {})
            status = info.get("status", "pending")
            time_  = info.get("time",   "")
            icon   = "✓" if status == "done" else "○"
            print(f"  {icon} {name:<20} {status:<8}  {time_[:16]}")
        print()
        return

    if args.reset:
        p = load_progress()
        if args.reset in p:
            del p[args.reset]
            with open(PROGRESS_FILE, "w") as f:
                json.dump(p, f, indent=2)
            print(f"Reset: {args.reset}")
        return

    # Change to project root
    os.chdir(Path(__file__).parent.parent)

    t_start = time.time()
    logger.info(f"\n{'═'*60}")
    logger.info(f"  BirdCLEF 2025 Pipeline  |  Stage: {args.stage}")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"{'═'*60}\n")

    if args.stage == "all":
        stage_precompute_hdf5()
        stage_supervised_eca()
        stage_supervised_ebs()
        stage_novel_components()
        stage_pseudo_iter1()
        stage_pseudo_iter2()
        stage_pseudo_iter3()
        stage_specialist()
        stage_submit()

    elif args.stage == "supervised":
        stage_supervised_eca()
        stage_supervised_ebs()

    elif args.stage in STAGES and STAGES[args.stage][0] is not None:
        STAGES[args.stage][0]()

    else:
        logger.error(f"Unknown stage: {args.stage}")
        sys.exit(1)

    elapsed = timedelta(seconds=int(time.time() - t_start))
    logger.info(f"\n{'═'*60}")
    logger.info(f"  Stage '{args.stage}' complete  |  Elapsed: {elapsed}")
    logger.info(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
