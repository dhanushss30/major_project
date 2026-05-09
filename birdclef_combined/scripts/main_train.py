"""
main_train.py — K-Fold Training Script
========================================
Exact CV strategy from paper §5.1:
  "5-fold cross-validation, stratified by primary species and grouped by author"
  "Undersampled species (≤1-2 samples) placed in validation fold at minimum 1 each"

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/main_train.py configs/train/selected_eca.py
  CUDA_VISIBLE_DEVICES=0 python scripts/main_train.py configs/train/selected_ebs.py
  CUDA_VISIBLE_DEVICES=0 python scripts/main_train.py configs/train/pseudo_iter1_eca.py
"""

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

try:
    import lightning as L
    from lightning.pytorch.callbacks import (
        ModelCheckpoint, LearningRateMonitor, RichProgressBar,
        StochasticWeightAveraging,
    )
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
except ImportError:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    try:
        from pytorch_lightning.callbacks import StochasticWeightAveraging
    except ImportError:
        StochasticWeightAveraging = None

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code_base.models.train_module import BirdCLEFModule
from code_base.datasets.audio_dataset import (
    BirdCLEFDataset, PseudoLabelDataset, BalancedSampler
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Config loader ────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load config from a .py file (defines `config` dict) or a .json file."""
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    spec   = importlib.util.spec_from_file_location("cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config"):
        raise AttributeError(f"{path} must define a `config` dict")
    return module.config


# ─── Cross-validation splits ─────────────────────────────────────────────

def build_cv_splits(
    df:       pd.DataFrame,
    n_folds:  int = 5,
    strategy: str = "stratified_group",   # paper §5.1
) -> pd.DataFrame:
    """
    Build CV splits using paper's strategy:
    "stratified by primary species and grouped by author"

    Returns df with a "fold" column (0..n_folds-1).
    """
    df = df.copy().reset_index(drop=True)

    if "fold" in df.columns:
        logger.info("Using existing 'fold' column in metadata")
        return df

    # Group by author (recordist) to prevent leakage
    author_col = next(
        (c for c in ["author", "recordist", "recordist_id"] if c in df.columns),
        None
    )
    if author_col is None:
        logger.warning("No author column found; using ungrouped StratifiedKFold")
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        df["fold"] = -1
        for fold_idx, (_, val_idx) in enumerate(
            skf.split(df, df["primary_label"])
        ):
            df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx
        return df

    # StratifiedGroupKFold: stratified by species, grouped by author
    labels  = df["primary_label"].values
    groups  = df[author_col].values
    df["fold"] = -1

    # Handle undersampled species: ensure ≥1 per fold in validation
    label_counts = df["primary_label"].value_counts()
    undersampled  = label_counts[label_counts < n_folds].index.tolist()

    # Separate undersampled
    mask_under = df["primary_label"].isin(undersampled)
    df_common  = df[~mask_under]
    df_under   = df[mask_under]

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(df_common, df_common["primary_label"], df_common[author_col])
    ):
        global_idx = df_common.iloc[val_idx].index
        df.loc[global_idx, "fold"] = fold_idx

    # Distribute undersampled: one per fold in rotation
    if len(df_under) > 0:
        for i, global_idx in enumerate(df_under.index):
            df.loc[global_idx, "fold"] = i % n_folds

    assert (df["fold"] == -1).sum() == 0, "Some samples have no fold assigned"
    logger.info(f"CV splits built: {n_folds} folds, "
                f"{len(undersampled)} undersampled species distributed")
    return df


# ─── Dataset builders ────────────────────────────────────────────────────

def build_datasets(cfg: dict, df: pd.DataFrame, fold: int, label_to_idx: dict):
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df   = df[df["fold"] == fold].reset_index(drop=True)

    logger.info(f"Fold {fold}: {len(train_df):,} train, {len(val_df):,} val")

    common_kwargs = dict(
        hdf5_root       = cfg.get("hdf5_root"),
        audio_root      = cfg.get("audio_root"),
        label_to_idx    = label_to_idx,
        n_classes       = cfg.get("n_classes", 206),
        sr              = cfg.get("sr", 32_000),
        chunk_duration  = cfg.get("chunk_duration", 5.0),
        label_smoothing = cfg.get("label_smoothing", 0.05),
        use_secondary   = cfg.get("use_secondary_labels", True),
    )

    train_labeled_ds = BirdCLEFDataset(
        df             = train_df,
        chunk_strategy = cfg.get("chunk_strategy", "firstlast7"),
        augment        = True,
        is_train       = True,
        **common_kwargs,
    )

    # Override label_smoothing=0.0 for validation (no smoothing on val labels)
    val_kwargs = {**common_kwargs, "label_smoothing": 0.0}
    val_ds = BirdCLEFDataset(
        df             = val_df,
        chunk_strategy = "random",
        augment        = False,
        is_train       = False,
        **val_kwargs,
    )

    # Add pseudo-labeled data if configured
    pseudo_path = cfg.get("pseudo_labels_path")
    if pseudo_path and Path(pseudo_path).exists():
        pseudo_df = pd.read_parquet(pseudo_path)
        logger.info(f"Loaded {len(pseudo_df):,} pseudo-labeled chunks from {pseudo_path}")

        train_ds = PseudoLabelDataset(
            labeled_dataset  = train_labeled_ds,
            pseudo_df        = pseudo_df,
            soundscape_root  = cfg["soundscape_root"],
            labeled_frac     = cfg.get("labeled_fraction", 0.5),
            sr               = cfg.get("sr", 32_000),
            chunk_duration   = cfg.get("chunk_duration", 5.0),
            label_smoothing  = cfg.get("label_smoothing", 0.05),
        )
    else:
        train_ds = train_labeled_ds

    return train_ds, val_ds


def build_dataloader(ds, cfg: dict, is_train: bool):
    if is_train:
        gamma = cfg.get("sampler_gamma", -0.5)

        if hasattr(ds, "labeled_ds"):
            # PseudoLabelDataset: build balanced weights covering [0, len(ds)).
            # Indices [0, n_labeled) → labeled data; [n_labeled, total) → pseudo.
            sampler_df = ds.labeled_ds.df
            labeled_sampler = BalancedSampler(
                df               = sampler_df,
                gamma            = gamma,
                n_samples        = len(sampler_df),
                replacement      = False,
                oversampling_map = cfg.get("oversampling_map", None),
                min_count        = cfg.get("min_oversample_count", 0),
            )
            src_w = labeled_sampler.weights
            n_src = len(src_w)

            # Resize labeled weights to exactly ds.n_labeled (cycle if needed)
            if ds.n_labeled <= n_src:
                labeled_w = src_w[:ds.n_labeled]
            else:
                repeats = (ds.n_labeled + n_src - 1) // n_src
                labeled_w = src_w.repeat(repeats)[:ds.n_labeled]

            n_pseudo = len(ds) - ds.n_labeled
            if n_pseudo > 0:
                pseudo_w = torch.full((n_pseudo,), labeled_w.mean().item(), dtype=torch.float64)
                all_w = torch.cat([labeled_w, pseudo_w])
            else:
                all_w = labeled_w
            all_w = all_w / all_w.sum()
            n_samples = cfg.get("steps_per_epoch", len(ds)) * cfg.get("batch_size", 64)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights     = all_w,
                num_samples = n_samples,
                replacement = True,
            )
        else:
            sampler = BalancedSampler(
                df               = ds.df,
                gamma            = gamma,
                n_samples        = cfg.get("steps_per_epoch", len(ds)) * cfg.get("batch_size", 64),
                replacement      = True,
                oversampling_map = cfg.get("oversampling_map", None),
                min_count        = cfg.get("min_oversample_count", 0),
                use_rating_weight = cfg.get("use_rating_weight", False),
            )
        shuffle = False
    else:
        sampler = None
        shuffle = False

    return torch.utils.data.DataLoader(
        ds,
        batch_size   = cfg.get("batch_size", 64),
        sampler      = sampler,
        shuffle      = shuffle,
        num_workers  = cfg.get("num_workers", 8),
        pin_memory   = True,
        drop_last    = is_train,
        persistent_workers = cfg.get("num_workers", 8) > 0,
    )


# ─── Single fold training ────────────────────────────────────────────────

def train_fold(cfg: dict, df: pd.DataFrame, label_to_idx: dict, fold: int, logdir: Path):
    logger.info(f"\n{'='*60}\n  Fold {fold} | {cfg['exp_name']}\n{'='*60}")
    L.seed_everything(cfg.get("seed", 42) + fold)

    train_ds, val_ds = build_datasets(cfg, df, fold, label_to_idx)
    train_dl = build_dataloader(train_ds, cfg, is_train=True)
    val_dl   = build_dataloader(val_ds,   cfg, is_train=False)

    # Compute steps_per_epoch for scheduler
    steps_per_epoch = len(train_dl)
    cfg_with_steps  = {**cfg, "steps_per_epoch": steps_per_epoch}

    model = BirdCLEFModule(cfg_with_steps)

    # Initialize taxonomy mapper (Novel #7) if enabled
    if cfg.get("use_taxonomy", False):
        species_to_taxon = cfg.get("species_to_taxon", {})
        if not species_to_taxon:
            taxon_json = cfg.get("species_to_taxon_path",
                                 str(PROJECT_ROOT / "configs" / "species_to_taxon.json"))
            if Path(taxon_json).exists():
                with open(taxon_json) as f:
                    species_to_taxon = json.load(f)
                logger.info(f"Loaded taxonomy mapping: {len(species_to_taxon)} species from {taxon_json}")
            else:
                logger.warning(
                    f"Taxonomy mapping empty and {taxon_json} not found. "
                    "Run: python scripts/build_taxonomy_map.py --data_root <path> first. "
                    "All species will default to Aves (taxon 0)."
                )
        model.set_taxonomy_mapper(label_to_idx, species_to_taxon)

    # Initialize prototypical head rarity weights (Novel #4)
    if cfg.get("use_prototypical", False):
        train_df = df[df["fold"] != fold]
        class_counts = torch.zeros(cfg.get("n_classes", 206))
        for species, idx in label_to_idx.items():
            count = (train_df["primary_label"] == species).sum()
            class_counts[idx] = count
        model.set_class_counts(class_counts)
        logger.info(f"Prototypical rarity weights set: "
                    f"min_count={int(class_counts.min())}, "
                    f"max_count={int(class_counts.max())}")

    fold_dir = logdir / f"fold_{fold}"

    callbacks = [
        ModelCheckpoint(
            dirpath   = str(fold_dir / "checkpoints"),
            filename  = "best-epoch{epoch:02d}-auc{val/auc:.4f}",
            monitor   = "val/auc",
            mode      = "max",
            save_top_k = 3,
            save_last  = True,
            verbose    = True,
            auto_insert_metric_name = False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # SWA: average top checkpoints for better generalization (2nd place trick)
    if cfg.get("use_swa", False) and StochasticWeightAveraging is not None:
        swa_lr = cfg.get("swa_lr", cfg.get("min_lr", 1e-6))
        swa_start_epoch = int(cfg.get("epochs", 50) * cfg.get("swa_start_frac", 0.75))
        callbacks.append(StochasticWeightAveraging(
            swa_lrs        = swa_lr,
            swa_epoch_start = swa_start_epoch,
        ))
        logger.info(f"SWA: enabled from epoch {swa_start_epoch} with lr={swa_lr}")

    logger_cls = WandbLogger if cfg.get("use_wandb") else TensorBoardLogger
    pl_logger  = logger_cls(save_dir=str(fold_dir), name="logs")

    trainer = L.Trainer(
        max_epochs              = cfg.get("epochs", 50),
        accelerator             = "gpu" if torch.cuda.is_available() else "cpu",
        devices                 = 1,
        precision               = cfg.get("precision", "32-true"),
        gradient_clip_val       = cfg.get("grad_clip", 5.0),
        accumulate_grad_batches = cfg.get("accumulate_grad", 1),
        limit_train_batches     = cfg.get("limit_train_batches", 1.0),
        callbacks               = callbacks,
        logger                  = pl_logger,
        log_every_n_steps       = 50,
        deterministic           = False,
        enable_progress_bar     = True,
    )

    trainer.fit(model, train_dl, val_dl)
    best_ckpt = callbacks[0].best_model_path
    best_auc  = float(callbacks[0].best_model_score or 0.0)
    logger.info(f"Fold {fold} best checkpoint: {best_ckpt}")
    logger.info(f"Fold {fold} best val AUC:    {best_auc:.4f}")

    # Write results.json so the automated pipeline can read the AUC
    results = {
        "fold":         fold,
        "best_val_auc": best_auc,
        "best_ckpt":    str(best_ckpt),
        "exp_name":     cfg.get("exp_name", ""),
    }
    with open(fold_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_ckpt


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?",
                        help="Path to config .py or .json file")
    parser.add_argument("--config_json", type=str, default=None,
                        help="Alternate: path to config .json (for pipeline automation)")
    parser.add_argument("--fold",   type=int, default=None, help="Train single fold")
    parser.add_argument("--logdir", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config_json or args.config
    if config_path is None:
        parser.error("Provide a config file (positional) or --config_json")

    cfg = load_config(config_path)

    # Load and prepare metadata
    data_root = Path(cfg["data_root"])
    meta_path = data_root / cfg.get("metadata_file", "train_metadata.csv")
    df = pd.read_csv(meta_path)
    logger.info(f"Loaded metadata: {len(df):,} samples, "
                f"{df['primary_label'].nunique()} species")

    # Build label mapping
    all_labels   = sorted(df["primary_label"].unique())
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    logger.info(f"Classes: {len(label_to_idx)}")

    # Build CV folds
    df = build_cv_splits(df, n_folds=cfg.get("n_folds", 5))

    # Save updated metadata with fold column
    logdir = Path(args.logdir or cfg.get("logdir", "logdir")) / cfg["exp_name"]
    logdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(logdir / "metadata_with_folds.csv", index=False)

    # Save config + label map
    with open(logdir / "config.json", "w") as f:
        json.dump({k: str(v) if not isinstance(v, (int,float,bool,list,dict,type(None)))
                   else v for k, v in cfg.items()}, f, indent=2)
    with open(logdir / "label_to_idx.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)

    # Train folds
    n_folds = cfg.get("n_folds", 5)
    folds   = [args.fold] if args.fold is not None else list(range(n_folds))

    best_ckpts = {}
    for fold in folds:
        ckpt = train_fold(cfg, df, label_to_idx, fold, logdir)
        best_ckpts[f"fold_{fold}"] = ckpt

    with open(logdir / "checkpoint_manifest.json", "w") as f:
        json.dump(best_ckpts, f, indent=2)

    logger.info(f"\nTraining complete.\n{json.dumps(best_ckpts, indent=2)}")


if __name__ == "__main__":
    main()
