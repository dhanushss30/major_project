"""
smoke_test.py — Pre-training sanity check
==========================================
Run this BEFORE starting actual training to catch crashes early. Drives the
full Lightning pipeline through ``Trainer(fast_dev_run=True)`` — one train
batch + one val batch with every hook fired — then verifies the FiLM head
state_dict can round-trip Lightning → inference model. ~60s on RTX 4090.

Usage:
    python scripts/smoke_test.py configs/train/selected_eca.py

Exits 0 if everything passes; non-zero otherwise. If this passes, training
is safe to launch.
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

from code_base.models.train_module import BirdCLEFModule
from code_base.datasets.audio_dataset import BirdCLEFDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_cfg(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("cfg", path)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.config


def check_paths(cfg: dict) -> list:
    """Return list of missing-path issues."""
    issues = []
    for key in ("data_root", "audio_root", "soundscape_root"):
        p = cfg.get(key)
        if p and not Path(p).exists():
            issues.append(f"  {key} = {p} -> MISSING")
        elif p:
            print(f"  {key} = {p} -> ok")

    meta = Path(cfg["data_root"]) / cfg.get("metadata_file", "train.csv")
    if not meta.exists():
        issues.append(f"  metadata file = {meta} -> MISSING")
    else:
        print(f"  metadata file = {meta} -> ok")

    if cfg.get("negative_audio_dir"):
        if not Path(cfg["negative_audio_dir"]).exists():
            issues.append(f"  negative_audio_dir = {cfg['negative_audio_dir']} -> MISSING")
        else:
            n_files = sum(1 for _ in Path(cfg["negative_audio_dir"]).rglob("*.wav"))
            print(f"  negative_audio_dir = {cfg['negative_audio_dir']} -> ok ({n_files} wav)")
    else:
        print("  negative_audio_dir = None -> open-set rejection training DISABLED")

    for k in ("bg_esc50_paths", "bg_soundscape_paths"):
        v = cfg.get(k, [])
        if v:
            n_exist = sum(1 for p in v if Path(p).exists())
            print(f"  {k}: {n_exist}/{len(v)} paths exist")
        else:
            print(f"  {k}: EMPTY -> bg noise augmentation off")

    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="Path to a train config .py")
    args = p.parse_args()

    print(f"\n{'='*60}\n  Smoke test: {args.config}\n{'='*60}\n")
    cfg = load_cfg(args.config)
    print(f"Backbone: {cfg['backbone']}, n_classes: {cfg.get('n_classes', 206)}")
    print(f"FiLM enabled: {cfg.get('use_noise_conditioning')}")
    print(f"Aux modules: dann={cfg.get('use_dann')} causal={cfg.get('use_causal')} "
          f"tcr={cfg.get('use_tcr')} tax={cfg.get('use_taxonomy')} "
          f"proto={cfg.get('use_prototypical')}")
    print(f"p_negative: {cfg.get('p_negative')}, neg_dir: {cfg.get('negative_audio_dir')}")
    print()

    print("Checking data paths...")
    issues = check_paths(cfg)
    if issues:
        print("\nPATH PROBLEMS - fix these before training:")
        for i in issues:
            print(i)
        sys.exit(1)
    print("Paths OK.\n")

    n_classes = cfg.get("n_classes", 206)
    sr        = cfg.get("sr", 32_000)

    # ── Build small train/val DataLoaders from real audio ────────────────
    print("Loading 32 rows from train.csv to build tiny train/val loaders...")
    df = pd.read_csv(Path(cfg["data_root"]) / cfg.get("metadata_file", "train.csv"))
    labels = sorted(df["primary_label"].unique())
    l2i    = {l: i for i, l in enumerate(labels)}

    head_df = df.head(32).reset_index(drop=True)
    train_df = head_df.iloc[:24].reset_index(drop=True)
    val_df   = head_df.iloc[24:].reset_index(drop=True)

    def _build(d, is_train):
        return BirdCLEFDataset(
            df              = d,
            audio_root      = cfg.get("audio_root"),
            label_to_idx    = l2i,
            n_classes       = n_classes,
            sr              = sr,
            chunk_duration  = cfg.get("chunk_duration", 5.0),
            label_smoothing = cfg.get("label_smoothing", 0.05),
            chunk_strategy  = cfg.get("chunk_strategy", "firstlast7"),
            use_secondary   = cfg.get("use_secondary_labels", True),
            augment         = True,
            is_train        = is_train,
        )

    train_ds, val_ds = _build(train_df, True), _build(val_df, False)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)
    print(f"  train_ds={len(train_ds)}, val_ds={len(val_ds)}\n")

    # ── Build model ────────────────────────────────────────────────────
    print("Building BirdCLEFModule...")
    cfg_with_steps = {**cfg, "steps_per_epoch": 6}
    model = BirdCLEFModule(cfg_with_steps)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")
    print(f"  feat_dim: {model.model.feat_dim}")
    print(f"  head type: {type(model.model.head).__name__}")
    assert type(model.model.head).__name__ == "NoisyConditionedHead", \
        f"Expected NoisyConditionedHead, got {type(model.model.head).__name__}"
    print("  Head is NoisyConditionedHead (FiLM enabled) -> ok\n")

    # ── fast_dev_run: 1 train batch + 1 val batch, every hook fires ───
    print("Running Trainer(fast_dev_run=True) ...")
    trainer = L.Trainer(
        fast_dev_run        = True,
        accelerator         = "gpu" if torch.cuda.is_available() else "cpu",
        devices             = 1,
        precision           = cfg.get("precision", "bf16-mixed"),
        enable_progress_bar = False,
        enable_checkpointing= False,
        logger              = False,
    )
    trainer.fit(model, train_dl, val_dl)
    print("Trainer.fit(fast_dev_run=True) completed.\n")

    # ── State_dict round-trip (Lightning <-> inference) ───────────────
    print("Verifying FiLM head state_dict can round-trip Lightning <-> inference...")
    sd = model.model.state_dict()
    film_keys = [k for k in sd if "head.film" in k or "head.linear" in k]
    print(f"  FiLM/head keys in checkpoint: {len(film_keys)}")
    for k in film_keys[:6]:
        print(f"    {k}: {tuple(sd[k].shape)}")

    from code_base.models.backbone import BirdCLEFModel
    inf_model = BirdCLEFModel(
        backbone_name          = cfg["backbone"],
        n_classes              = n_classes,
        pretrained             = False,
        use_noise_conditioning = True,
        noise_dim              = cfg.get("noise_dim", 128),
    ).eval()
    missing, unexpected = inf_model.load_state_dict(sd, strict=False)
    head_missing = [k for k in missing if k.startswith("head.")]
    if head_missing:
        print(f"  FAIL: inference model missing head keys: {head_missing[:5]}")
        sys.exit(1)
    print("  Inference model loads FiLM head cleanly -> ok\n")

    print(f"{'='*60}\n  ALL CHECKS PASSED. Training is safe to launch.\n{'='*60}")


if __name__ == "__main__":
    main()
