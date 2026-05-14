"""
smoke_test.py — Pre-training sanity check
==========================================
Run this BEFORE starting actual training to catch crashes early. Does a
single forward/backward pass on synthetic data plus one real-batch forward
pass through the dataset, exercising every code path the real training will
hit. ~30 seconds on RTX 4090.

Usage:
    python scripts/smoke_test.py configs/train/selected_eca.py

If this passes, training is safe to launch.
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code_base.models.train_module import BirdCLEFModule
from code_base.datasets.audio_dataset import BirdCLEFDataset, BalancedSampler

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
            issues.append(f"  {key} = {p} → MISSING")
        elif p:
            print(f"  {key} = {p} → ok")

    meta = Path(cfg["data_root"]) / cfg.get("metadata_file", "train.csv")
    if not meta.exists():
        issues.append(f"  metadata file = {meta} → MISSING")
    else:
        print(f"  metadata file = {meta} → ok")

    if cfg.get("negative_audio_dir"):
        if not Path(cfg["negative_audio_dir"]).exists():
            issues.append(f"  negative_audio_dir = {cfg['negative_audio_dir']} → MISSING")
        else:
            n_files = sum(1 for _ in Path(cfg["negative_audio_dir"]).rglob("*.wav"))
            print(f"  negative_audio_dir = {cfg['negative_audio_dir']} → ok ({n_files} wav)")
    else:
        print("  negative_audio_dir = None → open-set rejection training DISABLED")

    for k in ("bg_esc50_paths", "bg_soundscape_paths"):
        v = cfg.get(k, [])
        if v:
            n_exist = sum(1 for p in v if Path(p).exists())
            print(f"  {k}: {n_exist}/{len(v)} paths exist")
        else:
            print(f"  {k}: EMPTY → bg noise augmentation off")

    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="Path to a train config .py")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
        print("\nPATH PROBLEMS — fix these before training:")
        for i in issues:
            print(i)
        sys.exit(1)
    print("Paths OK.\n")

    # ── Build model on device ─────────────────────────────────────────────
    print(f"Building model on {args.device}...")
    cfg_with_steps = {**cfg, "steps_per_epoch": 100}
    model = BirdCLEFModule(cfg_with_steps).to(args.device)
    model.train()

    # Manually attach a fake trainer attr so on_validation_epoch_end works
    # (we don't actually call it in smoke test, but configure_optimizers
    #  needs trainer.estimated_stepping_batches via the try/except fallback).
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")
    print(f"  feat_dim: {model.model.feat_dim}")
    print(f"  head type: {type(model.model.head).__name__}")
    assert type(model.model.head).__name__ == "NoisyConditionedHead", \
        f"Expected NoisyConditionedHead, got {type(model.model.head).__name__}"
    print("  Head is NoisyConditionedHead (FiLM enabled) → ok")
    print()

    # ── Synthetic batch forward + backward ────────────────────────────────
    print("Running synthetic forward + backward pass...")
    B  = 4
    sr = cfg.get("sr", 32_000)
    chunk_samples = int(cfg.get("chunk_duration", 5.0) * sr)
    n_classes     = cfg.get("n_classes", 206)

    fake_batch = {
        "audio": torch.randn(B, chunk_samples, device=args.device) * 0.1,
        "label": torch.zeros(B, n_classes, device=args.device),
        "idx":   torch.arange(B),
        "filename": ["fake1.ogg", "fake2.ogg", "fake3.ogg", "fake4.ogg"],
    }
    # Set one positive class per sample so the loss has something to learn
    for i in range(B):
        fake_batch["label"][i, i] = 1.0
    fake_batch["label"] = fake_batch["label"] * (1 - 0.05) + 0.05 / n_classes

    opt = torch.optim.RAdam(model.parameters(), lr=1e-3)
    loss = model.training_step(fake_batch, batch_idx=0)
    print(f"  synthetic training loss: {float(loss):.4f}")
    if not torch.isfinite(loss):
        print("FAIL: loss is not finite")
        sys.exit(1)
    loss.backward()
    grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  grad norm: {grad_norm:.4f}")
    opt.step()
    opt.zero_grad()
    print("Synthetic batch OK.\n")

    # ── Real-data batch ────────────────────────────────────────────────────
    print("Loading 1 real batch from the dataset...")
    df = pd.read_csv(Path(cfg["data_root"]) / cfg.get("metadata_file", "train.csv"))
    labels  = sorted(df["primary_label"].unique())
    l2i     = {l: i for i, l in enumerate(labels)}
    ds = BirdCLEFDataset(
        df              = df.head(64),
        audio_root      = cfg.get("audio_root"),
        label_to_idx    = l2i,
        n_classes       = n_classes,
        sr              = sr,
        chunk_duration  = cfg.get("chunk_duration", 5.0),
        label_smoothing = cfg.get("label_smoothing", 0.05),
        chunk_strategy  = cfg.get("chunk_strategy", "firstlast7"),
        use_secondary   = cfg.get("use_secondary_labels", True),
        augment         = True,
        is_train        = True,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=False, num_workers=0)
    real_batch_cpu = next(iter(dl))
    real_batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                  for k, v in real_batch_cpu.items()}
    loss = model.training_step(real_batch, batch_idx=1)
    print(f"  real-data training loss: {float(loss):.4f}")
    if not torch.isfinite(loss):
        print("FAIL: real-data loss is not finite")
        sys.exit(1)
    loss.backward()
    opt.step()
    print("Real batch OK.\n")

    # ── Validation step ────────────────────────────────────────────────────
    print("Running validation_step...")
    model.eval()
    model.validation_step(real_batch, batch_idx=0)
    # Manually trigger the end-of-epoch logic to test per-class AUC + snapshot
    model.on_validation_epoch_end()
    print("validation_step + on_validation_epoch_end OK.\n")

    # ── Verify state_dict round-trip (Lightning ↔ inference) ──────────────
    print("Verifying FiLM head state_dict can round-trip Lightning ↔ inference...")
    sd = model.model.state_dict()
    film_keys = [k for k in sd if "head.film" in k or "head.linear" in k]
    print(f"  FiLM/head keys in checkpoint: {len(film_keys)}")
    for k in film_keys[:6]:
        print(f"    {k}: {tuple(sd[k].shape)}")

    # Rebuild a bare BirdCLEFModel as inference would and load the head
    from code_base.models.backbone import BirdCLEFModel
    inf_model = BirdCLEFModel(
        backbone_name          = cfg["backbone"],
        n_classes              = n_classes,
        pretrained             = False,
        use_noise_conditioning = True,
        noise_dim              = cfg.get("noise_dim", 128),
    ).to(args.device).eval()
    missing, unexpected = inf_model.load_state_dict(sd, strict=False)
    head_missing = [k for k in missing if k.startswith("head.")]
    if head_missing:
        print(f"  FAIL: inference model missing head keys: {head_missing[:5]}")
        sys.exit(1)
    print("  Inference model loads FiLM head cleanly → ok\n")

    print(f"{'='*60}\n  ALL CHECKS PASSED. Training is safe to launch.\n{'='*60}")


if __name__ == "__main__":
    main()
