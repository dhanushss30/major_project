"""inspect_higher_auc_ckpt.py — Examine the 0.7841 / 0.7840 checkpoints.

Reports:
  - state_dict top-level keys (to identify Lightning vs raw torch)
  - module name prefixes (to identify which novel modules are present)
  - tensor count + total param count
  - per-module shapes for the head and any novel modules

Use this to determine if a checkpoint is loadable as-is, needs config flag
flips, or is fundamentally incompatible with the current model code.

Usage:
    python scripts/inspect_higher_auc_ckpt.py \
        --ckpt /workspace/major_project/birdclef_combined/logdir/_v3_*/fold_0/checkpoints/best-epoch06-auc0.7841.ckpt
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to the .ckpt file (may contain shell glob - resolve before passing)")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] checkpoint not found: {ckpt_path}")
        sys.exit(1)
    print(f"=== inspecting: {ckpt_path.name} ===")
    print(f"file size: {ckpt_path.stat().st_size / (1024**2):.1f} MB")
    print()

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ─── top-level keys ───────────────────────────────────────────────────
    print(f"top-level keys: {list(state.keys()) if isinstance(state, dict) else type(state)}")
    print()

    if "state_dict" in state:
        sd = state["state_dict"]
        print(f"hyper_parameters keys: "
              f"{list(state.get('hyper_parameters', {}).keys())[:20]}")
        print(f"epoch: {state.get('epoch')}")
        print(f"global_step: {state.get('global_step')}")
        print()
    else:
        sd = state

    # ─── prefix histogram ─────────────────────────────────────────────────
    prefixes = Counter()
    for k in sd.keys():
        prefix = ".".join(k.split(".")[:3])   # first 3 components
        prefixes[prefix] += 1

    print("=== top-level module prefixes (count) ===")
    for prefix, count in prefixes.most_common(30):
        print(f"  {prefix:<50}  {count}")
    print()

    # ─── novel-module detection ───────────────────────────────────────────
    print("=== novel-module fingerprint ===")
    novel_markers = {
        "FiLM (noise conditioning)":  ["film", "gamma_net", "noise_proj"],
        "DANN (domain discrim)":      ["domain_discriminator", "grl"],
        "Prototypical head":          ["prototype", "proto_"],
        "Causal disentangle":         ["causal", "spurious", "hsic"],
        "Taxonomy aux":               ["taxonomy", "taxon_"],
        "Multi-res mel":              ["mel_lo", "mel_hi", "multi_res"],
        "TCR":                        ["tcr_"],
        "EMA model":                  ["ema_"],
        "Standard head":              ["head."],
        "Standard backbone":          ["backbone."],
    }
    for label, markers in novel_markers.items():
        hits = [k for k in sd.keys() if any(m in k.lower() for m in markers)]
        print(f"  {label:<28}  {'PRESENT  ' + str(len(hits)) + ' tensors' if hits else 'absent'}")
        if hits and len(hits) <= 5:
            for h in hits:
                print(f"        {h}  {tuple(sd[h].shape)}")
    print()

    # ─── total param count ────────────────────────────────────────────────
    total_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"total state_dict params (incl EMA if any): {total_params/1e6:.2f}M")
    print()

    # ─── head shapes ──────────────────────────────────────────────────────
    print("=== head shapes (classifier output) ===")
    for k, v in sd.items():
        if "head" in k.lower() and "weight" in k:
            print(f"  {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    sys.exit(main())
