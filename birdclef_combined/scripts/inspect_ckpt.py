"""inspect_ckpt.py — Diagnose a Lightning checkpoint.

Prints state_dict structure (main vs EMA), FiLM bias values, and the
weight drift between the main and EMA models. Used to debug the
"pseudo-labels are near-constant" problem: typically EMA model is what
val/auc measured, but generate_pseudo_labels.py loads the main model.

Usage:
    python scripts/inspect_ckpt.py <path-to-ckpt>
"""

import argparse
import sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    main_keys = [k for k in sd if k.startswith("model.")]
    ema_keys  = [k for k in sd if k.startswith("_ema_model.")]
    other     = [k for k in sd if not k.startswith(("model.", "_ema_model."))]

    print(f"main model keys:  {len(main_keys)}")
    print(f"ema  model keys:  {len(ema_keys)}")
    print(f"other keys:       {len(other)}  (e.g. {other[:3]})")

    for prefix, label in [("model.", "MAIN"), ("_ema_model.", "EMA ")]:
        gb = sd.get(f"{prefix}head.film.gamma_net.bias")
        bb = sd.get(f"{prefix}head.film.beta_net.bias")
        gw = sd.get(f"{prefix}head.film.gamma_net.weight")
        if gb is None:
            print(f"\n{label} FiLM: not present in checkpoint")
            continue
        print(f"\n{label} FiLM stats:")
        print(f"  gamma_net.bias    mean={float(gb.mean()):+.4f}  "
              f"std={float(gb.std()):.4f}  abs_max={float(gb.abs().max()):.4f}")
        print(f"  beta_net.bias     mean={float(bb.mean()):+.4f}  "
              f"std={float(bb.std()):.4f}  abs_max={float(bb.abs().max()):.4f}")
        print(f"  gamma_net.weight  abs_mean={float(gw.abs().mean()):.6f}  "
              f"abs_max={float(gw.abs().max()):.4f}")

    for layer in ["head.linear1", "head.linear2", "head.film.gamma_net",
                  "head.film.beta_net"]:
        mw = sd.get(f"model.{layer}.weight")
        ew = sd.get(f"_ema_model.{layer}.weight")
        if mw is not None and ew is not None:
            diff = float((mw - ew).abs().mean())
            rel  = float((mw - ew).abs().mean() / (ew.abs().mean() + 1e-9))
            print(f"\n{layer}.weight  |main-ema| mean = {diff:.6f}  "
                  f"(rel to ema = {rel:.4%})")


if __name__ == "__main__":
    sys.exit(main())
