"""
pseudo_iter1_eca.py — Noisy Student Iteration 1 (ECA-NFNet, full pseudo)
=========================================================================
Stage 2: First pseudo-label training.
Teacher = Stage 1 ECA × 5 folds.
Pseudo-labels: max_prob≥0.5, class_prob≥0.1, power=1.0 (no transform yet)
               → 19,405 chunks from 4,430 files (Table 4)

1st place power transform: set power_alpha=0.6 in generate_pseudo_labels.py

Pipeline:
  1. Train selected_eca.py (Stage 1)
  2. python scripts/generate_pseudo_labels.py --iteration 1 --power_alpha 0.6 ...
  3. Set pseudo_labels_path below → train this config

SoftAUC enabled (auc_weight=0.3) since we're optimizing AUC metric.
"""

# Import base config and override
import importlib.util, sys
from pathlib import Path

_spec   = importlib.util.spec_from_file_location(
    "eca", Path(__file__).parent / "selected_eca.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
config  = dict(_module.config)

config.update({
    "exp_name": (
        "eca_nfnet_l0_pseudo_iter1_v3capped_"
        "SqrtBalancing_Radamlr5e4_CosBatchLR1e6_Epoch20_FocalBCELoss_LSF1005"
    ),
    "stage":                "pseudo",

    # ── Pseudo-label path (v3 ckpt → masked → capped @ 300/class, drop cls 45) ──
    "pseudo_labels_path":   "/workspace/pseudo_labels/pseudo_labels_iter1_v3capped.parquet",

    # ── labeled_fraction=0.5 caused stall (47%-trokin pseudo pollution dominated
    # the gradient and pulled the fresh head toward a collapsed predictor).
    # 0.85 keeps pseudo influence as a 15% domain-bridging signal instead.
    "labeled_fraction":     0.85,

    # ── SoftAUC DISABLED for iter-1. With a freshly re-initialised head, SoftAUC
    # produces noisy/near-zero gradients that drown out the BCE+Focal signal.
    # First iter-1 attempt with auc_weight=0.3 stalled at val/auc 0.50-0.51
    # for 5 epochs. v3 trained with auc_weight=0.0 and converged fine.
    # Re-enable in iter-2/3 once the head is well-trained.
    "auc_weight":           0.0,

    # ── Slightly reduced LR for fine-tuning from v3 backbone ──────────────
    "lr":                   5e-4,

    # ── Same epoch count as v3 (need full convergence from ImageNet init) ──
    "epochs":               30,

    # ── ImageNet-pretrained backbone, same as v3 Stage 1 (works) ──────────
    # Earlier attempt set pretrained=False + pretrained_path=<v3.ckpt> to
    # warm-start from v3. The _load_pretrained() key-rename logic is buggy
    # (chained .replace() calls leave a 'model.' prefix that doesn't match
    # the bare backbone state_dict) → load silently fails, model trains
    # with a fully random backbone and stalls at val/auc ~0.50 for 5+ epochs.
    # Reverting to ImageNet init (same as v3 Stage 1, which converged to
    # 0.7756 in 7 epochs). Pseudo data still drives the soundscape bridge.
    "pretrained":           True,
    "pretrained_path":      None,

    # ── BG mixing: disable soundscape BG (v4 collapse cause), keep ESC-50 ─
    # v4 trained with bg_soundscape_dir = train_soundscapes and collapsed:
    # those soundscapes contain unlabeled target birds, mixing them as "BG"
    # into focal recordings creates a labeling contradiction (model is
    # told "no bird" while target species are present in the BG track).
    # ESC-50 is safe — it's non-biological environmental sounds.
    "bg_soundscape_dir":    None,
    "bg_esc50_dir":         "/workspace/ESC-50-nobird",
})
