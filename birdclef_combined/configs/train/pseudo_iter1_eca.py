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
        "eca_nfnet_l0_noamp_64bs_SqrtBalancing_"
        "Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_"
        "PseudoF2PT05MT01P06I1_Full"
    ),
    "stage":                "pseudo",

    # ── Pseudo-label path (set after running generate_pseudo_labels.py) ──
    "pseudo_labels_path":   "D:/birdclef_data/pseudo_labels/pseudo_labels_iter1_full.parquet",
    "labeled_fraction":     0.5,

    # ── SoftAUC enabled for pseudo stage ──────────────────────────────────
    "auc_weight":           0.3,

    # ── Slightly reduced LR for fine-tuning ───────────────────────────────
    "lr":                   5e-4,

    # ── Start from Stage 1 best checkpoint ────────────────────────────────
    # Set pretrained_path to the best fold_0 checkpoint from selected_eca
    "pretrained":           False,
    "pretrained_path":      None,  # e.g. "logdir/eca_nfnet.../fold_0/checkpoints/best.ckpt"
})
