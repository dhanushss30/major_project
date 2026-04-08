"""
pseudo_iter2_ebs.py — Noisy Student Iteration 2 (EfficientNetV2, OOF+Full)
============================================================================
From paper Table 5, best ensemble config:
  "OOF I2 + Full I2 (EffNetV2-S) → Public 0.917, Private 0.910"
  → Combined into final 0.925/0.928 ensemble with postprocessing

Pipeline:
  1. Train selected_ebs.py (Stage 1)
  2. python scripts/generate_pseudo_labels.py --mode full --iteration 2 --power_alpha 0.5
  3. python scripts/generate_pseudo_labels.py --mode oof  --iteration 2 --power_alpha 0.5
  4. Merge both parquets → combined_iter2.parquet
  5. Set pseudo_labels_path → train this config

SoftAUC weight increased to 0.3 since metric is AUC.
"""

import importlib.util, sys
from pathlib import Path

_spec   = importlib.util.spec_from_file_location(
    "ebs", Path(__file__).parent / "selected_ebs.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
config  = dict(_module.config)

config.update({
    "exp_name": (
        "tf_efficientnetv2_s_in21k_noamp_64bs_EqualBalancing_"
        "AdamW1e4_CosBatchLR1e6_Epoch50_FocalBCELoss_"
        "PseudoF2PT05MT01P05I2_OOFplusFull"
    ),
    "stage":              "pseudo",
    "pseudo_labels_path": "D:/birdclef_data/pseudo_labels/pseudo_labels_iter2_combined.parquet",
    "labeled_fraction":   0.5,
    "auc_weight":         0.3,
    "lr":                 3e-4,
    "pretrained":         False,
    "pretrained_path":    None,  # Set to Stage 1 best checkpoint
})


"""
pseudo_iter3_eca.py — Noisy Student Iteration 3 (ECA-NFNet, Full)
==================================================================
Final iteration. Teacher = Student_2. Power alpha=0.4 (most aggressive).
This model is the primary contributor to the final ensemble.

From paper Table 4: Iteration 3 adds 4,108 chunks from 1,437 files.
Total pseudo-labeled pool across all iterations:
  19,405 + 3,750 + 4,108 = 27,263 chunks
"""

import importlib.util
from pathlib import Path

_spec2   = importlib.util.spec_from_file_location(
    "eca", Path(__file__).parent / "selected_eca.py"
)
_module2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_module2)
config_iter3 = dict(_module2.config)

config_iter3.update({
    "exp_name": (
        "eca_nfnet_l0_noamp_64bs_SqrtBalancing_"
        "Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_"
        "PseudoF2PT05MT01P04I3_Full_MinorOverSampleV1"
    ),
    "stage":              "pseudo",
    # Cumulative pseudo: all 3 iterations merged
    "pseudo_labels_path": "D:/birdclef_data/pseudo_labels/pseudo_labels_all_iters_merged.parquet",
    "labeled_fraction":   0.5,
    "auc_weight":         0.3,
    "lr":                 2e-4,
    "pretrained":         False,
    "pretrained_path":    None,  # Set to Iter 2 best checkpoint
})
