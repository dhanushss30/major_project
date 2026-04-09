"""
pseudo_iter3_eca.py — Noisy Student Iteration 3 (ECA-NFNet, Final)
===================================================================
Final pseudo iteration. Teacher = Student from Iteration 2.
Power alpha=0.4 (most aggressive compression).

From paper Table 4: Iteration 3 adds 4,108 chunks from 1,437 files.
Cumulative pseudo pool: 19,405 + 3,750 + 4,108 = 27,263 chunks.

This is the model used in the final 15-model ensemble.

Pipeline before running this:
  1. selected_eca.py        (Stage 1 supervised)
  2. generate_pseudo_labels  --iteration 1 --power_alpha 0.6
  3. pseudo_iter1_eca.py    (Stage 2)
  4. generate_pseudo_labels  --iteration 2 --power_alpha 0.5
  5. pseudo_iter2_ebs.py    (Stage 3)
  6. generate_pseudo_labels  --iteration 3 --power_alpha 0.4  ← feeds THIS config
  7. THIS CONFIG            (Stage 4 — final ECA student)
"""

import importlib.util
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
        "Radamlr1e3_CosBatchLR1e6_Epoch20_FocalBCELoss_"
        "PseudoF2PT05MT01P04I3_Full"
    ),
    "stage":   "pseudo",

    # All 3 iterations merged into one parquet
    # Run: python scripts/merge_pseudo_labels.py to create this file
    "pseudo_labels_path": "D:/birdclef_data/pseudo_labels/pseudo_labels_all_iters_merged.parquet",

    "labeled_fraction":   0.5,     # 50% real data, 50% pseudo
    "auc_weight":         0.3,     # SoftAUC enabled (paper §5.3)
    "lr":                 2e-4,    # Lower LR — fine-tuning, not training from scratch
    "epochs":             20,      # Fewer epochs for pseudo stage
    "pretrained":         False,
    "pretrained_path":    None,    # Set to best checkpoint from pseudo_iter1_eca
})
