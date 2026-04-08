"""
ensemble_final.py — Final 15-Model Ensemble Config
====================================================
Combining the exact 3 experiments × 5 folds = 15 models from 2nd place
(Private AUC 0.928) + TTA and specialist blending from 1st place.

Paper §5.7:
  "simple average over the predictions of all five folds across three
   selected experiments, resulting in an ensemble of 15 models."

Optuna-selected experiments (Table 6):
  Best Public ensemble:   0.925 Public / 0.928 Private
  Optuna ensemble:        0.921 Public / 0.929 Private

Post-processing: TopN N=1 (paper §5.6)
TTA: ±2.5s shifts (1st place)
"""

config = {
    # ── Shared settings ───────────────────────────────────────────────────
    "sr":          32_000,
    "n_mels":      128,
    "fmin":        20.0,
    "n_fft":       2_048,
    "hop_length":  512,
    "top_db":      80.0,
    "amin":        1e-10,
    "n_classes":   206,
    "chunk_secs":  5.0,

    # ── Data paths ────────────────────────────────────────────────────────
    "data_root":             "D:/birdclef_data/birdclef-2025",
    "metadata_file":         "train.csv",
    "hdf5_root":             None,   # Disabled — reads raw audio directly
    "audio_root":            "D:/birdclef_data/birdclef-2025/train_audio",
    "test_soundscape_dir":   "D:/birdclef_data/birdclef-2025/test_soundscapes",
    "logdir":    "logdir",
    "exp_name":  "eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005",
    "n_folds":   5,

    # ── Experiment(s) — update manifest path after training ──────────────
    # After training fold 0, the manifest is written to:
    #   logdir/<exp_name>/checkpoint_manifest.json
    "experiments": [
        {
            "name":     "eca_nfnet_stage1",
            "backbone": "eca_nfnet_l0",
            "manifest": "logdir/eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/checkpoint_manifest.json",
            "weight":   1.0,
        },
    ],

    # ── Specialist blend — disabled until specialist model is trained ──────
    "specialist_experiment": None,
    "specialist_class_indices": [],

    # ── Post-processing ───────────────────────────────────────────────────
    "postprocess_n": 1,    # TopN with N=1 (paper §5.6, best result)

    # ── TTA (1st place ±2.5s shifts) ──────────────────────────────────────
    "use_tta":       True,
    "tta_shifts":    [-2.5, 0.0, 2.5],

    # ── Primary experiment for OOF metric & ONNX export ───────────────────
    "primary_exp_name": (
        "eca_nfnet_l0_noamp_64bs_SqrtBalancing_"
        "Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_"
        "PseudoF2PT05MT01P04I3_Full_MinorOverSampleV1"
    ),

    # ── ONNX/OpenVINO export settings ─────────────────────────────────────
    "onnx_opset":        17,
    "export_fp16_ov":    True,
}
