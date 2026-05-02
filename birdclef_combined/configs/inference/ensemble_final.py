"""
ensemble_final.py — Final 20-Model Ensemble Config
====================================================
4 experiments × 5 folds = 20 models (all runs included; Optuna trims to best 3).

Experiments:
  1. ECA-NFNet-L0  supervised Stage 1           (backbone: eca_nfnet_l0)
  2. ECA-NFNet-L0  Pseudo Iter 1 student        (backbone: eca_nfnet_l0)
  3. EfficientNetV2-S Pseudo Iter 2 student     (backbone: tf_efficientnetv2_s)
  4. ECA-NFNet-L0  Pseudo Iter 3 student (FINAL)(backbone: eca_nfnet_l0)

Post-processing: TopN N=1 (paper §5.6)
TTA: ±2.5s shifts (1st place)

Novel additions:
  - Per-class temperature calibration + adaptive thresholds (Novel #3)
  - Species co-occurrence GCN refinement (Novel #2)
  - Specialist insects/amphibians blending (1st place)

manifest paths are auto-filled by run_pipeline.py; update below after training.
"""

config = {
    # ── Shared audio settings ──────────────────────────────────────────────
    "sr":          32_000,
    "n_mels":      128,
    "fmin":        20.0,
    "n_fft":       2_048,
    "hop_length":  512,
    "top_db":      80.0,
    "amin":        1e-10,
    "n_classes":   206,
    "chunk_secs":  5.0,

    # ── Data paths (update to your paths) ────────────────────────────────
    "data_root":           "D:/birdclef_data/birdclef-2025",
    "metadata_file":       "train.csv",
    "hdf5_root":           None,
    "audio_root":          "D:/birdclef_data/birdclef-2025/train_audio",
    "test_soundscape_dir": "D:/birdclef_data/birdclef-2025/test_soundscapes",
    "logdir":              "logdir",
    "n_folds":             5,

    # ── Experiments (3-4 × 5 folds = 15-20 models) ───────────────────────
    # Manifests are auto-written by main_train.py at:
    #   logdir/<exp_name>/checkpoint_manifest.json
    # run_pipeline.py fills these in automatically.
    "experiments": [
        {
            # Stage 1: ECA-NFNet supervised (baseline backbone)
            "name":     "eca_nfnet_stage1",
            "backbone": "eca_nfnet_l0",
            "manifest": (
                "logdir/eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_"
                "Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005"
                "/checkpoint_manifest.json"
            ),
            "weight":   1.0,
        },
        {
            # Pseudo Iter 1: ECA-NFNet student trained on iter-1 pseudo labels
            "name":     "eca_nfnet_pseudo1",
            "backbone": "eca_nfnet_l0",
            "manifest": (
                "logdir/eca_nfnet_l0_noamp_64bs_SqrtBalancing_"
                "Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_"
                "PseudoF2PT05MT01P06I1_Full"
                "/checkpoint_manifest.json"
            ),
            "weight":   1.0,
        },
        {
            # Pseudo Iter 2: EfficientNetV2-S student (OOF+Full merged pseudo)
            "name":     "ebs_pseudo2",
            "backbone": "tf_efficientnetv2_s",
            "manifest": (
                "logdir/tf_efficientnetv2_s_in21k_noamp_64bs_EqualBalancing_"
                "AdamW1e4_CosBatchLR1e6_Epoch50_FocalBCELoss_"
                "PseudoF2PT05MT01P05I2_OOFplusFull"
                "/checkpoint_manifest.json"
            ),
            "weight":   1.0,
        },
        {
            # Pseudo Iter 3: ECA-NFNet final student (all 3 iter pseudo merged)
            "name":     "eca_nfnet_pseudo3",
            "backbone": "eca_nfnet_l0",
            "manifest": (
                "logdir/eca_nfnet_l0_noamp_64bs_SqrtBalancing_"
                "Radamlr1e3_CosBatchLR1e6_Epoch20_FocalBCELoss_"
                "PseudoF2PT05MT01P04I3_Full"
                "/checkpoint_manifest.json"
            ),
            "weight":   1.0,
        },
    ],

    # ── Specialist blend (1st place) ──────────────────────────────────────
    # Set after training insects_amphibians.py specialist model.
    # specialist_class_indices: integer indices in [0, 205] for non-bird species.
    # After building label_to_idx.json, run:
    #   python -c "import json; lbl=json.load(open('logdir/<exp>/label_to_idx.json'));
    #              import pandas as pd; df=pd.read_csv('train.csv');
    #              non_bird=df[df.class_name!='Aves'].primary_label.unique();
    #              print([lbl[x] for x in non_bird if x in lbl])"
    "specialist_experiment": {
        "name":     "efficientnet_b0_specialist",
        "backbone": "efficientnet_b0",
        "manifest": (
            "logdir/efficientnet_b0_Insects_Amphibians_Specialist_"
            "EqualBalancing_Epoch30/checkpoint_manifest.json"
        ),
        "weight":   0.4,    # Blend weight vs generalist
    },
    # Indices of non-bird species in the 206-class label space.
    # Leave empty [] to disable specialist blending.
    "specialist_class_indices": [],   # Populated by run_pipeline.py

    # ── Novel #2: Species Co-occurrence GCN ───────────────────────────────
    "gcn_checkpoint": "logdir/novel_components/species_gcn.pt",

    # ── Novel #3: Calibration artifacts (from train_novel_components.py) ──
    "calibration_dir": "logdir/novel_components",

    # ── Novel #4: Prototypical head for rare species ───────────────────────
    # Built by: python scripts/build_prototypes.py --manifest <eca_stage1_manifest>
    #           --data_root D:/birdclef_data/birdclef-2025
    #           --output_dir logdir/eca_nfnet_l0_.../
    #           --backbone eca_nfnet_l0
    # Set to None to disable prototype blending.
    "prototypes_path":   "logdir/eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/class_prototypes.pt",
    "class_counts_path": "logdir/eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/class_counts.pt",
    "proto_temperature": 0.05,
    "proto_half_life":   50,
    "proto_max_weight":  0.75,

    # ── Novel #5: Noise conditioning (inference-time) ──────────────────────
    # Only activates if models were trained with use_noise_conditioning=True
    "use_noise_conditioning": True,
    "noise_dim": 128,

    # ── Post-processing ───────────────────────────────────────────────────
    "postprocess_n": 1,    # TopN with N=1 (paper §5.6, best result)

    # ── TTA (1st place ±2.5s shifts) ──────────────────────────────────────
    "use_tta":    True,
    "tta_shifts": [-2.5, 0.0, 2.5],

    # ── Primary experiment for label map + OOF metric ─────────────────────
    # Must match an exp_name whose label_to_idx.json exists in logdir/.
    "primary_exp_name": (
        "eca_nfnet_l0_noamp_64bs_SqrtBalancing_"
        "Radamlr1e3_CosBatchLR1e6_Epoch20_FocalBCELoss_"
        "PseudoF2PT05MT01P04I3_Full"
    ),

    # ── Novel #7: Taxonomy-aware post-processing ────────────────────────
    # If taxonomy head was trained, taxon-level predictions can gate
    # species predictions at inference time.
    "use_taxonomy_gating":  True,
    "taxonomy_gate_thresh": 0.3,

    # ── Novel #8: Multi-Resolution Mel ────────────────────────────────────
    # Models trained with multi-res mel need this flag at inference.
    "use_multi_res_mel":  False,
    "multi_res_mode":     "stack",

    # ── ONNX/OpenVINO export settings ─────────────────────────────────────
    "onnx_opset":     17,
    "export_fp16_ov": True,
}
