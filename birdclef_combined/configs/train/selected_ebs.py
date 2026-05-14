"""
selected_ebs.py — Final EfficientNetV2-S Config (2nd Place Best Selected)
==========================================================================
Matches logdir name from GitHub README:
  tf_efficientnetv2_s_in21k_Exp_noamp_64bs_5sec_BasicAug_EqualBalancing_
  AdamW1e4_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005_FromPrebs1_
  PseudoF2PT05MT01P04I2_AddRareBirdsNoLeak

Key differences from ECA config:
  - AdamW, lr=1e-4, ε=1e-8, β=(0.9,0.999)     [not RAdam]
  - EqualBalancing (γ=-1.0)                     [not Sqrt γ=-0.5]
  - in21k pretrained backbone                   [stronger initialization]
"""

config = {
    "exp_name": (
        "tf_efficientnetv2_s_in21k_noamp_64bs_5sec_BasicAug_EqualBalancing_"
        "AdamW1e4_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005"
    ),
    "stage": "supervised",

    # ── Data (Vast.ai /workspace) ─────────────────────────────────────────
    # 2025-only paths (no merge step). To use merged 2021-2025, run
    # merge_datasets.py first and switch to /workspace/birdclef-merged.
    "data_root":          "/workspace/birdclef-2025",
    "metadata_file":      "train.csv",
    "audio_root":         "/workspace/birdclef-2025/train_audio",
    "hdf5_root":          None,   # Disabled — reads raw audio directly
    "soundscape_root":    "/workspace/birdclef-2025/train_soundscapes",
    "pseudo_labels_path": None,
    "n_folds": 5,
    "seed":    42,

    # ── Model ─────────────────────────────────────────────────────────────
    "backbone":        "tf_efficientnetv2_s",
    "n_classes":       206,
    "pretrained":      True,       # ImageNet-21k pretrain
    "pretrained_path": None,
    "hidden_dim":      512,
    "dropout1":        0.25,
    "dropout2":        0.5,
    "gem_p":           3.0,
    "use_sed_head":    False,

    # ── Spectrogram (exact Appendix B) ────────────────────────────────────
    "sr":         32_000,
    "n_mels":     128,
    "fmin":       20.0,
    "fmax":       None,
    "n_fft":      2_048,
    "hop_length": 512,
    "top_db":     80.0,
    "amin":       1e-10,

    # ── Loss ─────────────────────────────────────────────────────────────
    "bce_weight":   0.5,
    "focal_weight": 0.5,
    "auc_weight":   0.0,
    "focal_gamma":  2.0,
    "label_smoothing": 0.05,

    # ── Optimizer: AdamW (EffNetV2, Appendix B) ───────────────────────────
    "lr":           1e-4,       # AdamW for EffNetV2
    "min_lr":       1e-6,
    "weight_decay": 1e-4,
    "grad_clip":    5.0,

    # ── Schedule: cosine, NO warmup (Appendix B) ──────────────────────────
    "epochs": 50,
    "limit_train_batches": 3000,  # Must match ECA — without this, epoch = ~146K steps (forever)

    # ── Training — tuned for RTX 4090 24GB VRAM ─────────────────────────────
    "batch_size":      16,   # 24GB VRAM allows batch 16; was 4 for 4GB laptop
    "num_workers":     8,    # 16 CPU cores available on server; 8 workers for fast loading
    "precision":       "bf16-mixed",
    "accumulate_grad": 4,    # 16×4=64 effective batch (same as before, fewer accum steps)

    # ── EqualBalancing: γ=-1.0 for EffNetV2 (paper §5.2) ─────────────────
    "sampler_gamma":  -1.0,

    # ── Augmentations ─────────────────────────────────────────────────────
    "mixup_p":              0.5,
    "mixup_scale":          0.5,
    "chunk_strategy":       "firstlast7",
    "use_secondary_labels": True,
    "bg_soundscape_paths":  [],
    "bg_esc50_paths":       [],

    # ── Aux loss warmup ──────────────────────────────────────────────────
    # Disabled — only FiLM (identity-initialised) is active, no aux loss to ramp.
    "aux_warmup_epochs":   0,

    # ── NOVEL #1: DANN Domain Adaptation — DISABLED (caused collapse) ──────
    "use_dann":          False,
    "dann_lambda_max":   0.025,
    "dann_hidden_dim":   512,

    # ── NOVEL #4: Prototypical Head — DISABLED
    "use_prototypical":    False,
    "proto_temperature":   0.1,
    "proto_margin":        0.3,
    "proto_rare_thr":      0.2,
    "proto_loss_weight":   0.025,
    "proto_half_life":     50,
    "proto_max_weight":    0.5,
    "prototypes_path":     None,
    "class_counts_path":   None,

    # ── NOVEL #5: FiLM Noise Conditioning — KEPT (the novel contribution) ─
    "use_noise_conditioning": True,
    "noise_dim":              128,

    # ── NOVEL #6: Temporal Consistency Regularization — DISABLED
    "use_tcr":            False,
    "tcr_weight":         0.05,
    "tcr_max_gap":        5.0,
    "tcr_temperature":    2.0,

    # ── NOVEL #7: Taxonomy-Aware Hierarchical Loss — DISABLED
    "use_taxonomy":                  False,
    "taxonomy_hidden_dim":           128,
    "taxonomy_aux_weight":           0.025,
    "taxonomy_consistency_weight":   0.01,
    "taxonomy_confusion_weight":     0.005,
    "species_to_taxon":              {},
    "species_to_taxon_path":         None,

    # ── NOVEL #8: Multi-Resolution Mel — DISABLED
    "use_multi_res_mel":  False,
    "multi_res_mode":     "stack",

    # ── NOVEL #10: Causal Feature Disentanglement — DISABLED (primary collapse culprit)
    "use_causal":              False,
    "causal_dim":              768,
    "spurious_dim":            512,
    "causal_dropout":          0.1,
    "causal_lambda_inv":       0.01,
    "causal_lambda_hsic":      0.01,
    "causal_warmup_steps":     5000,
    "causal_cls_weight":       0.025,

    # ── Early stopping (saves compute, never reduces AUC) ─────────────
    "early_stopping":            True,
    "early_stopping_patience":   10,      # epochs without improvement before stop
    "early_stopping_min_delta":  0.0005,  # AUC improvement threshold

    # ── SWA ──────────────────────────────────────────────────────────────
    "use_swa":          False,  # Disabled: deepcopy conflict with custom modules
    "swa_lr":           1e-6,
    "swa_start_frac":   0.75,

    # ── Rare species oversampling ────────────────────────────────────────
    "oversampling_map":      {},
    "min_oversample_count":  100,
    "use_rating_weight":     True,

    # ── Pseudo-label (Stage 2) ────────────────────────────────────────────
    "labeled_fraction": 0.5,
    "auc_weight_pseudo": 0.3,

    # ── EMA ──────────────────────────────────────────────────────────────
    "use_ema":   True,
    "ema_decay": 0.999,

    # ── Logging ──────────────────────────────────────────────────────────
    "logdir":    "logdir",
    "use_wandb": False,
}
