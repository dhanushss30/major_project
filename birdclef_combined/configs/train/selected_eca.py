"""
selected_eca.py — Final ECA-NFNet-L0 Config (2nd Place Best Selected)
=======================================================================
Matches logdir name from GitHub README:
  eca_nfnet_l0_Exp_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_
  CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005_FromXCV2Best_
  PseudoF2PT05MT01P04I3_MinorOverSampleV1

Key parameters (Appendix B):
  - RAdam, lr=1e-3
  - Cosine to 1e-6, NO warmup
  - BalancedSampler γ=-0.5 (SqrtBalancing)
  - n_fft=2048, hop=512, n_mels=128

IMPORTANT: Set data_root, hdf5_root, soundscape_root, pseudo_labels_path
"""

config = {
    # ── Identity ──────────────────────────────────────────────────────────
    "exp_name": (
        "eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_"
        "Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005"
    ),
    "stage": "supervised",   # or "pseudo" when pseudo_labels_path is set

    # ── Data paths — UPDATE THESE ─────────────────────────────────────────
    "data_root":         "D:/birdclef_data/birdclef-2025",
    "metadata_file":     "train.csv",
    "audio_root":        "D:/birdclef_data/birdclef-2025/train_audio",
    "hdf5_root":         "D:/birdclef_data/birdclef_hdf5",  # Pre-decoded audio cache
    "soundscape_root":   "D:/birdclef_data/birdclef-2025/train_soundscapes",
    "pseudo_labels_path": None,   # Set to parquet path for Stage 2

    # ── Cross-validation ──────────────────────────────────────────────────
    "n_folds": 5,
    "seed":    42,

    # ── Model ─────────────────────────────────────────────────────────────
    "backbone":        "eca_nfnet_l0",
    "n_classes":       206,
    "pretrained":      True,
    "pretrained_path": None,   # Path to pre-trained backbone ckpt
    "hidden_dim":      512,    # Exact from paper Appendix B
    "dropout1":        0.25,   # Post-CNN (Appendix B)
    "dropout2":        0.5,    # Post-hidden (Appendix B)
    "gem_p":           3.0,
    "use_sed_head":    False,

    # ── Spectrogram (exact from paper Table 11 / Appendix B) ──────────────
    "sr":         32_000,
    "n_mels":     128,
    "fmin":       20.0,
    "fmax":       None,    # 16,000 (sr/2)
    "n_fft":      2_048,
    "hop_length": 512,
    "top_db":     80.0,
    "amin":       1e-10,

    # ── Loss ──────────────────────────────────────────────────────────────
    "bce_weight":   0.5,
    "focal_weight": 0.5,
    "auc_weight":   0.0,   # Stage 1: no SoftAUC; Stage 2: set to 0.3
    "focal_gamma":  2.0,

    # ── Label smoothing α=0.05 (paper Eq. 2) ──────────────────────────────
    "label_smoothing": 0.05,

    # ── Optimizer: RAdam (NFNet, Appendix B) ──────────────────────────────
    "lr":           1e-3,   # RAdam for NFNet
    "min_lr":       1e-6,
    "weight_decay": 1e-4,
    "grad_clip":    5.0,

    # ── Scheduler: cosine, NO warmup (Appendix B) ─────────────────────────
    "epochs":              20,
    "limit_train_batches": 3000,  # cap steps/epoch: 3000×8=24K samples vs full 183K

    # ── Training setup (Appendix B) ───────────────────────────────────────
    "batch_size":        8,
    "num_workers":       0,
    "precision":         "bf16-mixed", # bf16 stable with NFNet; no fp16 overflow
    "accumulate_grad":   8,   # effective batch = 64 (8×8)

    # ── Balanced sampler: γ=-0.5 for NFNet (paper Eq. 1, §5.2) ───────────
    "sampler_gamma":     -0.5,        # SqrtBalancing

    # ── Data augmentations (paper §5.2) ───────────────────────────────────
    "mixup_p":             0.5,       # Audio domain MixUp
    "mixup_scale":         0.5,
    "chunk_strategy":      "firstlast7",  # paper §5.5
    "use_secondary_labels": True,         # XC secondary labels

    # Background noise paths (set to actual paths)
    "bg_soundscape_paths": [],   # Prior-year soundscape paths
    "bg_esc50_paths":      [],   # ESC-50 audio paths

    # ── Pseudo-label config (Stage 2) ─────────────────────────────────────
    "labeled_fraction":    0.5,   # 50% labeled, 50% pseudo (Figure 4)

    # ── EMA ───────────────────────────────────────────────────────────────
    "use_ema":    True,
    "ema_decay":  0.999,

    # ── Logging ───────────────────────────────────────────────────────────
    "logdir":     "logdir",
    "use_wandb":  False,
}
