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

    # ── Data paths (Vast.ai /workspace) ──────────────────────────────────
    # 2025-only paths (no merge step). To use merged 2021-2025, run
    # merge_datasets.py first and switch to /workspace/birdclef-merged.
    "data_root":         "/workspace/birdclef-2025",
    "metadata_file":     "train.csv",
    "audio_root":        "/workspace/birdclef-2025/train_audio",
    "hdf5_root":         None,   # Disabled — reads raw audio directly
    "soundscape_root":   "/workspace/birdclef-2025/train_soundscapes",
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
    "epochs":              50,     # Paper uses 50 epochs (was incorrectly set to 20)
    "limit_train_batches": 3000,  # cap steps/epoch: 3000×8=24K samples vs full 183K

    # ── Training setup — tuned for RTX 4090 24GB VRAM ───────────────────────
    "batch_size":        16,   # 24GB VRAM allows batch 16; was 4 for 4GB laptop
    "num_workers":       8,    # 16 CPU cores available on server; 8 workers for fast loading
    "precision":         "bf16-mixed",
    "accumulate_grad":   4,    # 16×4=64 effective batch (same as before, fewer accum steps)

    # ── Balanced sampler: γ=-0.5 for NFNet (paper Eq. 1, §5.2) ───────────
    "sampler_gamma":     -0.5,        # SqrtBalancing

    # ── Data augmentations (paper §5.2) ───────────────────────────────────
    "mixup_p":             0.5,       # Audio domain MixUp
    "mixup_scale":         0.5,
    "chunk_strategy":      "firstlast7",  # paper §5.5
    "use_secondary_labels": True,         # XC secondary labels

    # Background noise paths (set to actual paths)
    # On Vast.ai instance these are populated by the dataset-download step.
    "bg_soundscape_paths": [],   # Prior-year soundscape paths (BirdCLEF 2023 train_soundscapes)
    "bg_esc50_paths":      [],   # ESC-50 audio paths (https://github.com/karolpiczak/ESC-50)

    # ── Open-set negative-class training ─────────────────────────────────
    # With probability p_negative, replace a training sample's audio with a
    # random clip from negative_audio_dir (e.g. ESC-50) and set its label to
    # all-zero. This teaches the model "no bird present → all sigmoids low,"
    # which makes the inference-time max-prob threshold a real rejection
    # signal instead of just a low-confidence proxy.
    "negative_audio_dir":  None,   # Set to ESC-50 root on instance
    "p_negative":          0.1,    # 10% of training samples are non-bird

    # ── Aux loss warmup ──────────────────────────────────────────────────
    # Disabled — only FiLM is active and it is identity-initialised, so the
    # main BCE+Focal loss is the only signal training has to balance from step 0.
    "aux_warmup_epochs":   0,

    # ── NOVEL #1: DANN Domain Adaptation — DISABLED (caused collapse) ──────
    "use_dann":          False,
    "dann_lambda_max":   0.025,
    "dann_hidden_dim":   512,

    # ── NOVEL #4: Prototypical Head — DISABLED (auxiliary, removed for stability)
    "use_prototypical":    False,
    "proto_temperature":   0.1,
    "proto_margin":        0.3,
    "proto_rare_thr":      0.2,
    "proto_loss_weight":   0.025,
    "proto_half_life":     50,
    "proto_max_weight":    0.5,
    "prototypes_path":     None,
    "class_counts_path":   None,

    # ── NOVEL #5: Soundscape Noise-Conditioned FiLM Head — KEPT (the novel contribution)
    # Initialised as identity (γ=1, β=0) so it starts as a no-op and can only learn
    # to help. SoundscapeAudioBuffer is now constructed independently of DANN below.
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

    # ── NOVEL #8: Multi-Resolution Spectral Fusion — DISABLED
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

    # ── SWA (Stochastic Weight Averaging) — from 2nd place ─────────────
    "use_swa":          False,  # Disabled: deepcopy conflict with custom modules
    "swa_lr":           1e-6,      # SWA learning rate (= min_lr)
    "swa_start_frac":   0.75,      # Start SWA at 75% of training

    # ── Rare species oversampling — from 2nd place GitHub ───────────────
    # NFNet: 64 rare species get minor oversampling (10–96 extra draws)
    "oversampling_map":      {},    # Dict[species_code, target_count]
    "min_oversample_count":  100,   # Classes < 100 samples duplicated to 100
    "use_rating_weight":     True,  # Weight samples by XC rating (0-5); cleaner audio sampled more

    # ── Pseudo-label config (Stage 2) ─────────────────────────────────────
    "labeled_fraction":    0.5,   # 50% labeled, 50% pseudo (Figure 4)

    # ── EMA ───────────────────────────────────────────────────────────────
    "use_ema":    True,
    "ema_decay":  0.999,

    # ── Logging ───────────────────────────────────────────────────────────
    "logdir":     "logdir",
    "use_wandb":  False,
}
