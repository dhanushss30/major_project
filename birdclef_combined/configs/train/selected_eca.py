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
    # lr lowered from 1e-3 (paper) to 5e-4 — at the paper rate the cosine
    # decay over 50 epochs was too slow, keeping LR ~76% of peak well past
    # the AUC peak (epoch 6, val/auc 0.69) and causing collapse to 0.61 by
    # epoch 11. 5e-4 + epochs=30 lands the model in a stable basin.
    "lr":           5e-4,
    "min_lr":       1e-6,
    "weight_decay": 1e-4,
    "grad_clip":    5.0,

    # ── Scheduler: cosine, NO warmup (Appendix B) ─────────────────────────
    "epochs":              30,    # Was 50, but val/auc plateaus early. Tighter cosine.
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

    # ── Background noise augmentation (paper §5.2, p=0.5, SNR 3-20 dB) ──
    # main_train.py globs bg_soundscape_dir / bg_esc50_dir into the *_paths
    # lists below, sampled down to bg_max_paths each to bound the mixer's
    # in-memory cache. Set the *_dir to None to skip BG mixing.
    #
    # Stage 1 v3 trained without BG mixing (these dirs were empty) and
    # produced a model that's confident on focal recordings but blind on
    # soundscape audio (max common-bird prob in pseudo-gen: 0.34). Adding
    # real soundscape + ESC-50 BG at training time bridges the domain gap.
    "bg_soundscape_dir":   "/workspace/birdclef-2025/train_soundscapes",
    "bg_esc50_dir":        "/workspace/ESC-50-nobird",
    "bg_max_paths":        500,
    "bg_soundscape_paths": [],   # auto-populated from bg_soundscape_dir
    "bg_esc50_paths":      [],   # auto-populated from bg_esc50_dir

    # ── Rare-class exclusion ─────────────────────────────────────────────
    # Drop training rows for classes with fewer than this many samples.
    # Stage 1 v3 trained on all 206 classes including 78 rare iNat taxa
    # (amphibians/insects) with 2-3 samples each. The model memorized those
    # few samples and produced spurious high-confidence predictions on
    # soundscape audio, completely dominating pseudo-label generation
    # (top-8 pseudo-classes were all rare taxa, zero common birds passed
    # the max-prob threshold). label_to_idx stays at 206 entries; rare
    # classes get only label-smoothing (~0.00024) supervision and the
    # model learns to output ≈0 for them — exactly what we want.
    "min_class_samples_to_train": 50,

    # ── Open-set negative-class training ─────────────────────────────────
    # With probability p_negative, replace a training sample's audio with a
    # random clip from negative_audio_dir (e.g. ESC-50) and set its label to
    # all-zero. This teaches the model "no bird present → all sigmoids low,"
    # which makes the inference-time max-prob threshold a real rejection
    # signal instead of just a low-confidence proxy.
    # negative_audio_dir is wired but p_negative=0.0 disables injection for
    # Stage 1. At 10% the zero-label pressure dominated once the model began
    # to learn species (val/auc collapsed 0.69 -> 0.61 over 5 epochs).
    # "not a bird" detection is handled at inference via max-prob thresholding,
    # so disabling negatives here does not affect the deliverable.
    # Re-enable (p_negative=0.05) in pseudo iter 1 once the model is stronger.
    "negative_audio_dir":  "/workspace/ESC-50-nobird",
    "p_negative":          0.0,

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

    # ── NOVEL #5: FiLM Noise Conditioning — DISABLED ──────────────────────
    # Was initialised as identity (γ=1, β=0) and trained end-to-end. In practice
    # both main and EMA gamma_net.bias collapsed from 1.0 to 0.04 / 0.19 over
    # ~6 epochs, attenuating features 25x. Val/auc 0.7841 was preserved
    # (val is close to training distribution) but pseudo-label inference on
    # OOD soundscapes produced near-constant outputs (per-chunk range 0.059).
    # Disabled for stability. Novel contributions move to inference-time:
    #   #1 open-set max-prob rejection (handles "not a bird" without training)
    #   #2 per-class temperature calibration + adaptive thresholds (post-hoc)
    "use_noise_conditioning": False,
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
