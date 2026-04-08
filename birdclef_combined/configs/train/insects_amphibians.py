"""
insects_amphibians.py — Specialist Pipeline (1st Place, +0.003 AUC)
=====================================================================
From 1st place writeup:
  "Separate pipeline for insects and amphibians" → 0.930 → 0.933 AUC

Strategy:
  - Filter training data to Insecta (17 species) + Amphibia (34 species)
  - Use EfficientNet-B0 (lighter, faster for specialist task)
  - Equal balancing (γ=-1.0): all classes are rare here
  - During inference: blend specialist predictions for those class indices
  - EfficientNet-B0 for insects/amphibians specialist (paper mentions EfficientNet-B0)

Taxonomic breakdown (Table 1 in paper):
  Aves:     146 species
  Amphibia:  34 species
  Insecta:   17 species
  Mammalia:   9 species

The specialist only outputs scores for the 51 non-bird classes.
During ensemble blending, these scores replace/augment the generalist's output.
"""

# Indices of non-bird species in the 206-class taxonomy
# UPDATE THIS based on your train_metadata.csv — these are placeholder values
# Query: df[df['class_name'].isin(['Amphibia','Insecta','Mammalia'])]['primary_label'].unique()
INSECT_AMPHIBIAN_MAMMAL_LABELS = []  # Populate from metadata

config = {
    "exp_name": "efficientnet_b0_Insects_Amphibians_Specialist_EqualBalancing_Epoch30",
    "stage":    "supervised",

    # ── Data ─────────────────────────────────────────────────────────────
    "data_root":          "D:/birdclef_data/birdclef-2025",
    "metadata_file":      "train.csv",
    "audio_root":         "D:/birdclef_data/birdclef-2025/train_audio",
    "hdf5_root":          None,   # Disabled — reads raw audio directly
    "soundscape_root":    "D:/birdclef_data/birdclef-2025/train_soundscapes",
    "pseudo_labels_path": None,

    # Filter metadata to only insects, amphibians, mammals
    "filter_labels": INSECT_AMPHIBIAN_MAMMAL_LABELS,
    # If empty: train on full 206 classes but upweight non-bird species

    "n_folds": 5,
    "seed":    42,

    # ── Model (lighter for specialist) ────────────────────────────────────
    "backbone":        "efficientnet_b0",
    "n_classes":       206,     # Keep full 206 for easy ensemble blending
    "pretrained":      True,
    "pretrained_path": None,
    "hidden_dim":      256,     # Smaller head for specialist
    "dropout1":        0.3,
    "dropout2":        0.5,
    "gem_p":           3.0,
    "use_sed_head":    False,

    # ── Spectrogram ───────────────────────────────────────────────────────
    "sr":         32_000,
    "n_mels":     128,
    "fmin":       20.0,
    "fmax":       None,
    "n_fft":      2_048,
    "hop_length": 512,
    "top_db":     80.0,
    "amin":       1e-10,

    # ── Loss (SoftAUC enabled: rare classes benefit more from AUC opt) ────
    "bce_weight":      0.5,
    "focal_weight":    0.5,
    "auc_weight":      0.3,
    "focal_gamma":     2.0,
    "label_smoothing": 0.05,

    # ── Optimizer: AdamW ─────────────────────────────────────────────────
    "lr":          1e-4,
    "min_lr":      1e-6,
    "weight_decay": 1e-4,
    "grad_clip":   5.0,

    # ── Fewer epochs (smaller dataset) ───────────────────────────────────
    "epochs": 30,

    # ── Training ─────────────────────────────────────────────────────────
    "batch_size":      8,
    "num_workers":     0,
    "precision":       "32-true",
    "accumulate_grad": 8,    # Effective batch = 64

    # ── Equal balancing: all classes are rare in this subset ──────────────
    "sampler_gamma":   -1.0,

    # ── Augmentations ─────────────────────────────────────────────────────
    "mixup_p":              0.5,
    "mixup_scale":          0.5,
    "chunk_strategy":       "firstlast7",
    "use_secondary_labels": True,
    "bg_soundscape_paths":  [],
    "bg_esc50_paths":       [],

    # ── Ensemble blending ─────────────────────────────────────────────────
    # These are the class indices in the 206-class taxonomy for non-bird species
    # Populate from: sorted(df[df.class_name != 'Aves'].primary_label.unique())
    "specialist_class_indices": [],  # Update after building label_to_idx
    "specialist_blend_weight":  0.4, # How much to weight specialist vs generalist

    # ── EMA ──────────────────────────────────────────────────────────────
    "use_ema":   True,
    "ema_decay": 0.999,

    # ── Logging ──────────────────────────────────────────────────────────
    "logdir":    "logdir",
    "use_wandb": False,
}
