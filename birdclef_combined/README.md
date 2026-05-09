# BirdCLEF+ 2025 — Multi-Taxonomic Species Identification from Noisy Soundscapes

> **Major Project | Deep Learning & Bioacoustics**

---

## Problem Statement

**Task**: Identify 206 species (birds, amphibians, insects, mammals) from 1-minute
passive acoustic monitoring soundscapes recorded in Colombia's Magdalena Valley.

**Key challenges**:
- Severe domain shift: clean single-species training clips vs. noisy multi-species soundscapes
- Extreme class imbalance: some species have only 2 recordings (~11 seconds total)
- Multi-label nature: soundscapes contain up to 25 simultaneous species
- Multi-taxonomic classification across four biological classes

---

## Contributions

This project introduces components on top of a strong audio classification baseline:

| # | Feature | What it does |
|---|---------|-------------|
| 1 | **DANN Domain Adaptation** | Aligns training clip features with soundscape features via gradient reversal to close the domain gap |
| 2 | **MC Dropout Uncertainty** | Filters unreliable pseudo-labels using Monte Carlo Dropout variance — only high-confidence predictions are retained |
| 3 | **Causal Feature Disentanglement** | Separates bird vocalizations (causal) from environmental noise (spurious) using HSIC independence penalty |
| 4 | **Prototypical Head for Rare Species** | Prototype-based nearest-neighbor classifier for species with fewer than 20 recordings |
| 5 | **Noise-Conditioned FiLM Layers** | Estimates ambient noise level from the spectrogram and adapts feature normalization dynamically |
| 6 | **Temporal Consistency Regularization** | Penalizes contradictory predictions across overlapping time windows of the same soundscape |
| 7 | **Taxonomy-Aware Hierarchical Loss** | Adds auxiliary loss at the taxonomic group level (Aves / Amphibia / Insecta / Mammalia) to improve rare-class generalization |
| 8 | **Multi-Resolution Spectral Fusion** | Stacks mel spectrograms at three time-frequency resolutions to capture both fine call structure and coarse song patterns |
| 9 | **Iterative Pseudo-Labeling with Uncertainty Filtering** | Power-scaled soft pseudo-labels across 3 iterations, filtered by MC Dropout confidence |
| 10 | **Stochastic Weight Averaging (SWA) + EMA** | SWA from epoch 37 onward combined with EMA decay (0.999) for better generalization |

---

## Architecture

### Backbone Ensemble
- **ECA-NFNet-L0** — RAdam optimizer, lr=1e-3
- **EfficientNetV2-S (ImageNet-21k)** — AdamW optimizer, lr=1e-4

Both backbones share:
- GeM pooling → 512-channel classification head
- Dropout 0.25 (post-backbone) + 0.5 (post-hidden)
- Cosine LR schedule to 1e-6, no warmup
- 50 epochs supervised, 20 epochs pseudo-label stage

### Training Strategy
- 5-fold cross-validation on merged BirdCLEF 2021–2025 dataset (~80K recordings)
- Balanced sampler with per-class frequency weighting
- MixUp augmentation (audio domain, p=0.5)
- Background noise injection from training soundscapes
- Gradient accumulation (effective batch size = 64)
- bf16-mixed precision training

---

## Exact Training Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 32,000 Hz |
| n_mels | 128 |
| f_min | 20 Hz |
| n_fft | 2,048 |
| hop_length | 512 |
| top_db | 80 |
| amin | 1e-10 |
| Effective batch size | 64 (4 × 16 grad accum) |
| Epochs (supervised) | 50 |
| Epochs (pseudo) | 20 |
| NFNet LR | 1e-3 (RAdam) |
| EffNetV2 LR | 1e-4 (AdamW, ε=1e-8, β=(0.9, 0.999)) |
| Min LR | 1e-6 (cosine schedule, no warmup) |
| Dropout | 0.25 (post-backbone), 0.5 (post-hidden) |
| Label smoothing α | 0.05 |
| MixUp | p=0.5, audio domain |
| SWA start | 75% of training |
| EMA decay | 0.999 |

---

## Project Structure

```
birdclef_combined/
├── code_base/
│   ├── models/
│   │   ├── backbone.py               # ECA-NFNet-L0, EfficientNetV2-S via timm
│   │   ├── train_module.py           # PyTorch Lightning training module
│   │   ├── causal_disentangle.py     # Novel #3: Causal feature disentanglement
│   │   ├── domain_adaptation.py      # Novel #1: DANN gradient reversal
│   │   ├── noise_conditioning.py     # Novel #5: FiLM noise conditioning
│   │   ├── prototypical_head.py      # Novel #4: Prototypical rare-species head
│   │   ├── cooccurrence_graph.py     # Species co-occurrence graph
│   │   └── ensemble.py               # Multi-backbone ensemble inference
│   ├── datasets/
│   │   └── audio_dataset.py          # Audio loading, balanced sampler, pseudo-label mixing
│   ├── losses/
│   │   ├── focal_bce.py              # FocalBCE + BCE combined loss
│   │   ├── taxonomy_loss.py          # Novel #7: Taxonomy-aware hierarchical loss
│   │   └── temporal_consistency.py   # Novel #6: Temporal consistency regularization
│   ├── augmentations/
│   │   ├── audio_aug.py              # MixUp, background noise injection
│   │   ├── multi_resolution_mel.py   # Novel #8: Multi-resolution spectral fusion
│   │   └── nnaudio_mel.py            # GPU mel spectrogram extraction
│   └── utils/
│       ├── calibration.py            # Temperature scaling + threshold optimization
│       ├── hdf5_utils.py             # HDF5 audio cache utilities
│       ├── pseudo_labels.py          # Power-scaled pseudo-label generation
│       ├── postprocessing.py         # Segment-level post-processing
│       └── uncertainty.py            # Novel #2: MC Dropout uncertainty estimation
├── scripts/
│   ├── run_full_pipeline.py          # One-command automated training pipeline
│   ├── main_train.py                 # Single fold training entry point
│   ├── generate_pseudo_labels.py     # Teacher inference → pseudo-label parquet
│   ├── merge_datasets.py             # Merge BirdCLEF 2021–2025 into unified dataset
│   ├── calibrate_and_threshold.py    # Post-training calibration
│   ├── build_taxonomy_map.py         # Build species → taxon JSON mapping
│   ├── build_prototypes.py           # Build prototype vectors for rare species
│   ├── main_inference.py             # Inference + submission generation
│   └── precompute_features.py        # Audio → HDF5 pre-computation
├── configs/
│   ├── species_to_taxon.json         # 206-species taxonomic group mapping
│   ├── train/
│   │   ├── selected_eca.py           # ECA-NFNet-L0 training config
│   │   ├── selected_ebs.py           # EfficientNetV2-S training config
│   │   ├── insects_amphibians.py     # Specialist config for non-bird classes
│   │   ├── pseudo_iter1_eca.py       # Pseudo-label iteration 1 config
│   │   ├── pseudo_iter2_ebs.py       # Pseudo-label iteration 2 config
│   │   └── pseudo_iter3_eca.py       # Pseudo-label iteration 3 config
│   └── inference/
│       └── ensemble_final.py         # Ensemble inference config
├── setup_vast.sh                     # One-command Vast.ai cloud GPU setup
└── pyproject.toml
```

---

## Setup & Training (Cloud — Vast.ai)

The recommended way to train is on a cloud GPU instance using the automated setup script.

### 1. Prepare (on your local machine)

- Accept BirdCLEF dataset rules on Kaggle (2021–2025)
- Get your Kaggle API key from kaggle.com → Settings → API → Create New Token
- Rent an RTX 3090 instance (24 GB VRAM, 200 GB disk) on vast.ai

### 2. Run the one-command setup (on the server)

```bash
git clone https://github.com/dhanushss30/major_project.git birdclef
cd birdclef/birdclef_combined
bash setup_vast.sh YOUR_KAGGLE_USERNAME YOUR_KAGGLE_KEY
```

This installs all dependencies, downloads BirdCLEF 2021–2025, and merges them into a unified training set.

### 3. Start training

```bash
python scripts/run_full_pipeline.py \
    --data_root   /workspace/birdclef-merged \
    --metadata_file train_merged.csv \
    --logdir      /workspace/logdir \
    --gpu 0
```

The pipeline runs end-to-end without any intervention:
- Stage 1: Supervised training (5 folds × 2 backbones)
- Stage 2: Pseudo-label generation with MC Dropout filtering
- Stage 3: Pseudo-label retraining (3 iterations)
- Stage 4: Calibration and threshold optimization

To resume after a crash:
```bash
python scripts/run_full_pipeline.py --data_root /workspace/birdclef-merged --resume
```

---

## Setup & Training (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2. Download the data

Download BirdCLEF 2025 from Kaggle and arrange as:

```
/your/data/root/
├── train.csv
├── train_audio/
├── train_soundscapes/
└── test_soundscapes/
```

### 3. Update data paths in configs

Edit `configs/train/selected_eca.py` and `configs/train/selected_ebs.py`:

```python
"data_root":       "/your/data/root",
"audio_root":      "/your/data/root/train_audio",
"soundscape_root": "/your/data/root/train_soundscapes",
```

### 4. Train

```bash
python scripts/run_full_pipeline.py --data_root /your/data/root --gpu 0
```
