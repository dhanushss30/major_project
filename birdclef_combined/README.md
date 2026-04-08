# BirdCLEF+ 2025 — Combined 1st & 2nd Place Solution
## Multi-Taxonomic Species Identification from Noisy Soundscapes

> **Major Project | Deep Learning & Bioacoustics**  
> Combines winning solutions from the BirdCLEF+ 2025 Kaggle Competition (2,025 teams)

---

## Competition Context

**Task**: Identify 206 species (birds, amphibians, insects, mammals) from 1-minute  
passive acoustic monitoring soundscapes recorded in Colombia's Magdalena Valley.

**Key challenges**:
- Severe domain shift: clean single-species training clips vs. noisy multi-species soundscapes
- Extreme class imbalance: some species have only 2 recordings (~11 seconds total)
- Strict CPU-only inference: 700 soundscapes in 90 minutes
- Multi-label nature: soundscapes contain up to 25 simultaneous species

---

## Solution Architecture

### From 2nd Place (Sydorskyi & Gonçalves — Private AUC 0.928)
- ECA-NFNet-L0 + EfficientNetV2-S-in21k dual-backbone ensemble
- HDF5 byte-wise random 5-sec chunk loading
- nnAudio on-the-fly GPU mel extraction (n_fft=2048, hop=512, n_mels=128)
- GeM pooling with 512-channel classification head (dropout 0.25 + 0.5)
- RAdam (NFNet) / AdamW (EffNetV2) + cosine LR to 1e-6, 50 epochs, batch 64
- MixUp (audio domain, p=0.5) + BackgroundNoise (soundscape + ESC-50, p=0.5)
- SpecAugment + RandomFiltering (random equalizer for channel distortion simulation)
- Power-scaled soft pseudo-labels, iterative: 3 rounds (19,405 + 3,750 + 4,108 chunks)
- TopN post-processing (N=1): multiply per-segment prob by top-1 segment prob in same file
- Optuna-based ensemble selection across 15 models (3 experiments × 5 folds)
- OpenVINO FP16 export for CPU inference

### From 1st Place (Babych — Private AUC ~0.933)
- Multi-iterative noisy student: power transform p^α (α: 0.6→0.5→0.4 per iteration)
- SED (Sound Event Detection) frame-level temporal localization for pseudo-label refinement
- Separate specialist pipeline for insects and amphibians (+0.003 AUC)
- TTA with ±2.5s temporal delta shifts
- SoftAUC loss: differentiable pairwise AUC optimization (since metric = AUC, optimize AUC)
- EfficientNet-L0, B3, B4 + RegNetY ensemble diversity

---

## Exact Training Parameters (from paper Appendix B)

| Parameter | Value |
|---|---|
| Sample rate | 32,000 Hz |
| n_mels | 128 |
| f_min | 20 Hz |
| n_fft | 2,048 |
| hop_length | 512 |
| top_db | 80 |
| amin | 1e-10 |
| Batch size | 64 |
| Epochs | 50 |
| NFNet LR | 1e-3 (RAdam) |
| EffNetV2 LR | 1e-4 (AdamW, ε=1e-8, β=(0.9,0.999)) |
| Min LR | 1e-6 (cosine schedule, no warmup) |
| Dropout | 0.25 (post-CNN), 0.5 (post-hidden) |
| Label smoothing α | 0.05 |
| MixUp | p=0.5, audio domain, max label |
| Background noise | p=0.5, soundscape + ESC-50 |
| Classification head | hidden=512 channels, ReLU |

---

## Project Structure

```
birdclef_combined/
├── code_base/
│   ├── models/
│   │   ├── backbone.py           # ECA-NFNet-L0, EfficientNetV2-S, B3/B4, RegNetY
│   │   ├── classification_head.py # 512-hidden GeM head (exact 2nd place architecture)
│   │   ├── sed_model.py          # SED with attention pooling (1st place)
│   │   └── ensemble.py           # 15-model ensemble + TTA + TopN postprocessing
│   ├── datasets/
│   │   ├── audio_dataset.py      # HDF5 + raw audio, balanced sampler (γ=-0.5, -1)
│   │   ├── soundscape_dataset.py # Unlabeled soundscape sliding window
│   │   └── pseudo_dataset.py     # Combined labeled + pseudo-labeled dataset
│   ├── losses/
│   │   ├── focal_bce.py          # FocalBCE + BCE combined loss
│   │   └── soft_auc.py           # Differentiable pairwise AUC (1st place)
│   ├── augmentations/
│   │   ├── audio_aug.py          # MixUp (audio), BackgroundNoise (ESC-50/soundscape)
│   │   ├── spec_aug.py           # SpecAugment, RandomFiltering
│   │   └── nnaudio_mel.py        # nnAudio GPU mel extractor (exact params from paper)
│   ├── utils/
│   │   ├── hdf5_utils.py         # Audio→HDF5 with byte-wise access
│   │   ├── pseudo_labels.py      # Power transform, filtering, OOF strategy
│   │   ├── postprocessing.py     # TopN postprocessing (exact 2nd place method)
│   │   └── metrics.py            # Padded macro ROC-AUC
│   └── nn_blocks/
│       └── gem_pooling.py        # GeM pooling [Radenović et al.]
├── scripts/
│   ├── precompute_features.py    # Audio→HDF5 (--use_torchaudio for MP3)
│   ├── main_train.py             # Full k-fold training pipeline
│   ├── generate_pseudo_labels.py # Teacher inference → pseudo-label selection
│   ├── main_inference.py         # Eval + ONNX + OpenVINO FP16 export
│   ├── create_pretrain_backbone.py # Extract backbone from pretrain checkpoint
│   └── download_data.py          # Kaggle API downloader
├── configs/
│   ├── train/
│   │   ├── pretrain_eca_2025.py   # Pre-train NFNet on 819k Xeno-Canto samples
│   │   ├── pretrain_ebs_2025.py   # Pre-train EffNetV2 on Xeno-Canto
│   │   ├── selected_eca.py        # Final NFNet config (exact 2nd place)
│   │   ├── selected_ebs.py        # Final EffNetV2 config (exact 2nd place)
│   │   ├── pseudo_iter1_eca.py    # Noisy student iter 1
│   │   ├── pseudo_iter2_ebs.py    # Noisy student iter 2
│   │   └── insects_amphibians.py  # Specialist pipeline (1st place)
│   └── inference/
│       └── ensemble_final.py      # 15-model ensemble config
├── notebooks/
│   ├── 01_EDA_Species_Distribution.ipynb
│   ├── 02_Domain_Shift_Analysis.ipynb
│   ├── 03_Pseudo_Label_Pipeline.ipynb
│   └── 04_Ensemble_Optuna_Selection.ipynb
├── pyproject.toml
└── rock_that_bird.sh              # End-to-end runner
```

---

## Setup & How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

> **CPU-only (no GPU):** Remove `+cu121` from the three torch lines in `requirements.txt`, then run `pip install -r requirements.txt`.

---

### 2. Download the data

Download the BirdCLEF+ 2025 dataset from Kaggle and place it so your folder structure looks like:

```
/your/data/root/
├── train.csv
├── train_audio/          # Per-species OGG clips
├── train_soundscapes/    # Unlabeled soundscapes for pseudo-labelling
└── test_soundscapes/     # Competition test soundscapes
```

---

### 3. Update data paths in configs

Open `configs/train/selected_eca.py` and `configs/train/selected_ebs.py` and set:

```python
"data_root":       "/your/data/root",
"audio_root":      "/your/data/root/train_audio",
"soundscape_root": "/your/data/root/train_soundscapes",
"hdf5_root":       "/your/data/hdf5",   # Where pre-computed HDF5 files will be saved
```

Do the same in `configs/inference/ensemble_final.py`:

```python
"data_root":           "/your/data/root",
"audio_root":          "/your/data/root/train_audio",
"test_soundscape_dir": "/your/data/root/test_soundscapes",
```

---

### 4. (Recommended) Pre-compute audio to HDF5

Converts raw audio to HDF5 for fast byte-wise chunk loading during training.
Use `--use_torchaudio` for MP3 files to avoid librosa artefacts.

```bash
cd birdclef_combined

python scripts/precompute_features.py \
    /your/data/root/train_audio \
    /your/data/hdf5 \
    --n_cores 8 \
    --use_torchaudio
```

---

### 5. Stage 1 — Supervised training (clean labels)

Train ECA-NFNet-L0 and EfficientNetV2-S on clean labelled data, 5-fold CV.

```bash
cd birdclef_combined

# ECA-NFNet-L0  (RAdam, lr=1e-3)
CUDA_VISIBLE_DEVICES=0 python scripts/main_train.py configs/train/selected_eca.py

# EfficientNetV2-S  (AdamW, lr=1e-4)
CUDA_VISIBLE_DEVICES=0 python scripts/main_train.py configs/train/selected_ebs.py
```

Checkpoints are saved to `logdir/<exp_name>/fold_<N>/checkpoints/best.ckpt`.
A `checkpoint_manifest.json` is written to `logdir/<exp_name>/` when all folds finish.

---

### 6. Generate pseudo-labels (noisy student, iteration 1)

Uses the Stage 1 ECA ensemble to label the unlabelled soundscapes.

```bash
cd birdclef_combined

python scripts/generate_pseudo_labels.py \
    --checkpoint_manifest logdir/eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/checkpoint_manifest.json \
    --soundscape_dir /your/data/root/train_soundscapes \
    --output_dir /your/data/pseudo_labels \
    --iteration 1 \
    --power_alpha 0.6 \
    --backbone eca_nfnet_l0
```

Output: `/your/data/pseudo_labels/pseudo_labels_iter1_full.parquet`

---

### 7. Stage 2 — Pseudo-label training (noisy student)

Set the `pseudo_labels_path` in `configs/train/pseudo_iter1_eca.py` to the parquet file above, then train:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/main_train.py configs/train/pseudo_iter1_eca.py
```

Repeat steps 6–7 for iterations 2 and 3 (`power_alpha` 0.5 → 0.4), using the latest
ensemble checkpoints each time.

---

### 8. Run inference / generate submission

```bash
cd birdclef_combined

# Generate competition submission CSV
python scripts/main_inference.py configs/inference/ensemble_final.py \
    --mode submit \
    --soundscape_dir /your/data/root/test_soundscapes \
    --output submission.csv

# Evaluate OOF validation AUC
python scripts/main_inference.py configs/inference/ensemble_final.py --mode eval

# Export ONNX + OpenVINO FP16 (for CPU deployment)
python scripts/main_inference.py configs/inference/ensemble_final.py --mode export
```

---

## References

1. Sydorskyi & Gonçalves. *Tackling Domain Shift in Bird Audio Classification via Transfer Learning and Semi-Supervised Distillation.* CLEF 2025 Working Notes.
2. Babych. *Multi-Iterative Noisy Student Is All You Need.* BirdCLEF+ 2025 Kaggle Discussion.
3. Cañas et al. *Overview of BirdCLEF+ 2025.* CLEF 2025.
4. Park et al. *SpecAugment.* Interspeech 2019.
5. Radenović et al. *Fine-tuning CNN Image Retrieval with No Human Annotation.* TPAMI 2019. (GeM pooling)
