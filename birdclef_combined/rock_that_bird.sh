#!/usr/bin/env bash
# =============================================================================
# rock_that_bird.sh — BirdCLEF+ 2025 Combined Solution End-to-End Pipeline
# =============================================================================
# Reproduces the combined 1st + 2nd place solution:
#   2nd place: Private AUC 0.928, Public 0.925
#   1st place additions: ~+0.003 AUC (TTA, specialists, power transform, SED)
#
# Usage: bash rock_that_bird.sh "0"         # GPU 0
#        bash rock_that_bird.sh "0,1"       # Multi-GPU (not yet supported)
#
# Prerequisites:
#   - eval $(poetry env activate)
#   - KAGGLE_USERNAME and KAGGLE_KEY set in environment
#   - ≥600GB disk, ≥10GB VRAM GPU, 126GB RAM
#
# Paper reference: Sydorskyi & Gonçalves, CLEF 2025 Working Notes
# =============================================================================

set -euo pipefail
GPU="${1:-0}"

DATA_ROOT="/data/birdclef-2025"
HDF5_ROOT="/data/birdclef-2025-hdf5"
PSEUDO_ROOT="/data/birdclef-2025-pseudo"
LOGDIR="logdir"

echo "============================================================"
echo " BirdCLEF+ 2025 Combined 1st & 2nd Place Solution"
echo " GPU: $GPU | Data: $DATA_ROOT"
echo "============================================================"

# ── Step 0: Download data ────────────────────────────────────────────────
echo ""
echo "[STEP 0] Downloading competition data..."
mkdir -p "$DATA_ROOT"

# Main competition data
kaggle competitions download -c birdclef-2025 -p "$DATA_ROOT" --quiet
cd "$DATA_ROOT" && unzip -qo birdclef-2025.zip && cd -

# 2nd place additional datasets (Kaggle datasets)
kaggle datasets download -d vladimirsydor/bird-clef-2025-add-data \
    -p "$DATA_ROOT/add_data" --quiet
kaggle datasets download -d vladimirsydor/bird-clef-2025-all-pretrained-models \
    -p "$DATA_ROOT/pretrained" --quiet

echo "[STEP 0] ✓ Data downloaded"

# ── Step 1: Pre-compute HDF5 ────────────────────────────────────────────
# Paper Appendix B: "audio files were converted to h5py format"
echo ""
echo "[STEP 1] Pre-computing HDF5 features (MP3→use_torchaudio)..."
mkdir -p "$HDF5_ROOT"

# Training audio (.ogg and .mp3 mixed — must use torchaudio for MP3)
python scripts/precompute_features.py \
    "$DATA_ROOT/train_audio" \
    "$HDF5_ROOT/train_audio" \
    --n_cores 8 \
    --use_torchaudio \
    --sample_rate 32000

# Additional data
if [ -d "$DATA_ROOT/add_data/train_audio" ]; then
    python scripts/precompute_features.py \
        "$DATA_ROOT/add_data/train_audio" \
        "$HDF5_ROOT/add_data" \
        --n_cores 8 \
        --use_torchaudio \
        --sample_rate 32000
fi

echo "[STEP 1] ✓ HDF5 features precomputed"

# ── Step 2: Stage 1 — Supervised Training ───────────────────────────────
# Paper: 50 epochs, batch=64, NFNet RAdam lr=1e-3, EffNetV2 AdamW lr=1e-4
echo ""
echo "[STEP 2] Stage 1: Supervised fine-tuning (clean labels)..."

# ECA-NFNet-L0 × 5 folds (~5.7h on RTX 4090)
echo "  [2a] Training ECA-NFNet-L0 (γ=-0.5 SqrtBalancing, RAdam 1e-3)..."
CUDA_VISIBLE_DEVICES="$GPU" python scripts/main_train.py \
    configs/train/selected_eca.py

# EfficientNetV2-S × 5 folds
echo "  [2b] Training EfficientNetV2-S-in21k (γ=-1 EqualBalancing, AdamW 1e-4)..."
CUDA_VISIBLE_DEVICES="$GPU" python scripts/main_train.py \
    configs/train/selected_ebs.py

echo "[STEP 2] ✓ Stage 1 complete"

# ── Step 3: Insects & Amphibians Specialist (1st Place) ─────────────────
echo ""
echo "[STEP 3] Training insects/amphibians specialist (1st place +0.003 AUC)..."
CUDA_VISIBLE_DEVICES="$GPU" python scripts/main_train.py \
    configs/train/insects_amphibians.py
echo "[STEP 3] ✓ Specialist trained"

# ── Step 4: Generate Pseudo-Labels Iteration 1 ──────────────────────────
# Paper §5.4: max_prob≥0.5, class_prob≥0.1 → 19,405 chunks from 4,430 files
# 1st place: power_alpha=0.6
echo ""
echo "[STEP 4] Pseudo-labels iteration 1 (full mode, α=0.6)..."
mkdir -p "$PSEUDO_ROOT"

ECA_MANIFEST="$LOGDIR/$(ls "$LOGDIR" | grep eca_nfnet | head -1)/checkpoint_manifest.json"
python scripts/generate_pseudo_labels.py \
    --checkpoint_manifest "$ECA_MANIFEST" \
    --soundscape_dir      "$DATA_ROOT/unlabeled_soundscapes" \
    --output_dir          "$PSEUDO_ROOT" \
    --backbone            eca_nfnet_l0 \
    --n_classes           206 \
    --iteration           1 \
    --max_prob_threshold  0.5 \
    --class_prob_threshold 0.1 \
    --power_alpha         0.6 \
    --mode                full \
    --batch_size          64

echo "[STEP 4] ✓ Pseudo-labels iter 1: $(cat $PSEUDO_ROOT/pseudo_summary_iter1_full.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["n_selected"])') chunks"

# ── Step 5: Stage 2 — Noisy Student Iteration 1 ─────────────────────────
echo ""
echo "[STEP 5] Noisy Student Iteration 1..."

# Update pseudo_labels_path in config via env var (or edit config)
export PSEUDO_PATH_ITER1="$PSEUDO_ROOT/pseudo_labels_iter1_full.parquet"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/main_train.py \
    configs/train/pseudo_iter1_eca.py

echo "[STEP 5] ✓ Noisy Student iter 1 complete"

# ── Step 6: Generate Pseudo-Labels Iteration 2 ──────────────────────────
# Paper Table 4: 3,750 chunks from 1,483 files
# Both FULL and OOF modes (paper uses combination)
echo ""
echo "[STEP 6] Pseudo-labels iteration 2 (full + OOF modes, α=0.5)..."

STUDENT1_MANIFEST="$LOGDIR/$(ls "$LOGDIR" | grep PseudoF2PT05MT01P06I1 | head -1)/checkpoint_manifest.json"

python scripts/generate_pseudo_labels.py \
    --checkpoint_manifest "$STUDENT1_MANIFEST" \
    --soundscape_dir      "$DATA_ROOT/unlabeled_soundscapes" \
    --output_dir          "$PSEUDO_ROOT" \
    --backbone            eca_nfnet_l0 \
    --iteration           2 \
    --max_prob_threshold  0.5 \
    --class_prob_threshold 0.1 \
    --power_alpha         0.5 \
    --mode                full

python scripts/generate_pseudo_labels.py \
    --checkpoint_manifest "$STUDENT1_MANIFEST" \
    --soundscape_dir      "$DATA_ROOT/unlabeled_soundscapes" \
    --output_dir          "$PSEUDO_ROOT" \
    --backbone            eca_nfnet_l0 \
    --iteration           2 \
    --max_prob_threshold  0.5 \
    --class_prob_threshold 0.1 \
    --power_alpha         0.5 \
    --mode                oof

# Merge full + OOF (paper uses combined dataset)
python3 -c "
import pandas as pd
full = pd.read_parquet('$PSEUDO_ROOT/pseudo_labels_iter2_full.parquet')
oof  = pd.read_parquet('$PSEUDO_ROOT/pseudo_labels_iter2_oof.parquet')
combined = pd.concat([full, oof]).drop_duplicates(subset=['filename','start_sec'])
combined = combined.sort_values('max_confidence', ascending=False)
combined.to_parquet('$PSEUDO_ROOT/pseudo_labels_iter2_combined.parquet', index=False)
print(f'Combined iter2: {len(combined):,} chunks')
"

echo "[STEP 6] ✓ Pseudo-labels iter 2 generated"

# ── Step 7: Stage 2 — Noisy Student Iteration 2 ─────────────────────────
echo ""
echo "[STEP 7] Noisy Student Iteration 2 (EfficientNetV2-S)..."
CUDA_VISIBLE_DEVICES="$GPU" python scripts/main_train.py \
    configs/train/pseudo_iter2_ebs.py
echo "[STEP 7] ✓ Noisy Student iter 2 complete"

# ── Step 8: Generate Pseudo-Labels Iteration 3 ──────────────────────────
# Paper Table 4: 4,108 chunks from 1,437 files, cumulative pseudo pool
echo ""
echo "[STEP 8] Pseudo-labels iteration 3 (α=0.4, most aggressive)..."

STUDENT2_MANIFEST="$LOGDIR/$(ls "$LOGDIR" | grep PseudoF2PT05MT01P05I2 | head -1)/checkpoint_manifest.json"

python scripts/generate_pseudo_labels.py \
    --checkpoint_manifest "$STUDENT2_MANIFEST" \
    --soundscape_dir      "$DATA_ROOT/unlabeled_soundscapes" \
    --output_dir          "$PSEUDO_ROOT" \
    --backbone            tf_efficientnetv2_s \
    --iteration           3 \
    --max_prob_threshold  0.5 \
    --class_prob_threshold 0.1 \
    --power_alpha         0.4 \
    --mode                full

# Merge all iterations (cumulative pseudo pool)
python3 -c "
import pandas as pd
iter1 = pd.read_parquet('$PSEUDO_ROOT/pseudo_labels_iter1_full.parquet')
iter2 = pd.read_parquet('$PSEUDO_ROOT/pseudo_labels_iter2_combined.parquet')
iter3 = pd.read_parquet('$PSEUDO_ROOT/pseudo_labels_iter3_full.parquet')
merged = pd.concat([iter1, iter2, iter3]).drop_duplicates(subset=['filename','start_sec'])
merged = merged.sort_values('max_confidence', ascending=False)
merged.to_parquet('$PSEUDO_ROOT/pseudo_labels_all_iters_merged.parquet', index=False)
print(f'All-iter merged: {len(merged):,} chunks total')
"

echo "[STEP 8] ✓ Pseudo-labels iter 3 complete"

# ── Step 9: Stage 2 — Final Noisy Student (ECA, all pseudo) ─────────────
echo ""
echo "[STEP 9] Final Noisy Student (ECA-NFNet, all 3 iterations, α=0.4)..."
CUDA_VISIBLE_DEVICES="$GPU" python scripts/main_train.py \
    configs/train/pseudo_iter2_ebs.py   # Uses config_iter3 from this file
echo "[STEP 9] ✓ Final noisy student complete"

# ── Step 10: Inference + Post-processing + ONNX/OpenVINO ────────────────
echo ""
echo "[STEP 10] Ensemble inference + TopN postprocessing + ONNX/OpenVINO export..."
CUDA_VISIBLE_DEVICES="$GPU" python scripts/main_inference.py \
    configs/inference/ensemble_final.py \
    --mode all \
    --soundscape_dir "$DATA_ROOT/test_soundscapes" \
    --output submission_final.csv

echo "[STEP 10] ✓ Inference complete"

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ✓  PIPELINE COMPLETE"
echo ""
echo "   Submission CSV : submission_final.csv"
echo "   Model logdir   : $LOGDIR/"
echo "   ONNX model     : $LOGDIR/.../onnx_ensemble/"
echo "   OpenVINO FP16  : $LOGDIR/.../onnx_ensemble_openvino_fp16/"
echo ""
echo "   Expected performance (from paper):"
echo "     2nd place alone : Public 0.925 / Private 0.928"
echo "     With 1st place additions (TTA + specialist + SoftAUC):"
echo "               approx: Public 0.928+ / Private 0.930+"
echo "============================================================"
