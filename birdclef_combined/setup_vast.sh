#!/usr/bin/env bash
# =============================================================================
# setup_vast.sh — One-Command Vast.ai Instance Setup for BirdCLEF 2025
# =============================================================================
# Run this ONCE after SSH-ing into your Vast.ai instance:
#   bash setup_vast.sh <kaggle_username> <kaggle_key>
#
# What it does:
#   1. Installs all Python dependencies
#   2. Downloads BirdCLEF 2025 (main competition data)
#   3. Downloads BirdCLEF 2021-2024 (prior years, filtered to 2025 species)
#   4. Merges all years into one unified training set (~100K recordings)
#   5. Verifies setup + prints training command
#
# Requirements:
#   - Vast.ai instance with PyTorch image (CUDA 12.1+), >= 150 GB disk
#   - Kaggle API key (kaggle.com > Account > API > Create New Token)
#   - Accepted BirdCLEF 2025 rules at kaggle.com/competitions/birdclef-2025
#   - Prior years (2021-2024) rules auto-accepted (closed competitions)
# =============================================================================

set -euo pipefail

KAGGLE_USERNAME="${1:-}"
KAGGLE_KEY="${2:-}"

if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "Usage: bash setup_vast.sh <kaggle_username> <kaggle_key>"
    echo ""
    echo "Get your Kaggle key: kaggle.com > Settings > API > Create New Token"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_2025="/workspace/birdclef-2025"
PRIOR_DIR="/workspace/prior_years"
MERGED_DIR="/workspace/birdclef-merged"
LOGDIR="/workspace/logdir"

echo "================================================================"
echo " BirdCLEF 2025 — Vast.ai Full Setup"
echo " Project: $PROJECT_DIR"
echo " Kaggle:  $KAGGLE_USERNAME"
echo "================================================================"

# ── [1/6] System packages ─────────────────────────────────────────────────────
echo ""
echo "[1/6] System packages..."
apt-get update -qq 2>/dev/null
apt-get install -y -qq libsndfile1 ffmpeg unzip curl git p7zip-full 2>/dev/null || true
echo "  Done."

# ── [2/6] Python packages ─────────────────────────────────────────────────────
echo ""
echo "[2/6] Python packages..."
REQUIREMENTS="$PROJECT_DIR/../requirements.txt"
pip install --quiet --upgrade pip

CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | head -1 || echo "12.1")
echo "  CUDA: $CUDA_VER"

if [[ "$CUDA_VER" == 12* ]]; then
    TORCH_IDX="https://download.pytorch.org/whl/cu121"
elif [[ "$CUDA_VER" == 11* ]]; then
    TORCH_IDX="https://download.pytorch.org/whl/cu118"
else
    TORCH_IDX="https://download.pytorch.org/whl/cu121"
fi

pip install --quiet torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2 \
    --extra-index-url "$TORCH_IDX"

grep -v "^torch" "$REQUIREMENTS" | grep -v "^#" | grep -v "^$" | \
    xargs pip install --quiet

echo "  Done."

# ── [3/6] Kaggle API config ───────────────────────────────────────────────────
echo ""
echo "[3/6] Kaggle API setup..."
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << EOF
{"username":"$KAGGLE_USERNAME","key":"$KAGGLE_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json
echo "  Done."

# ── [4/6] Download BirdCLEF 2025 ─────────────────────────────────────────────
echo ""
echo "[4/6] Downloading BirdCLEF 2025 (~12 GB)..."
mkdir -p "$DATA_2025"

if [ ! -f "$DATA_2025/train.csv" ]; then
    kaggle competitions download -c birdclef-2025 -p "$DATA_2025" --quiet
    echo "  Extracting 2025 data..."
    cd "$DATA_2025" && unzip -qo birdclef-2025.zip && cd -
else
    echo "  Already downloaded."
fi

N_2025=$(find "$DATA_2025/train_audio" -name "*.ogg" 2>/dev/null | wc -l)
echo "  BirdCLEF 2025: $N_2025 recordings"

# ── [5/6] Download + merge prior years (2021-2024) ───────────────────────────
echo ""
echo "[5/6] Downloading prior years (2021-2024) — this may take 30-60 min..."
echo "      Prior year downloads may fail if you haven't accepted their rules."
echo "      Any year that fails is skipped (2025 data is sufficient to train)."

mkdir -p "$PRIOR_DIR"

download_year() {
    local YEAR=$1
    local COMP="birdclef-$YEAR"
    local YDIR="$PRIOR_DIR/$YEAR"
    mkdir -p "$YDIR"

    # Check if already downloaded
    if ls "$YDIR"/*.csv 2>/dev/null | head -1 | grep -q .; then
        echo "  $YEAR: already downloaded"
        return 0
    fi

    echo "  Downloading $YEAR..."
    if kaggle competitions download -c "$COMP" -p "$YDIR" --quiet 2>/dev/null; then
        # Extract whatever zip was downloaded
        for ZF in "$YDIR"/*.zip; do
            [ -f "$ZF" ] && unzip -qo "$ZF" -d "$YDIR" && rm -f "$ZF" || true
        done
        echo "  $YEAR: OK"
    else
        echo "  $YEAR: SKIPPED (download failed — rules not accepted or unavailable)"
    fi
}

download_year 2024
download_year 2023
download_year 2022
download_year 2021

# ── Run merge script ───────────────────────────────────────────────────────────
echo ""
echo "  Merging all years into unified dataset..."
cd "$PROJECT_DIR"
python scripts/merge_datasets.py \
    --data_root_2025 "$DATA_2025" \
    --download_dir   "$PRIOR_DIR" \
    --output_dir     "$MERGED_DIR" \
    --skip_download \
    --years 2021 2022 2023 2024 \
    2>&1 | tail -30

if [ -f "$MERGED_DIR/train_merged.csv" ]; then
    N_MERGED=$(python -c "import pandas as pd; df=pd.read_csv('$MERGED_DIR/train_merged.csv'); print(len(df))" 2>/dev/null || echo "?")
    echo "  Merged dataset: $N_MERGED total recordings"
    TRAIN_DATA="$MERGED_DIR"
    METADATA_FILE="train_merged.csv"
else
    echo "  Merge incomplete — using BirdCLEF 2025 only."
    TRAIN_DATA="$DATA_2025"
    METADATA_FILE="train.csv"
fi

echo "  Done."

# ── [6/6] Smoke test ─────────────────────────────────────────────────────────
echo ""
echo "[6/6] Smoke test..."
cd "$PROJECT_DIR"
python -c "
import torch, timm, lightning, librosa, h5py
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\"}')
print(f'  timm     : {timm.__version__}')
print(f'  lightning: {lightning.__version__}')
print(f'  librosa  : {librosa.__version__}')
print('  All OK.')
"
echo "  Done."

# ── Final instructions ────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " SETUP COMPLETE — Start training with:"
echo ""
echo "   cd $PROJECT_DIR"
echo ""
echo "   # Recommended: full automated pipeline with merged dataset"
echo "   nohup python scripts/run_full_pipeline.py \\"
echo "       --data_root   $TRAIN_DATA \\"
echo "       --metadata_file $METADATA_FILE \\"
echo "       --logdir      $LOGDIR \\"
echo "       --gpu 0 \\"
echo "       > /workspace/pipeline.log 2>&1 &"
echo ""
echo "   tail -f /workspace/pipeline.log"
echo ""
echo "   # To resume after crash:"
echo "   # add --resume to the command above"
echo ""
echo " Dataset used:    $TRAIN_DATA"
echo " Metadata file:   $METADATA_FILE"
echo " Expected time:   ~30-45 hours (RTX 3090/4090)"
echo " Expected AUC:    ~0.96-0.97"
echo "================================================================"

# Save paths for convenience
cat > /workspace/train_paths.env << EOF
TRAIN_DATA=$TRAIN_DATA
METADATA_FILE=$METADATA_FILE
PROJECT_DIR=$PROJECT_DIR
LOGDIR=$LOGDIR
EOF
echo " Saved paths to: /workspace/train_paths.env"
