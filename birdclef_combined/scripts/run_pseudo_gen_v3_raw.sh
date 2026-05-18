#!/usr/bin/env bash
# run_pseudo_gen_v3_raw.sh — Re-run pseudo-label generation against the v3
# checkpoint (renamed to _v3_*) with very low thresholds. The resulting parquet
# stores almost-raw model outputs so we can post-process with rare-class
# masking + real thresholds in a separate script.
set -euo pipefail

CB=/workspace/major_project/birdclef_combined
LOGDIR=$CB/logdir
V3_DIR=$LOGDIR/_v3_eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005
MANIFEST=$V3_DIR/checkpoint_manifest.json
SCD=/workspace/birdclef-2025/train_soundscapes
OUT=/workspace/pseudo_labels
LOG=/workspace/pseudo_iter1_gen_v3raw.log

if [[ ! -f "$MANIFEST" ]]; then
    echo "[ERROR] v3 manifest not found at: $MANIFEST"
    echo "Check that _v3_* logdir exists."
    exit 1
fi

mkdir -p "$OUT"
cd "$CB"

echo "MANIFEST     = $MANIFEST"
echo "SOUNDSCAPES  = $SCD"
echo "OUTPUT_DIR   = $OUT"
echo "ITERATION    = 1"
echo "max_prob_threshold   = 0.0  (keep everything)"
echo "class_prob_threshold = 0.05 (keep weaker signal)"
echo "LOG          = $LOG"
echo

# Note: writes pseudo_labels_iter1_full.parquet — same path as before. The
# post-process script reads this and writes pseudo_labels_iter1_v3masked.parquet.
python scripts/generate_pseudo_labels.py \
    --checkpoint_manifest "$MANIFEST" \
    --soundscape_dir "$SCD" \
    --output_dir "$OUT" \
    --iteration 1 \
    --power_alpha 0.6 \
    --max_prob_threshold 0.0 \
    --class_prob_threshold 0.05 \
    --backbone eca_nfnet_l0 \
    --n_classes 206 \
    --no_noise_conditioning \
    2>&1 | tee "$LOG"
