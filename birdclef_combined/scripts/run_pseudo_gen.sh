#!/usr/bin/env bash
# Pseudo-label generation wrapper. Usage:
#   bash scripts/run_pseudo_gen.sh <iteration> <power_alpha>
# Example:
#   bash scripts/run_pseudo_gen.sh 1 0.6
set -euo pipefail

ITERATION="${1:?iteration arg required (1, 2, or 3)}"
POWER_ALPHA="${2:?power_alpha arg required (e.g. 0.6, 0.5, 0.4)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$SCRIPT_DIR"

MANIFEST=$(ls logdir/eca_nfnet_l0_*/checkpoint_manifest.json | head -1)
SCD=/workspace/birdclef-2025/train_soundscapes
OUT=/workspace/pseudo_labels
LOG=/workspace/pseudo_iter${ITERATION}_gen.log

mkdir -p "$OUT"

echo "MANIFEST     = $MANIFEST"
echo "SOUNDSCAPES  = $SCD"
echo "OUTPUT_DIR   = $OUT"
echo "ITERATION    = $ITERATION"
echo "POWER_ALPHA  = $POWER_ALPHA"
echo "LOG          = $LOG"
echo

python scripts/generate_pseudo_labels.py \
    --checkpoint_manifest "$MANIFEST" \
    --soundscape_dir "$SCD" \
    --output_dir "$OUT" \
    --iteration "$ITERATION" \
    --power_alpha "$POWER_ALPHA" \
    --backbone eca_nfnet_l0 \
    --n_classes 206 \
    --use_noise_conditioning \
    --noise_dim 128 \
    2>&1 | tee "$LOG"
