#!/usr/bin/env bash
# run_final_eval.sh — End-to-end final-evaluation pipeline.
# Run this AFTER fold 1 + fold 2 training are both complete.
# Produces the final ensemble val/auc number + report figures.
set -euo pipefail

CB=/workspace/major_project/birdclef_combined
TRAIN_CSV=/workspace/birdclef-2025/train.csv
AUDIO_ROOT=/workspace/birdclef-2025/train_audio

# v3 fold 0 ckpts (existing)
F0_CLEAN=$CB/logdir/_v3_eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/fold_0/checkpoints/best-epoch07-auc0.7756.ckpt
F0_FILM=$CB/logdir/_v3_eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/fold_0/checkpoints/best-epoch06-auc0.7841.ckpt

# ESC-50 BG-trained peak (saved from aborted noise-augmented run)
ESC50_BG=/workspace/esc50_bg_peak.ckpt

# Newly trained fold 1 (clean, no BG) — pick best ckpt automatically
F1_DIR=$CB/logdir/eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/fold_1/checkpoints
F1_BEST=$(ls $F1_DIR/best-epoch*-auc*.ckpt 2>/dev/null | sort -t auc -k2 -n -r | head -1 || true)

echo "=== checkpoints to ensemble ==="
echo "  F0_CLEAN:  $F0_CLEAN  $([ -f $F0_CLEAN ] && echo '[OK]' || echo '[MISSING]')"
echo "  F0_FILM:   $F0_FILM  $([ -f $F0_FILM ] && echo '[OK]' || echo '[MISSING]')"
echo "  ESC50_BG:  $ESC50_BG  $([ -f $ESC50_BG ] && echo '[OK]' || echo '[MISSING]')"
echo "  F1_BEST:   ${F1_BEST:-NOT FOUND}"
echo

CKPTS=""
for c in "$F0_CLEAN" "$F0_FILM" "$ESC50_BG" "$F1_BEST"; do
    [[ -f "$c" ]] && CKPTS="$CKPTS $c"
done

if [[ -z "$CKPTS" ]]; then
    echo "[ERROR] no checkpoints found. Run training first."
    exit 1
fi

# --- Step 1: ensemble evaluation on val set ---
echo
echo "=== STEP 1: ensemble val/auc measurement ==="
python $CB/scripts/evaluate_ensemble.py \
    --ckpts $CKPTS \
    --train_csv $TRAIN_CSV \
    --audio_root $AUDIO_ROOT \
    --fold 0 \
    --output_csv /workspace/ensemble_per_class_auc.csv 2>&1 | tee /workspace/final_eval.log

# --- Step 2: generate report figures ---
echo
echo "=== STEP 2: generate report figures ==="
ENSEMBLE_AUC=$(grep "Macro-averaged ROC AUC" /workspace/final_eval.log | awk '{print $NF}' | head -1)
echo "  parsed ensemble AUC: $ENSEMBLE_AUC"

python $CB/scripts/make_report_figures.py \
    --logs /workspace/eca_stage1_fold1.log \
    --per_class_csv /workspace/ensemble_per_class_auc.csv \
    --train_csv $TRAIN_CSV \
    --ensemble_auc $ENSEMBLE_AUC \
    --single_aucs fold0_clean=0.7756 fold0_film=0.7841 esc50_bg=0.6814 \
    --output_dir /workspace/report_figures

# --- Step 3: sanity-test predict CLI on a known file ---
echo
echo "=== STEP 3: predict CLI smoke test ==="
SAMPLE_FILE=$(ls $AUDIO_ROOT/*/*.ogg 2>/dev/null | head -1)
if [[ -n "$SAMPLE_FILE" ]]; then
    python $CB/scripts/predict_cli.py --audio "$SAMPLE_FILE" --ckpts $CKPTS --top_k 3
else
    echo "  [SKIP] no sample audio found"
fi

echo
echo "=== DONE ==="
echo "Ensemble AUC: $ENSEMBLE_AUC"
echo "Per-class AUC: /workspace/ensemble_per_class_auc.csv"
echo "Figures:      /workspace/report_figures/"
echo "Full log:     /workspace/final_eval.log"
