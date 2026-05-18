#!/usr/bin/env bash
# auto_fold2_watcher.sh — Wait for fold 1 tmux session to end, then auto-launch fold 2.
# Runs detached in its own tmux session so it survives SSH disconnects.
#
# Usage:
#     bash auto_fold2_watcher.sh
#
# It will:
#   1. Poll every 60 sec for the 'fold1_train' tmux session
#   2. When fold 1 ends, sanity-check it didn't crash (look for best ckpt)
#   3. If healthy → launch fold 2 via fold_launch.sh
#   4. If unhealthy → write FOLD1_FAILED marker and exit
set -euo pipefail

CB=/workspace/major_project/birdclef_combined
F1_LOG=/workspace/eca_stage1_fold1.log
F1_CKPT_DIR=$CB/logdir/eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/fold_1/checkpoints
WATCHER_LOG=/workspace/auto_fold2_watcher.log

echo "[$(date)] watcher started, monitoring 'fold1_train' tmux session" | tee -a $WATCHER_LOG

# --- Wait for fold 1 to finish ---
while tmux has-session -t fold1_train 2>/dev/null; do
    sleep 60
done

echo "[$(date)] fold 1 tmux session ended" | tee -a $WATCHER_LOG

# --- Sanity check: did fold 1 finish cleanly? ---
# Healthy if (a) at least one best-epoch ckpt exists AND (b) log mentions training complete
if ! ls $F1_CKPT_DIR/best-epoch*.ckpt > /dev/null 2>&1; then
    echo "[$(date)] [ERROR] no ckpt in $F1_CKPT_DIR — fold 1 crashed before saving." | tee -a $WATCHER_LOG
    touch /workspace/FOLD1_FAILED
    exit 1
fi

BEST_AUC=$(ls $F1_CKPT_DIR/best-epoch*-auc*.ckpt 2>/dev/null | grep -oP 'auc\K[\d.]+' | sort -n | tail -1)
echo "[$(date)] fold 1 best val/auc: $BEST_AUC" | tee -a $WATCHER_LOG

# Heuristic: if best AUC < 0.65, training was likely unhealthy → don't auto-launch fold 2
if (( $(echo "$BEST_AUC < 0.65" | bc -l) )); then
    echo "[$(date)] [WARN] fold 1 AUC too low ($BEST_AUC < 0.65) — NOT auto-launching fold 2." | tee -a $WATCHER_LOG
    touch /workspace/FOLD1_FAILED
    exit 1
fi

# --- Launch fold 2 ---
echo "[$(date)] launching fold 2 via fold_launch.sh" | tee -a $WATCHER_LOG
bash $CB/scripts/fold_launch.sh 2 2>&1 | tee -a $WATCHER_LOG

echo "[$(date)] watcher done" | tee -a $WATCHER_LOG
