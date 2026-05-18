#!/usr/bin/env bash
# fold_launch.sh — Launch additional v3 fold (1, 2, 3, or 4) for ensemble.
# Uses selected_eca.py config with BG mixing DISABLED (matches v3 fold 0).
#
# Usage:
#     bash fold_launch.sh 1   # launches fold 1
#     bash fold_launch.sh 2   # launches fold 2
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <fold_number>"
    echo "  fold_number: 1, 2, 3, or 4"
    exit 1
fi

FOLD=$1
if ! [[ "$FOLD" =~ ^[1-4]$ ]]; then
    echo "[ERROR] fold must be 1, 2, 3, or 4 (got: $FOLD)"
    exit 1
fi

CB=/workspace/major_project/birdclef_combined
LOG=/workspace/eca_stage1_fold${FOLD}.log
SESSION=fold${FOLD}_train

cd "$CB"

# --- Step 1: verify BG mixing is disabled ---
echo "=== config check ==="
python - <<PYEOF
import sys
sys.path.insert(0, "/workspace/major_project/birdclef_combined")
sys.path.insert(0, "/workspace/major_project/birdclef_combined/configs/train")
from selected_eca import config as cfg
print(f"exp_name              = {cfg['exp_name']}")
print(f"bg_soundscape_dir     = {cfg['bg_soundscape_dir']}")
print(f"bg_esc50_dir          = {cfg['bg_esc50_dir']}")
print(f"min_class_samples     = {cfg.get('min_class_samples_to_train')}")
print(f"pretrained            = {cfg['pretrained']}")
print(f"epochs                = {cfg['epochs']}")
print(f"lr                    = {cfg['lr']}")
print(f"limit_train_batches   = {cfg.get('limit_train_batches')}")
print(f"stage                 = {cfg['stage']}")
print(f"pseudo_labels_path    = {cfg['pseudo_labels_path']}")
assert cfg['bg_soundscape_dir'] is None, \
    "BG soundscape mixing must be disabled (None) — would cause v4-style collapse"
# bg_esc50_dir is allowed (safe noise augmentation, no labeling contradiction)
assert cfg['stage'] == 'supervised', "must be supervised stage (no pseudo)"
assert cfg['pseudo_labels_path'] is None, "pseudo_labels_path must be None for v3-replica training"
if cfg['bg_esc50_dir'] is not None:
    print(f"[OK] ESC-50 BG mixing enabled (safe noise augmentation for real-world robustness)")
else:
    print(f"[OK] no BG mixing (matches v3 exactly)")
print("[OK] config matches v3 setup")
PYEOF

# --- Step 2: guard against double launch ---
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo
    echo "[ERROR] tmux session '$SESSION' already exists. Inspect with:"
    echo "    tmux attach -t $SESSION"
    echo "Or kill it: tmux kill-session -t $SESSION"
    exit 1
fi

if [[ -s "$LOG" ]]; then
    echo
    echo "[WARN] $LOG already has content ($(wc -l < "$LOG") lines)."
    echo "       Move aside if intentional: mv $LOG ${LOG}.prev"
    exit 1
fi

# --- Step 3: verify v3 fold_0 dir exists (this is what we're ensembling with) ---
V3_F0=$CB/logdir/_v3_eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/fold_0
if [[ ! -d "$V3_F0" ]]; then
    echo "[WARN] v3 fold_0 not found at expected path: $V3_F0"
    echo "       Ensemble will only use newly-trained folds."
fi

# --- Step 4: launch ---
echo
: > "$LOG"
tmux new -d -s "$SESSION" \
    "cd $CB && python scripts/main_train.py configs/train/selected_eca.py --fold $FOLD 2>&1 | tee $LOG"

sleep 2
echo "[OK] launched tmux session '$SESSION' (fold $FOLD)"
tmux ls
echo
echo "Log: $LOG"
echo "Watch live:  tail -f $LOG"
echo "Attach:      tmux attach -t $SESSION   (Ctrl-b d to detach)"
echo
echo "Expected: ~3.5h training, val/auc should reach ~0.74-0.78 by epoch 7-12."
