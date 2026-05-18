#!/usr/bin/env bash
# iter1_launch.sh — Launch noisy-student iter 1 training in detached tmux.
# - Validates v3 ckpt path
# - Validates pseudo parquet
# - Validates config loads
# - Launches in tmux session 'iter1_train'
set -euo pipefail

CB=/workspace/major_project/birdclef_combined
LOG=/workspace/eca_pseudo_iter1.log
SESSION=iter1_train
V3_CKPT=$CB/logdir/_v3_eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/fold_0/checkpoints/best-epoch07-auc0.7756.ckpt
PSEUDO=/workspace/pseudo_labels/pseudo_labels_iter1_v3capped.parquet

cd "$CB"

# --- Step 1: verify v3 ckpt exists ---
if [[ ! -f "$V3_CKPT" ]]; then
    echo "[ERROR] v3 ckpt not found at: $V3_CKPT"
    echo "Check the fold_0 logdir structure:"
    ls -la "$CB/logdir/_v3_eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005/fold_0/checkpoints/" 2>/dev/null || true
    exit 1
fi
echo "[OK] v3 ckpt found: $V3_CKPT"

# --- Step 2: verify pseudo parquet exists ---
if [[ ! -f "$PSEUDO" ]]; then
    echo "[ERROR] pseudo parquet not found at: $PSEUDO"
    exit 1
fi
echo "[OK] pseudo parquet found: $PSEUDO"
echo "    size: $(du -h $PSEUDO | cut -f1)"

# --- Step 3: validate config loads ---
echo
echo "=== config validation ==="
python - <<'PYEOF'
import sys
sys.path.insert(0, "/workspace/major_project/birdclef_combined")
sys.path.insert(0, "/workspace/major_project/birdclef_combined/configs/train")
from pseudo_iter1_eca import config as cfg
from pathlib import Path
import pandas as pd

print(f"exp_name              = {cfg['exp_name']}")
print(f"stage                 = {cfg['stage']}")
print(f"pseudo_labels_path    = {cfg['pseudo_labels_path']}")
print(f"  exists              = {Path(cfg['pseudo_labels_path']).exists()}")
print(f"labeled_fraction      = {cfg['labeled_fraction']}")
print(f"pretrained_path       = {cfg['pretrained_path']}")
print(f"  exists              = {Path(cfg['pretrained_path']).exists()}")
print(f"epochs                = {cfg['epochs']}")
print(f"lr                    = {cfg['lr']}")
print(f"auc_weight            = {cfg['auc_weight']}")
print(f"bg_soundscape_dir     = {cfg['bg_soundscape_dir']}")
print(f"bg_esc50_dir          = {cfg['bg_esc50_dir']}")
print(f"min_class_samples     = {cfg.get('min_class_samples_to_train')}")
print(f"p_negative            = {cfg.get('p_negative')}")

# Sanity-check pseudo parquet
df = pd.read_parquet(cfg['pseudo_labels_path'])
print()
print(f"pseudo parquet rows   = {len(df):,}")
print(f"pseudo columns        = {list(df.columns)}")
import numpy as np
P = np.stack(df['soft_labels'].values).astype(np.float32)
print(f"pseudo shape          = {P.shape}")
print(f"distinct top-1        = {len(np.unique(P.argmax(1)))}")
PYEOF

# --- Step 4: guard against double launch ---
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

# --- Step 5: launch in tmux ---
echo
: > "$LOG"
tmux new -d -s "$SESSION" \
    "cd $CB && python scripts/main_train.py configs/train/pseudo_iter1_eca.py --fold 0 2>&1 | tee $LOG"

sleep 2
echo "[OK] launched tmux session '$SESSION'"
tmux ls
echo
echo "Log: $LOG"
echo "Watch live:  tail -f $LOG"
echo "Attach:      tmux attach -t $SESSION   (Ctrl-b d to detach)"
