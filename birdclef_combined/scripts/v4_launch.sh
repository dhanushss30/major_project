#!/usr/bin/env bash
# v4_launch.sh — Launch Stage 1 v4 training in a detached tmux session.
# Survives SSH disconnects. Logs to /workspace/eca_stage1_fold0_v4.log.
set -euo pipefail

CB=/workspace/major_project/birdclef_combined
LOG=/workspace/eca_stage1_fold0_v4.log
SESSION=v4_train

cd "$CB"

# Refuse to start if a previous v4 session is already running.
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[ERROR] tmux session '$SESSION' already exists. Inspect with:"
    echo "    tmux attach -t $SESSION"
    echo "Or kill it: tmux kill-session -t $SESSION"
    exit 1
fi

# Refuse if a v4 log already exists with content — guards against re-running over
# a previous run's log file.
if [[ -s "$LOG" ]]; then
    echo "[WARN] $LOG already has content ($(wc -l < "$LOG") lines)."
    echo "       Move it aside if you intended to start fresh:"
    echo "           mv $LOG ${LOG}.prev"
    exit 1
fi

# Wipe any stale log and launch.
: > "$LOG"
tmux new -d -s "$SESSION" \
    "cd $CB && python scripts/main_train.py configs/train/selected_eca.py --fold 0 2>&1 | tee $LOG"

sleep 2
echo "[OK] launched tmux session '$SESSION'"
tmux ls
echo
echo "Log: $LOG"
echo "Watch live:  tail -f $LOG"
echo "Attach:      tmux attach -t $SESSION   (Ctrl-b d to detach)"
