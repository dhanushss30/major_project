#!/usr/bin/env bash
# v4_preflight.sh — Rename v3 logdir, verify uploaded files, and run the
# config-validation smoke test in one go. Avoids long-line paste issues.
set -euo pipefail

CB=/workspace/major_project/birdclef_combined
LOGDIR=$CB/logdir
V3_NAME=eca_nfnet_l0_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005
V3_RENAMED=_v3_$V3_NAME

cd "$CB"

# --- Step 1: rename v3 logdir if not already renamed ---
if [[ -d "$LOGDIR/$V3_NAME" ]]; then
    mv "$LOGDIR/$V3_NAME" "$LOGDIR/$V3_RENAMED"
    echo "[OK] renamed v3 logdir: $V3_NAME -> $V3_RENAMED"
elif [[ -d "$LOGDIR/$V3_RENAMED" ]]; then
    echo "[SKIP] v3 logdir already renamed"
else
    echo "[WARN] no v3 logdir found"
fi

echo
echo "=== logdir/ contents ==="
ls -la "$LOGDIR/"

echo
echo "=== uploaded file timestamps ==="
ls -la "$CB/scripts/main_train.py" "$CB/configs/train/selected_eca.py"

echo
echo "=== config validation ==="
python - <<'PYEOF'
from pathlib import Path
import sys
sys.path.insert(0, "/workspace/major_project/birdclef_combined")
sys.path.insert(0, "/workspace/major_project/birdclef_combined/configs/train")
from selected_eca import config as cfg
import pandas as pd

bg_dir  = cfg.get("bg_soundscape_dir")
esc_dir = cfg.get("bg_esc50_dir")
ms      = cfg.get("min_class_samples_to_train")
mp      = cfg.get("bg_max_paths")

bg  = sorted(str(p) for p in Path(bg_dir).glob("*.ogg")) if bg_dir else []
esc = sorted(str(p) for p in Path(esc_dir).glob("*.wav")) if esc_dir else []
print(f"bg_soundscape_dir         = {bg_dir}")
print(f"  ogg files found         = {len(bg)}")
print(f"bg_esc50_dir              = {esc_dir}")
print(f"  wav files found         = {len(esc)}")
print(f"bg_max_paths              = {mp}")
print(f"min_class_samples_to_train = {ms}")
print(f"use_noise_conditioning    = {cfg.get('use_noise_conditioning')}")
print(f"lr                        = {cfg.get('lr')}")
print(f"epochs                    = {cfg.get('epochs')}")
print(f"p_negative                = {cfg.get('p_negative')}")

df = pd.read_csv(Path(cfg["data_root"]) / cfg["metadata_file"])
counts = df["primary_label"].value_counts()
rare   = counts[counts < ms].index.tolist()
df_f   = df[~df["primary_label"].isin(rare)]
print()
print(f"train.csv total rows        = {len(df):,}")
print(f"distinct classes            = {df['primary_label'].nunique()}")
print(f"rare classes (< {ms} samples) = {len(rare)}")
print(f"rows dropped by filter      = {len(df) - len(df_f):,}")
print(f"classes remaining in train  = {df_f['primary_label'].nunique()}")
PYEOF
