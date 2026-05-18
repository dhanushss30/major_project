# Morning Workflow — BirdCLEF 2025 Final Delivery

When you wake up (anywhere between 02:30 IST and 07:30 IST), follow these steps in order.

---

## Step 0 — Reconnect SSH

```
ssh -p 24037 root@ssh3.vast.ai -L 8080:localhost:8080
```

The `-L 8080:localhost:8080` is important — it tunnels the Streamlit port to your laptop's browser.

---

## Step 1 — Verify fold 1 finished cleanly

```
tmux ls
```

Expected: `ssh_tmux` only (fold1_train session should have ended).

```
grep "val/auc" /workspace/eca_stage1_fold1.log | tail -10
```

Expected: final epochs around 0.65-0.70 (peak around epoch 8 = 0.6976).

```
ls /workspace/major_project/birdclef_combined/logdir/eca_*/fold_1/checkpoints/ | head
```

Expected: multiple `best-epoch*-auc*.ckpt` files. We want the one with highest AUC for the ensemble.

---

## Step 2 — Upload all 9 inference / helper scripts (from PowerShell, separately)

Open a fresh PowerShell window. Run each scp one at a time:

```
scp -P 24037 C:\Users\User\Downloads\birdclef_2025_combined_FINAL\birdclef_combined\scripts\ensemble_inference.py root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/scripts/
```
```
scp -P 24037 C:\Users\User\Downloads\birdclef_2025_combined_FINAL\birdclef_combined\scripts\predict_cli.py root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/scripts/
```
```
scp -P 24037 C:\Users\User\Downloads\birdclef_2025_combined_FINAL\birdclef_combined\scripts\evaluate_ensemble.py root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/scripts/
```
```
scp -P 24037 C:\Users\User\Downloads\birdclef_2025_combined_FINAL\birdclef_combined\scripts\streamlit_app.py root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/scripts/
```
```
scp -P 24037 C:\Users\User\Downloads\birdclef_2025_combined_FINAL\birdclef_combined\scripts\fastapi_backend.py root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/scripts/
```
```
scp -P 24037 C:\Users\User\Downloads\birdclef_2025_combined_FINAL\birdclef_combined\scripts\make_report_figures.py root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/scripts/
```
```
scp -P 24037 C:\Users\User\Downloads\birdclef_2025_combined_FINAL\birdclef_combined\scripts\noise_robust_preprocessing.py root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/scripts/
```
```
scp -P 24037 C:\Users\User\Downloads\birdclef_2025_combined_FINAL\birdclef_combined\scripts\find_demo_audio.py root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/scripts/
```
```
scp -P 24037 C:\Users\User\Downloads\birdclef_2025_combined_FINAL\birdclef_combined\scripts\run_final_eval.sh root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/scripts/
```

---

## Step 3 — Install streamlit + fastapi (in SSH)

```
pip install streamlit fastapi uvicorn python-multipart 2>&1 | tail -5
```

---

## Step 4 — Curate demo audio clips (in SSH)

```
python /workspace/major_project/birdclef_combined/scripts/find_demo_audio.py --train_csv /workspace/birdclef-2025/train.csv --audio_root /workspace/birdclef-2025/train_audio --output_dir /workspace/demo_audio --species_count 8 --min_samples 200 --clips_per_species 2
```

This copies ~16 demo clips into `/workspace/demo_audio/`. These are clean training-distribution audio of common species the model knows well.

---

## Step 5 — Run the final evaluation pipeline (in SSH)

```
bash /workspace/major_project/birdclef_combined/scripts/run_final_eval.sh
```

This runs **3 things in sequence (~30-40 min):**

1. Ensemble inference on fold 0 val (with TTA + noise preprocessing) → final val/auc number
2. Generates report figures (training curves, per-class AUC, ensemble comparison)
3. Smoke-tests `predict_cli.py` on one sample audio

**Watch for the line:**
```
=== ENSEMBLE FOLD 0 VAL AUC ===
  Macro-averaged ROC AUC: 0.XXXX
```

That's your final number. Paste it to me.

---

## Step 6 — Test predict CLI on demo audio (in SSH)

```
python /workspace/major_project/birdclef_combined/scripts/predict_cli.py --audio /workspace/demo_audio/$(ls /workspace/demo_audio/*.ogg 2>/dev/null | head -1 | xargs basename) --top_k 3
```

Should print per-5s top-3 species with confidence bars.

Try another with noise preprocessing OFF for comparison:
```
python /workspace/major_project/birdclef_combined/scripts/predict_cli.py --audio /workspace/demo_audio/$(ls /workspace/demo_audio/*.ogg | head -2 | tail -1 | xargs basename) --no_noise_preprocess --top_k 3
```

---

## Step 7 — Launch Streamlit UI (in SSH, runs in foreground)

```
streamlit run /workspace/major_project/birdclef_combined/scripts/streamlit_app.py --server.port 8080 --server.address 0.0.0.0 --server.headless true
```

Wait for "You can now view your Streamlit app in your browser" message.

Then in **your laptop's browser**: open `http://localhost:8080` (the SSH `-L 8080:localhost:8080` tunnels the port).

Test:
- Upload a demo audio clip → predictions appear
- Toggle "Spectral noise gating" on/off → see how predictions change
- Try the "Record live" tab → record yourself imitating a bird (won't work great, but proves the pipeline)

When done testing, press `Ctrl + C` to stop Streamlit.

---

## Step 8 — Backup everything to laptop (PowerShell)

In PowerShell, create a local backup directory:

```
mkdir C:\Users\User\Downloads\birdclef_2025_combined_FINAL\BACKUP_FINAL
```

Then download the artifacts (one scp at a time):

```
scp -P 24037 -r root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/logdir/_v3_*/fold_0/checkpoints/best-epoch07-auc0.7756.ckpt C:\Users\User\Downloads\birdclef_2025_combined_FINAL\BACKUP_FINAL\
```
```
scp -P 24037 -r root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/logdir/_v3_*/fold_0/checkpoints/best-epoch06-auc0.7841.ckpt C:\Users\User\Downloads\birdclef_2025_combined_FINAL\BACKUP_FINAL\
```
```
scp -P 24037 root@ssh3.vast.ai:/workspace/esc50_bg_peak.ckpt C:\Users\User\Downloads\birdclef_2025_combined_FINAL\BACKUP_FINAL\
```
```
scp -P 24037 -r "root@ssh3.vast.ai:/workspace/major_project/birdclef_combined/logdir/eca_*/fold_1/checkpoints" C:\Users\User\Downloads\birdclef_2025_combined_FINAL\BACKUP_FINAL\fold_1_ckpts
```
```
scp -P 24037 -r root@ssh3.vast.ai:/workspace/report_figures C:\Users\User\Downloads\birdclef_2025_combined_FINAL\BACKUP_FINAL\
```
```
scp -P 24037 root@ssh3.vast.ai:/workspace/ensemble_per_class_auc.csv C:\Users\User\Downloads\birdclef_2025_combined_FINAL\BACKUP_FINAL\
```
```
scp -P 24037 root@ssh3.vast.ai:/workspace/eca_stage1_fold1.log C:\Users\User\Downloads\birdclef_2025_combined_FINAL\BACKUP_FINAL\
```

Total ~1.5 GB.

---

## Step 9 — Destroy Vast instance (when fully done)

In your Vast.ai dashboard:
1. Find instance 36744037
2. Click "Destroy"
3. Confirm

Stops all billing.

---

## What you have at the end

- 4 trained checkpoints (v3 clean, v3 FiLM, ESC-50 BG, new fold 1)
- Ensemble val/auc number (computed in Step 5)
- Per-class AUC breakdown CSV
- Report figures (PNGs)
- All training logs
- Demo audio collection
- Complete inference/UI/API codebase

This is your full deliverable.

---

## If anything goes wrong, ping me

The most likely failure mode is the FiLM ckpt (`best-epoch06-auc0.7841.ckpt`) failing to load due to architecture mismatch. The script handles this gracefully (it'll skip that ckpt and use 3 instead of 4).

If `streamlit` fails to launch — make sure port 8080 isn't taken by something else.

If `predict_cli.py` errors on audio loading — check the audio file exists and is a valid wav/mp3/ogg.

I'll be here whenever you need.
