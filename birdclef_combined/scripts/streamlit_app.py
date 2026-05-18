"""streamlit_app.py — Web UI for BirdCLEF 2025 5-second species predictor.

Features:
  - Audio file upload (wav, mp3, ogg, flac)
  - In-browser live recording (via Streamlit's audio_input widget)
  - Per-5-second chunk predictions with confidence bars
  - "Not a bird" verdict via max-prob rejection threshold (adjustable)
  - Top-K species per chunk (adjustable)
  - Spectrogram visualization per chunk

Launch (from inside the Vast.ai instance, after fold 1 + 2 trained):
    streamlit run scripts/streamlit_app.py --server.port 8080 --server.address 0.0.0.0

Then on your laptop with the SSH tunnel up (-L 8080:localhost:8080),
open: http://localhost:8080
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_inference import ensemble_predict, load_ckpt
from predict_cli import DEFAULT_CKPTS, DEFAULT_TRAIN_CSV, load_label_names


# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title = "BirdCLEF 2025 — Species Predictor",
    page_icon  = "🐦",
    layout     = "wide",
)


# ============================================================================
# Cached resources
# ============================================================================
@st.cache_resource(show_spinner="Loading ensemble checkpoints…")
def load_models(ckpt_paths: tuple, device: str = "cuda"):
    loaded = []
    for c in ckpt_paths:
        if Path(c).exists():
            loaded.append(load_ckpt(c, device=device))
    return loaded


@st.cache_resource
def load_labels(train_csv: str):
    return load_label_names(train_csv)


# ============================================================================
# Sidebar — config
# ============================================================================
st.sidebar.title("Configuration")

available_ckpts = [c for c in DEFAULT_CKPTS if Path(c).exists()]
if not available_ckpts:
    st.error("No checkpoints found. Train the models first or edit DEFAULT_CKPTS in predict_cli.py.")
    st.stop()

st.sidebar.markdown(f"**Checkpoints loaded:** {len(available_ckpts)}")
for c in available_ckpts:
    st.sidebar.caption(f"  • {Path(c).name}")

top_k          = st.sidebar.slider("Top-K species per chunk", 1, 10, 3)
no_bird_thresh = st.sidebar.slider("'Not a bird' threshold", 0.05, 0.80, 0.10, 0.01)
tta_hop        = st.sidebar.slider("TTA hop (seconds)", 0.5, 5.0, 2.5, 0.5)

st.sidebar.markdown("**Noise robustness:**")
noise_preprocess   = st.sidebar.checkbox("Spectral noise gating + bandpass", value=True,
                       help="Estimates noise floor, subtracts it. Filters to 1–8 kHz (bird call range).")
consistency_boost  = st.sidebar.slider("Adjacent-chunk consistency boost", 0.0, 0.2, 0.05, 0.01,
                       help="Boosts species that appear in neighbouring 5-sec chunks.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About:**\n"
    "ECA-NFNet-L0 backbone trained on the BirdCLEF 2025 dataset "
    "(206 Neotropical bird species). Inference uses a multi-fold ensemble "
    "with test-time augmentation."
)

device = "cuda"
models = load_models(tuple(available_ckpts), device=device)
labels = load_labels(DEFAULT_TRAIN_CSV)


# ============================================================================
# Inference + rendering helper (defined BEFORE the tabs that call it)
# ============================================================================
def _do_inference_and_render(audio_path, models, labels, top_k, thresh, tta_hop,
                              noise_preprocess=True, consistency_boost=0.05):
    t0 = time.time()
    with st.spinner("Running inference…"):
        probs = ensemble_predict(
            audio_path             = audio_path,
            ckpts                  = models,
            device                 = "cuda",
            tta_hop_sec            = tta_hop,
            noise_preprocess       = noise_preprocess,
            consistency_vote_boost = consistency_boost,
        )
    dt = time.time() - t0
    n_chunks, n_classes = probs.shape
    st.success(f"Predicted {n_chunks} chunk(s) in {dt:.1f}s.")

    # Summary
    n_no_bird = sum(probs[i].max() < thresh for i in range(n_chunks))
    n_bird    = n_chunks - n_no_bird
    col1, col2, col3 = st.columns(3)
    col1.metric("Total chunks", n_chunks)
    col2.metric("Bird detected",  n_bird)
    col3.metric("No bird",         n_no_bird)

    st.markdown("### Per-chunk breakdown")
    for i in range(n_chunks):
        order = np.argsort(-probs[i])[:top_k]
        max_p = probs[i, order[0]]
        chunk_label = f"**Chunk {i+1}** ({i*5}–{(i+1)*5}s)"

        if max_p < thresh:
            st.warning(f"{chunk_label}: **no bird detected** (max probability = {max_p:.3f})")
        else:
            with st.container(border=True):
                st.markdown(chunk_label)
                for c in order:
                    lbl = labels[c]
                    p   = float(probs[i, c])
                    st.progress(p, text=f"{lbl} — {p:.3f}")


# ============================================================================
# Main page
# ============================================================================
st.title("🐦 BirdCLEF 2025 — Species Predictor")
st.markdown(
    "Upload an audio recording or record live. The model predicts up to 206 "
    "Neotropical bird species per 5-second chunk, with confidence scores and "
    "open-set 'not a bird' rejection."
)

tab_upload, tab_record, tab_info = st.tabs(["📁 Upload audio", "🎙️ Record live", "ℹ️ Info"])


# ─── Upload tab ────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Drop a WAV/MP3/OGG/FLAC file",
        type = ["wav", "mp3", "ogg", "flac"],
    )
    if uploaded is not None:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes)

        # Save to a temp path so torchaudio can load it
        suffix = "." + uploaded.name.split(".")[-1]
        tmp_path = f"/tmp/streamlit_upload{suffix}"
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        if st.button("🔍 Run prediction", key="predict_upload"):
            _do_inference_and_render(tmp_path, models, labels, top_k, no_bird_thresh, tta_hop,
                                     noise_preprocess, consistency_boost)


# ─── Record tab ────────────────────────────────────────────────────────────
with tab_record:
    st.markdown("Press the mic button to record (≥5 seconds recommended).")
    rec = st.audio_input("Record audio")
    if rec is not None:
        st.audio(rec)
        tmp_path = "/tmp/streamlit_recording.wav"
        with open(tmp_path, "wb") as f:
            f.write(rec.read())
        if st.button("🔍 Run prediction", key="predict_record"):
            _do_inference_and_render(tmp_path, models, labels, top_k, no_bird_thresh, tta_hop,
                                     noise_preprocess, consistency_boost)


# ─── Info tab ──────────────────────────────────────────────────────────────
with tab_info:
    st.markdown("""
### Model details
- **Backbone**: ECA-NFNet-L0 (`timm`), 23.1M params
- **Input**: 5-second mono audio @ 32 kHz → 128-mel spectrogram
- **Output**: 206-class sigmoid probabilities (multi-label)
- **Training data**: BirdCLEF 2025 focal recordings (xeno-canto)
- **Augmentation**: ESC-50 environmental noise mixing (SNR 3–20 dB, p=0.5)
- **Rare-class filter**: classes with <50 samples dropped from training (128 of 206 classes well-trained)

### Inference pipeline
1. Resample input audio to 32 kHz mono
2. Split into 5-second chunks (with overlapping TTA windows)
3. Run each chunk through ensemble (multiple checkpoints)
4. Average sigmoid probabilities across ensemble + TTA windows
5. Apply max-prob threshold for "not a bird" rejection
6. Return top-K species per chunk

### Known limitations
- Trained on focal (single-bird, clean) recordings → soundscape performance is lower
- 78 rare classes (<50 samples) won't be predicted reliably
- Heavy multi-bird overlap → model picks the dominant species
- Phone-compressed audio degrades accuracy
""")
