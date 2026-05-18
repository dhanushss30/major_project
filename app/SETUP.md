# WildEar Research Lab — Setup Guide

Two parts: **backend** (Python/FastAPI) and **frontend** (React/Vite). Run both, open browser.

---

## Prerequisites

1. **Python 3.12+** — https://www.python.org/downloads/
2. **Node.js 18+** — https://nodejs.org/
3. **Groq API key (free)** — sign up at https://console.groq.com/keys → create an API key
4. **train.csv** — from the BirdCLEF 2025 Kaggle dataset (source data only). Place at:
   ```
   C:\Users\User\Downloads\birdclef-2025\train.csv
   ```
   (Or set `TRAIN_CSV` env var in `backend/.env` to a different path.)

---

## ✅ Step 1 — Backend setup

Open **PowerShell** in the project folder:

```powershell
cd C:\Users\User\Downloads\birdclef_2025_combined_FINAL\app\backend
```

Create a virtual environment + install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**(This takes ~5-10 min — torch is ~2 GB.)**

Set up the `.env` file:

```powershell
copy .env.example .env
notepad .env
```

In Notepad, paste your Groq API key on the `GROQ_API_KEY=` line:

```
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=llama-3.3-70b-versatile

HOST=0.0.0.0
PORT=8081
CORS_ALLOW_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173
```

Save and close Notepad.

**Launch the backend:**

```powershell
uvicorn main:app --reload --port 8081
```

You should see:
```
==============================================================
 WildEar Research Lab API — Starting
==============================================================
[species_db] loading 206 species…
[species_db] ready — 206 species
[rag_service] ready — using llama-3.3-70b-versatile
[inference_service] loading checkpoints...
  loaded best-epoch07-auc0.7756.ckpt  [clean]  val_auc=0.7756
  loaded best-epoch06-auc0.7841.ckpt  [FiLM]   val_auc=0.7841
  loaded esc50_bg_peak.ckpt           [clean]  val_auc=None
  loaded best-epoch08-auc0.6976.ckpt  [clean]  val_auc=0.6976
[inference_service] ready — 4 ckpts, 206 labels
==============================================================
 Ready — docs at /docs
==============================================================
INFO:     Uvicorn running on http://0.0.0.0:8081
```

Test backend in browser: http://localhost:8081/docs (interactive API docs).

**Keep this PowerShell window open** — backend must stay running.

---

## ✅ Step 2 — Frontend setup

Open a **second** PowerShell window:

```powershell
cd C:\Users\User\Downloads\birdclef_2025_combined_FINAL\app\frontend
```

Install dependencies:

```powershell
npm install
```

**(Takes ~2-3 min.)**

Launch the dev server:

```powershell
npm run dev
```

You should see:
```
  VITE v5.4.x  ready in 800 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

---

## ✅ Step 3 — Open the app

In your browser: **http://localhost:5173**

You should see the WildEar landing page.

---

## 🎯 Testing the app

1. Click **Predict** → drag a demo audio from `BACKUP_FINAL/demo_audio/trokin_rating5.0_XC115603.ogg`
2. Click **Run prediction** → watch ensemble inference (~5-10 seconds on CPU)
3. See:
   - Audio waveform
   - Mel spectrogram (heatmap)
   - Per-5-second top-K predictions
   - "Bird detected" / "No bird" verdicts
   - Confidence bars
4. Click any species in results → navigates to **Species detail** page with AI-generated info
5. Click **Dashboard** → see all model metrics (AUC charts, training trajectory)
6. Click **AI Chat** → ask anything about the model

---

## 🐛 Troubleshooting

### Backend won't start

- **"No checkpoints found"** → ensure `.ckpt` files are in `C:\Users\User\Downloads\birdclef_2025_combined_FINAL\BACKUP_FINAL\`
- **"train.csv not found"** → set `TRAIN_CSV` in `.env` to the actual path
- **"GROQ_API_KEY not set"** → AI chat will be disabled but everything else works. Set it in `.env` to enable chat.
- **Inference is very slow on CPU** → expected, ~5-10 sec per audio. Use GPU machine if available (set `DEVICE=cuda` in `.env`).

### Frontend won't start

- **`npm install` errors** → make sure Node.js 18+ is installed (`node --version`)
- **CORS errors in browser console** → verify backend `CORS_ALLOW_ORIGINS` includes `http://localhost:5173`
- **Connection refused** → make sure backend is running on port 8081

### Audio upload fails

- **File too large** → keep under 50 MB; longer audio takes longer to process
- **Format issues** → use WAV/MP3/OGG/FLAC; M4A may need conversion

---

## 🎨 Customization

### Change the color theme

Edit `frontend/tailwind.config.js` → modify the `colors.accent` palette.

### Add more curated species info

Edit `backend/species_db.py` → add entries to `CURATED_INFO`. Restart backend.

### Change the chatbot model

Edit `backend/.env` → set `GROQ_MODEL` (options: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`).

---

## 🚀 Production deployment (later)

- Backend: package as Docker container, deploy to AWS/GCP/Render
- Frontend: `npm run build` → static `dist/` folder, deploy to Vercel/Netlify/Cloudflare Pages
- Use a CDN for the checkpoint files (~1.5 GB total)

---

## What's where in the codebase

```
app/
├── backend/                    # Python FastAPI
│   ├── main.py                 # Entry — uvicorn main:app
│   ├── config.py               # Paths + env config
│   ├── inference_service.py    # Loads ckpts, runs ensemble
│   ├── species_db.py           # 206-class metadata
│   ├── rag_service.py          # Groq chatbot/RAG
│   ├── audio_utils.py          # Spectrogram + waveform PNG
│   ├── routes/                 # API endpoints
│   │   ├── predict.py          # POST /api/predict
│   │   ├── audio.py            # POST /api/audio/spectrogram, /waveform
│   │   ├── species.py          # GET /api/species, /api/species/{code}
│   │   ├── chat.py             # POST /api/chat
│   │   └── dashboard.py        # GET /api/metrics/*
│   └── requirements.txt
└── frontend/                   # React + Vite
    ├── src/
    │   ├── main.jsx
    │   ├── App.jsx
    │   ├── api/client.js       # All HTTP calls
    │   ├── components/
    │   │   ├── Layout.jsx
    │   │   └── SpeciesCard.jsx
    │   ├── pages/
    │   │   ├── HomePage.jsx
    │   │   ├── PredictPage.jsx       # ← main feature
    │   │   ├── DashboardPage.jsx
    │   │   ├── SpeciesPage.jsx
    │   │   ├── SpeciesDetailPage.jsx
    │   │   ├── ModelInfoPage.jsx
    │   │   └── ChatPage.jsx
    │   └── styles/globals.css
    ├── package.json
    ├── vite.config.js
    └── tailwind.config.js
```
