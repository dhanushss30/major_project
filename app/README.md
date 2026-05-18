# WildEar — Research Lab App

A research-grade web application for the WildEar Neotropical bird species classifier
(206 classes, 4-checkpoint ECA-NFNet-L0 ensemble, val/auc = 0.8246).

**Architecture:**
- **Backend**: FastAPI (Python) — serves the ML inference pipeline + RAG over species data
- **Frontend**: React 18 + Vite + Tailwind CSS — research lab dashboard aesthetic
- **RAG/LLM**: Groq API (free) for AI-powered species info + chatbot
- **Visualizations**: Plotly.js (spectrograms, charts), Wavesurfer.js (waveform)

---

## Project structure

```
app/
├── backend/                    # FastAPI Python backend
│   ├── main.py                # Entry point
│   ├── config.py              # Environment + paths
│   ├── inference_service.py   # Ensemble inference wrapper
│   ├── rag_service.py         # Groq-based RAG for species info
│   ├── species_db.py          # 206-class metadata database
│   ├── audio_utils.py         # Spectrogram + waveform generation
│   ├── routes/
│   │   ├── predict.py         # POST /api/predict
│   │   ├── species.py         # GET /api/species, /api/species/{code}
│   │   ├── chat.py            # POST /api/chat (RAG chatbot)
│   │   ├── dashboard.py       # GET /api/metrics, /api/per-class-auc
│   │   └── audio.py           # GET /api/spectrogram, /api/waveform
│   ├── data/
│   │   └── species_info.json  # Full 206-class info
│   ├── requirements.txt
│   └── .env.example
└── frontend/                   # React + Vite frontend
    ├── src/
    │   ├── main.jsx
    │   ├── App.jsx
    │   ├── api/client.js
    │   ├── components/
    │   │   ├── Layout.jsx
    │   │   ├── Sidebar.jsx
    │   │   ├── AudioUpload.jsx
    │   │   ├── LiveRecord.jsx
    │   │   ├── SpectrogramView.jsx
    │   │   ├── WaveformView.jsx
    │   │   ├── ConfidenceBars.jsx
    │   │   ├── SpeciesCard.jsx
    │   │   ├── ChatBot.jsx
    │   │   └── MetricCard.jsx
    │   ├── pages/
    │   │   ├── HomePage.jsx
    │   │   ├── PredictPage.jsx
    │   │   ├── DashboardPage.jsx
    │   │   ├── SpeciesPage.jsx
    │   │   ├── ModelInfoPage.jsx
    │   │   └── ChatPage.jsx
    │   ├── styles/
    │   │   └── globals.css
    │   └── utils/
    │       └── audio.js
    ├── public/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── postcss.config.js
```

---

## Setup

### Prerequisites

- **Python 3.12+** (https://www.python.org/downloads/)
- **Node.js 18+** (https://nodejs.org/)
- **Groq API key** (free, get at https://console.groq.com/keys)
- The 4 trained `.ckpt` files in `BACKUP_FINAL/`

### Backend setup

```powershell
cd app/backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env: add your GROQ_API_KEY
uvicorn main:app --reload --port 8081
```

Backend runs at `http://localhost:8081`. API docs at `http://localhost:8081/docs`.

### Frontend setup

```powershell
cd app/frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

### Open the app

`http://localhost:5173` in your browser.

---

## Features

- **Predict tab**: drag-and-drop or live-record audio. See:
  - Audio waveform
  - Mel spectrogram visualization (per chunk)
  - Per-5-second predictions with animated confidence bars
  - Top-1 species detail card with AI-generated info
- **Species explorer**: browse all 206 species, filter by family/region/rarity
- **Dashboard**: live model metrics — per-class AUC histogram, training trajectory, ensemble comparison
- **AI Chat**: ask the chatbot about any species, the model methodology, or bird identification
- **Model info**: architecture diagram, training details, novel research contributions

---

## Theme

Research-lab dark aesthetic: deep charcoal background, accents in cyan/lime, monospace headers for technical feel, ample whitespace, generous animations.
