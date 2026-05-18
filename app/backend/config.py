"""config.py — Centralized configuration loaded from environment + .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent.parent   # project root
APP_DIR         = ROOT_DIR / "app"
BACKEND_DIR     = APP_DIR / "backend"
SCRIPTS_DIR     = ROOT_DIR / "birdclef_combined" / "scripts"
CKPT_DIR        = ROOT_DIR / "BACKUP_FINAL"

DATA_DIR        = BACKEND_DIR / "data"
SPECIES_INFO_JSON = DATA_DIR / "species_info.json"
PER_CLASS_AUC_CSV = CKPT_DIR / "ensemble_per_class_auc.csv"
REPORT_FIGURES_DIR = CKPT_DIR / "report_figures"

# Train.csv — needed for label list. User must provide.
TRAIN_CSV = os.getenv(
    "TRAIN_CSV",
    str(ROOT_DIR / "birdclef-2025" / "train.csv")
)


# ── Checkpoints (4-ckpt ensemble) ─────────────────────────────────────────
DEFAULT_CKPTS = [
    CKPT_DIR / "best-epoch07-auc0.7756.ckpt",   # v3 clean
    CKPT_DIR / "best-epoch06-auc0.7841.ckpt",   # v3 FiLM
    CKPT_DIR / "esc50_bg_peak.ckpt",             # ESC-50 BG
    CKPT_DIR / "best-epoch08-auc0.6976.ckpt",   # new fold 1
]


# ── Device / inference ────────────────────────────────────────────────────
DEVICE              = os.getenv("DEVICE", "cpu")    # "cuda" on GPU machines, "cpu" on laptops
TTA_HOP_SEC         = float(os.getenv("TTA_HOP_SEC", "2.5"))
DEFAULT_TOP_K       = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_NO_BIRD_THR = float(os.getenv("DEFAULT_NO_BIRD_THR", "0.10"))
CONSISTENCY_BOOST   = float(os.getenv("CONSISTENCY_BOOST", "0.05"))


# ── Groq RAG ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── Server ────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8081"))
CORS_ALLOW_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
).split(",")
