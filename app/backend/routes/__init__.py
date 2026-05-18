# routes/__init__.py
from .predict import router as predict_router
from .species import router as species_router
from .chat import router as chat_router
from .dashboard import router as dashboard_router
from .audio import router as audio_router

__all__ = [
    "predict_router",
    "species_router",
    "chat_router",
    "dashboard_router",
    "audio_router",
]
