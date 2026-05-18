"""routes/chat.py — AI chatbot endpoint (RAG over project + species data)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from rag_service import rag_service
from species_db import species_db


router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role:    str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message:           str
    history:           Optional[List[ChatMessage]] = None
    species_context:   Optional[str] = None   # if user is asking about a specific species, pass its code


class ChatResponse(BaseModel):
    reply:   str
    enabled: bool


@router.post("/", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not rag_service.enabled:
        return ChatResponse(
            reply=("AI chat is disabled. Set GROQ_API_KEY in backend/.env "
                   "(get free key at https://console.groq.com/keys) and restart."),
            enabled=False,
        )

    species_info = None
    if req.species_context:
        species_info = species_db.to_dict(req.species_context)

    history_dicts = None
    if req.history:
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    reply = rag_service.chat(
        user_message    = req.message,
        context_species = species_info,
        history         = history_dicts,
    )
    return ChatResponse(reply=reply, enabled=True)


@router.get("/status")
def chat_status():
    return {"enabled": rag_service.enabled}
