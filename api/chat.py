from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json

from db.db import get_session
from services.services_chat import (
    build_project_context,
    query_groq,
    query_groq_stream,
    check_groq_status,
    get_suggestions,
    DEFAULT_MODEL,
)

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    project_id: int
    question: str
    history: List[ChatMessage] = []
    model: str = DEFAULT_MODEL
    current_page: Optional[str] = None
    dataset_id: Optional[int] = None
    api_key: Optional[str] = ""


# =====================================================
# STREAMING CHAT ENDPOINT
# =====================================================
@router.post("/stream")
async def stream_ask(req: ChatRequest):
    """
    Streams response token by token using SSE.
    Use this for the full chat page.
    """
    db = get_session()
    try:
        context = build_project_context(
            db,
            project_id=req.project_id,
            dataset_id=req.dataset_id,
            current_page=req.current_page,
        )
    finally:
        db.close()

    history = [{"role": m.role, "content": m.content} for m in req.history]

    def generate():
        for chunk in query_groq_stream(
            question=req.question,
            context=context,
            history=history,
            model=req.model,
            api_key=req.api_key or "",
        ):
            # SSE format
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# =====================================================
# NON-STREAMING ENDPOINT (sidebar widget)
# =====================================================
@router.post("/ask")
def ask(req: ChatRequest):
    db = get_session()
    try:
        context = build_project_context(
            db,
            project_id=req.project_id,
            dataset_id=req.dataset_id,
            current_page=req.current_page,
        )
        history = [{"role": m.role, "content": m.content} for m in req.history]
        answer  = query_groq(
            question=req.question,
            context=context,
            history=history,
            model=req.model,
            api_key=req.api_key or "",
        )
        return {"answer": answer}
    finally:
        db.close()


# =====================================================
# STATUS
# =====================================================
@router.get("/status")
def groq_status(api_key: str = ""):
    return check_groq_status(api_key)


# =====================================================
# SUGGESTIONS
# =====================================================
@router.get("/suggestions/{project_id}")
def suggestions(project_id: int, page: Optional[str] = None):
    db = get_session()
    try:
        return {"suggestions": get_suggestions(db, project_id, page)}
    finally:
        db.close()
