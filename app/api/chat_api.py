from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from typing import Any



from app.orchestrator.orchestrator import orchestrator

router = APIRouter()


# -----------------------------
# Request Model
# -----------------------------

class ChatRequest(BaseModel):

    message: str

    conversation_id: str

    model_key: Optional[str] = "gemini-3.1-flash-lite-preview"


# -----------------------------
# Response Model
# -----------------------------

class ChatResponse(BaseModel):

    reply: Any


# -----------------------------
# Chat Endpoint
# -----------------------------

@router.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(
    request: ChatRequest
):

    result = await orchestrator.handle(
        message=request.message,
        conversation_id=request.conversation_id,
        model_key=request.model_key
    )

    return ChatResponse(
        reply=result
    )