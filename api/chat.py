"""
Chat API endpoints for the RAG chatbot.
"""

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from rag.chain import get_chat_response, get_streaming_response


router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    status: str = "success"


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Non-streaming chat endpoint.
    Accepts a message and conversation history, returns the AI response.
    """
    try:
        history_dicts = [msg.model_dump() for msg in request.history]
        response = get_chat_response(request.message, history_dicts)
        return ChatResponse(response=response)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Sorry, I'm having trouble responding right now. Please try again."
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Yields response chunks as they are generated.
    """
    async def event_generator():
        try:
            history_dicts = [msg.model_dump() for msg in request.history]
            async for chunk in get_streaming_response(request.message, history_dicts):
                yield {
                    "event": "message",
                    "data": json.dumps({"chunk": chunk}),
                }
            yield {
                "event": "done",
                "data": json.dumps({"status": "complete"}),
            }
        except Exception as e:
            print(f"❌ Stream error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": "Failed to generate response"}),
            }

    return EventSourceResponse(event_generator())
