"""
FastAPI web server — exposes the MySQL chatbot as an HTTP API for UI integration.

Endpoints:
  GET  /health          Health check
  POST /chat            Blocking chat (returns full reply)
  POST /chat/stream     Streaming chat via Server-Sent Events
  POST /reset           Clear session history
  GET  /schema          Full DB schema (for UI schema explorer)
"""

import json
import os
import sys

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import HermesAgent
from db_tool import create_db_tool

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
    sys.exit(1)

with open("config.yaml") as fh:
    _config = yaml.safe_load(fh)

_db_tool = create_db_tool(_config)
_sessions: dict[str, HermesAgent] = {}

# Load skills.md once at startup — shared across all sessions
_skills_text: str = ""
_skills_file = _config.get("claude", {}).get("skills_file", "")
if _skills_file and os.path.exists(_skills_file):
    with open(_skills_file) as fh:
        _skills_text = fh.read().strip()
    print(f"Loaded domain knowledge from {_skills_file} ({len(_skills_text):,} chars)")


def _get_agent(session_id: str) -> HermesAgent:
    if session_id not in _sessions:
        _sessions[session_id] = HermesAgent(_config, _db_tool, skills_text=_skills_text)
    return _sessions[session_id]


app = FastAPI(title="MySQL Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ResetRequest(BaseModel):
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    """Blocking chat — waits for the full reply before returning."""
    agent = _get_agent(req.session_id)
    try:
        reply = agent.chat(req.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"reply": reply, "session_id": req.session_id}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streaming chat via Server-Sent Events.
    Each event: data: {"text": "..."}\n\n
    Final event: data: {"done": true}\n\n
    Error event: data: {"error": "..."}\n\n
    """
    agent = _get_agent(req.session_id)

    def generate():
        try:
            for chunk in agent.chat_stream(req.message):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reset")
def reset(req: ResetRequest):
    """Clear conversation history for a session."""
    _get_agent(req.session_id).reset()
    return {"status": "ok", "session_id": req.session_id}


@app.get("/schema")
def schema():
    """Return full DB schema — useful for a UI schema explorer panel."""
    result = _db_tool.get_schema()
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result
