"""FastAPI endpoints for library and session-scoped PDF CRAG searches."""

import io
import json
import ntpath
import os
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator, Iterator, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from uploads import MAX_FILE_MB, MAX_FILES, MAX_PAGES, SessionIndex, UploadError, build_session_index
from tracing import flush_async, shutdown as shutdown_tracing, start_request_trace


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    yield
    shutdown_tracing()


app = FastAPI(title="Self-Correcting RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

MAX_LIVE_SESSIONS = 20
SESSION_IDLE_SECONDS = 60 * 60
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MAX_UPLOAD_REQUEST_BYTES = MAX_FILES * MAX_FILE_BYTES + 1024 * 1024

_pipeline = None
_sessions_lock = threading.Lock()


@dataclass
class UploadSession:
    index: SessionIndex | None = None
    files: list[tuple[str, bytes]] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


_sessions: dict[str, UploadSession] = {}


class AskRequest(BaseModel):
    question: str
    mode: Literal["library", "uploads"] = "library"
    session_id: str | None = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        from graph import crag_app, initial_state

        _pipeline = crag_app, initial_state
    return _pipeline


def _load_embed_fn():
    import nodes

    return nodes.embed_model.get_text_embedding_batch


def _redact_error(error: Exception) -> str:
    message = str(error)
    secret_names = (
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "TAVILY_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
    )
    secrets = sorted(
        (os.environ[name] for name in secret_names if len(os.environ.get(name, "")) >= 8),
        key=len,
        reverse=True,
    )
    for secret in secrets:
        message = message.replace(secret, "[REDACTED]")
    message = re.sub(
        r"(?i)\b(api[_ -]?key|authorization|bearer)\b\s*[:=]\s*[^\s,;]+",
        "[REDACTED]",
        message,
    )
    return message[:300]


def _sse(event: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def _prune_sessions_locked(now: float) -> None:
    expired = [
        session_id
        for session_id, session in _sessions.items()
        if now - session.last_accessed > SESSION_IDLE_SECONDS
    ]
    for session_id in expired:
        del _sessions[session_id]


def _get_session(session_id: str) -> UploadSession:
    with _sessions_lock:
        _prune_sessions_locked(time.monotonic())
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Upload session not found. Please re-upload your PDFs.")
        session.last_accessed = time.monotonic()
        return session


def _safe_filename(filename: str) -> str:
    name = ntpath.basename(filename.replace("/", "\\")).strip().replace("\x00", "")
    return name or "uploaded.pdf"


def _validate_pdf(name: str, data: bytes) -> int:
    if not name.lower().endswith(".pdf"):
        raise UploadError("File must have a .pdf extension.")
    if not data.startswith(b"%PDF"):
        raise UploadError("File does not have a valid PDF signature.")
    if len(data) > MAX_FILE_BYTES:
        raise UploadError(f"File is larger than {MAX_FILE_MB} MB.")

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise UploadError("Password-protected PDFs are not supported.")
        pages = len(reader.pages)
    except UploadError:
        raise
    except Exception as exc:
        raise UploadError("File could not be read as a PDF.") from exc

    if pages > MAX_PAGES:
        raise UploadError(f"PDF has {pages} pages; the limit is {MAX_PAGES}.")
    return pages


def _stream_answer(
    question: str, session_index: SessionIndex | None
) -> Iterator[str]:
    trace = start_request_trace(
        question,
        "uploads" if session_index is not None else "library",
        len(session_index.files) if session_index is not None else 0,
    )
    try:
        with trace:
            crag_app, initial_state = _load_pipeline()
            state = {}
            config = trace.graph_config({"configurable": {"index": session_index}})
            updates = crag_app.stream(
                initial_state(question), config=config, stream_mode="updates"
            )
            for update in trace.iterate(updates):
                for node_name, node_update in update.items():
                    state.update(node_update)
                    if node_name == "retrieve":
                        payload = {"node": node_name, "documents": len(node_update.get("documents", []))}
                    elif node_name == "grade_documents":
                        payload = {
                            "node": node_name,
                            "kept": len(node_update.get("documents", [])),
                            "web_search_needed": node_update.get("web_search") == "Yes",
                        }
                    elif node_name == "generate":
                        payload = {"node": node_name, "attempt": state.get("retries", 0) + 1}
                    elif node_name == "grade_generation":
                        payload = {
                            "node": node_name,
                            "grounded": "yes" if node_update.get("grounded") == "yes" else "no",
                        }
                    elif node_name == "web_search":
                        payload = {"node": node_name}
                    else:
                        continue
                    yield _sse("node", payload)

            trace.finish(state)
            sources = []
            for document in state.get("documents", []):
                source_type = (
                    "web"
                    if document.startswith("WEB SEARCH RESULTS")
                    else "upload" if session_index is not None else "library"
                )
                sources.append({"type": source_type, "text": document[:800]})

            final_event = _sse(
                "final",
                {
                    "question": question,
                    "answer": state.get("generation", ""),
                    "grounded": "yes" if state.get("grounded") == "yes" else "no",
                    "critic_checks": state.get("retries", 0),
                    "sources": sources,
                },
            )
        yield final_event
    except Exception as exc:
        yield _sse("error", {"message": _redact_error(exc)})
    finally:
        flush_async()


@app.middleware("http")
async def reject_oversized_upload(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/sessions/") and request.url.path.endswith("/files"):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_UPLOAD_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Upload request exceeds the maximum allowed size."},
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}




@app.post("/sessions")
def create_session():
    now = time.monotonic()
    session_id = str(uuid.uuid4())
    with _sessions_lock:
        _prune_sessions_locked(now)
        if len(_sessions) >= MAX_LIVE_SESSIONS:
            oldest_id = min(_sessions, key=lambda item: _sessions[item].last_accessed)
            del _sessions[oldest_id]
        _sessions[session_id] = UploadSession(last_accessed=now)
    return {"session_id": session_id}


@app.post("/sessions/{session_id}/files")
def upload_session_files(session_id: str, files: list[UploadFile] = File(...)):
    session = _get_session(session_id)
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF.")

    statuses = []
    pending: list[tuple[str, bytes]] = []
    pending_statuses = []
    with session.lock:
        session.last_accessed = time.monotonic()
        for upload in files:
            name = _safe_filename(upload.filename or "")
            data = upload.file.read(MAX_FILE_BYTES + 1)
            if len(session.files) + len(pending) >= MAX_FILES:
                statuses.append(
                    {"name": name, "pages": 0, "ok": False, "error": f"You can upload at most {MAX_FILES} PDFs."}
                )
                continue
            try:
                pages = _validate_pdf(name, data)
            except UploadError as exc:
                statuses.append({"name": name, "pages": 0, "ok": False, "error": str(exc)})
                continue

            pending.append((name, data))
            status = {"name": name, "pages": pages, "ok": True}
            statuses.append(status)
            pending_statuses.append(status)

        if pending:
            candidate_files = session.files + pending
            try:
                new_index = build_session_index(candidate_files, _load_embed_fn())
            except UploadError as exc:
                for status in pending_statuses:
                    status["ok"] = False
                    status["error"] = str(exc)
            except Exception as exc:
                message = _redact_error(exc) or "Could not build the PDF index."
                for status in pending_statuses:
                    status["ok"] = False
                    status["error"] = f"Could not build the PDF index: {message}"
            else:
                session.files = candidate_files
                session.index = new_index

    return {"files": statuses, "total_files": len(session.files)}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = _get_session(session_id)
    with session.lock:
        return {
            "session_id": session_id,
            "files": [{"name": name, "pages": pages} for name, pages in session.index.files]
            if session.index is not None
            else [],
            "total_files": len(session.files),
        }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    with _sessions_lock:
        _prune_sessions_locked(time.monotonic())
        if _sessions.pop(session_id, None) is None:
            raise HTTPException(status_code=404, detail="Upload session not found.")
    return {"deleted": True}


@app.post("/ask")
def ask(request: AskRequest):
    session_index = None
    if request.mode == "uploads":
        if not request.session_id:
            raise HTTPException(status_code=400, detail="Upload mode requires a session_id.")
        session = _get_session(request.session_id)
        with session.lock:
            if session.index is None or not session.files:
                raise HTTPException(status_code=400, detail="Upload at least one PDF before asking a question.")
            session_index = session.index

    return StreamingResponse(
        _stream_answer(request.question, session_index),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
