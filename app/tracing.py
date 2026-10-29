"""Optional Langfuse tracing shared by the API, Streamlit UI, and LLM nodes."""

import contextvars
import os
import re
import threading
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_client = None
_client_lock = threading.Lock()
_warning_lock = threading.Lock()
_warning_emitted = False
_active_trace = contextvars.ContextVar("crag_langfuse_trace", default=None)

_OVERVIEW_RE = re.compile(
    r"\b(summari[sz]e|summary|summaries|overview|tl;?dr|main points|key points|key takeaways"
    r"|what (?:is|are) (?:this|these|they|it) about)\b",
    re.I,
)
_FRONT_RE = re.compile(r"\b(authors?|written by|who wrote|affiliations?|abstract)\b", re.I)
_COMPARISON_RE = re.compile(
    r"\b(agree\w*|disagree\w*|differ\w*|difference\w*|compar\w+|contrast\w+|versus|vs"
    r"|conflict\w*|contradict\w*|similar\w*|in common)\b"
    r"|\b(which|what)\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+)?(?:documents?|papers?|pdfs?|files?)\b"
    r"|\b(?:any|each|every|all|both)\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+)?(?:documents?|papers?|pdfs?|files?)\b",
    re.I,
)


def _enabled() -> bool:
    return (
        os.environ.get("MOCK_LLM") != "1"
        and bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
        and bool(os.environ.get("LANGFUSE_SECRET_KEY"))
    )


def _warn_once() -> None:
    global _warning_emitted
    with _warning_lock:
        if not _warning_emitted:
            print("[Langfuse] Tracing unavailable; continuing without tracing.")
            _warning_emitted = True


def warn_tracing_error() -> None:
    try:
        _warn_once()
    except Exception:
        pass


def _get_client():
    global _client
    if not _enabled():
        return None
    if _client is None:
        with _client_lock:
            if _client is None:
                try:
                    from langfuse import Langfuse

                    base_url = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
                    _client = Langfuse(
                        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                        base_url=base_url,
                        timeout=1,
                        flush_at=512,
                        flush_interval=60,
                    )
                except Exception:
                    _warn_once()
                    return None
    return _client


def request_kind(question: str, mode: str, file_count: int = 0) -> str:
    if mode != "uploads":
        return "normal"
    if _OVERVIEW_RE.search(question):
        return "overview"
    if _FRONT_RE.search(question):
        return "front"
    if file_count > 1 and _COMPARISON_RE.search(question):
        return "comparison"
    return "normal"


class RequestTrace:
    def __init__(self, question: str, mode: str, file_count: int = 0):
        self.question = question
        self.mode = mode
        self.question_length = len(question)
        self.file_count = file_count
        self.kind = request_kind(question, mode, file_count)
        self.started_at = time.perf_counter()
        self.capture_content = mode == "library" or os.environ.get("LANGFUSE_CAPTURE_CONTENT") == "1"
        self.client = None
        self.observation = None
        self.callback = None
        self.trace_context = None
        self.providers: set[str] = set()

    def __enter__(self):
        self.client = _get_client()
        if self.client is None:
            return self
        try:
            from langfuse.langchain import CallbackHandler

            initial_metadata = {
                "mode": self.mode,
                "question_length": self.question_length,
                "request_kind": self.kind,
            }
            trace_input = self.question if self.capture_content else {
                "mode": self.mode,
                "question_length": self.question_length,
            }
            self.observation = self.client.start_observation(
                name="crag-question",
                as_type="chain",
                input=trace_input,
                metadata=initial_metadata,
            )
            self.trace_context = {
                "trace_id": self.observation.trace_id,
                "parent_span_id": self.observation.id,
            }

            capture_content = self.capture_content
            mode = self.mode

            class CRAGCallbackHandler(CallbackHandler):
                def on_chain_start(
                    handler_self,
                    serialized,
                    inputs,
                    *,
                    run_id,
                    parent_run_id=None,
                    tags=None,
                    metadata=None,
                    **_kwargs,
                ):
                    safe_metadata = {
                        key: value
                        for key, value in (metadata or {}).items()
                        if key in {
                            "langfuse_trace_name",
                            "langfuse_tags",
                            "mode",
                            "question_length",
                            "request_kind",
                            "langgraph_node",
                            "langgraph_step",
                        }
                    }
                    node_name = safe_metadata.get("langgraph_node")
                    if node_name:
                        serialized = {"name": str(node_name)}
                    safe_tags = [f"mode:{mode}"]
                    if capture_content:
                        safe_tags.extend(tag for tag in (tags or []) if tag not in safe_tags)
                    if node_name:
                        safe_tags.append(f"node:{node_name}")
                    super().on_chain_start(
                        serialized,
                        inputs if capture_content else {"content_capture": "disabled"},
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        tags=safe_tags,
                        metadata=safe_metadata,
                        **(_kwargs if capture_content else {}),
                    )

                def on_chain_end(
                    handler_self,
                    outputs,
                    *,
                    run_id,
                    parent_run_id=None,
                    **_kwargs,
                ):
                    super().on_chain_end(
                        outputs if capture_content else {"content_capture": "disabled"},
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        **(
                            _kwargs
                            if capture_content
                            else {"inputs": {"content_capture": "disabled"}}
                        ),
                    )

                def on_chain_error(
                    handler_self,
                    error,
                    *,
                    run_id,
                    parent_run_id=None,
                    **_kwargs,
                ):
                    super().on_chain_error(
                        RuntimeError(f"Graph node failed ({type(error).__name__})"),
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        tags=[f"mode:{mode}"],
                        **(
                            _kwargs
                            if capture_content
                            else {"inputs": {"content_capture": "disabled"}}
                        ),
                    )

            self.callback = CRAGCallbackHandler(trace_context=self.trace_context)
        except Exception:
            _warn_once()
            self.callback = None
            if self.observation is not None:
                try:
                    self.observation.end()
                except Exception:
                    pass
            self.observation = None
            self.trace_context = None
            self.client = None
        return self

    def iterate(self, updates):
        iterator = iter(updates)
        while True:
            token = _active_trace.set(self)
            try:
                try:
                    update = next(iterator)
                except StopIteration:
                    return
            finally:
                try:
                    _active_trace.reset(token)
                except Exception:
                    _warn_once()
            yield update

    def graph_config(self, config: dict) -> dict:
        if self.callback is None:
            return config
        try:
            current_callbacks = config.get("callbacks")
            callbacks = list(current_callbacks) if isinstance(current_callbacks, (list, tuple)) else (
                [current_callbacks] if current_callbacks else []
            )
            callbacks.append(self.callback)
            config["callbacks"] = callbacks
            metadata = dict(config.get("metadata") or {})
            metadata.update(
                {
                    "langfuse_trace_name": "CRAG question",
                    "langfuse_tags": [f"mode:{self.mode}", f"request:{self.kind}"],
                    "mode": self.mode,
                    "question_length": self.question_length,
                    "request_kind": self.kind,
                }
            )
            config["metadata"] = metadata
        except Exception:
            _warn_once()
        return config

    def finish(self, state: dict) -> None:
        if self.observation is None:
            return
        try:
            metadata = {
                "mode": self.mode,
                "question_length": self.question_length,
                "request_kind": self.kind,
                "excerpt_count": len(state.get("documents", [])),
                "grounded": "yes" if state.get("grounded") == "yes" else "no",
                "critic_checks": state.get("retries", 0),
                "providers": sorted(self.providers),
                "models": sorted(getattr(self, "models", set())),
                "latency_ms": round((time.perf_counter() - self.started_at) * 1000, 2),
            }
            output = state.get("generation", "") if self.capture_content else None
            self.observation.update(output=output, metadata=metadata)
        except Exception:
            _warn_once()

    def record_generation(
        self,
        *,
        model: str,
        provider: str,
        latency_ms: float,
        usage: dict[str, int] | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
        excerpt_count: int | None = None,
        status_message: str | None = None,
    ) -> None:
        self.providers.add(provider)
        if not hasattr(self, "models"):
            self.models = set()
        self.models.add(model)
        if self.client is None:
            return
        try:
            observation = self.client.start_observation(
                trace_context=self.trace_context,
                name="llm-call",
                as_type="generation",
                model=model,
                input=input_text if self.capture_content else None,
                output=output_text if self.capture_content else None,
                usage_details=usage or None,
                status_message=status_message,
                level="ERROR" if status_message else None,
                metadata={
                    "mode": self.mode,
                    "provider": provider,
                    "latency_ms": round(latency_ms, 2),
                    "question_length": self.question_length,
                    "request_kind": self.kind,
                    "excerpt_count": excerpt_count if excerpt_count is not None else self.file_count,
                },
            )
            observation.end()
        except Exception:
            _warn_once()

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.observation is not None:
            try:
                if any(value is not None for value in (_exc_type, _exc_value, _traceback)):
                    self.observation.update(
                        level="ERROR", status_message="Request failed"
                    )
                self.observation.end()
            except Exception:
                _warn_once()
        return False


def start_request_trace(question: str, mode: str, file_count: int = 0) -> RequestTrace:
    return RequestTrace(question, mode, file_count)


def record_generation(**kwargs: Any) -> None:
    trace = _active_trace.get()
    if trace is not None:
        try:
            trace.record_generation(**kwargs)
        except Exception:
            _warn_once()


def flush_async() -> None:
    client = _get_client()
    if client is None:
        return

    def flush():
        try:
            client.flush()
        except Exception:
            _warn_once()

    try:
        threading.Thread(target=flush, name="langfuse-flush", daemon=True).start()
    except Exception:
        _warn_once()


def shutdown() -> None:
    client = _client
    if client is None:
        return

    def stop():
        try:
            client.shutdown()
        except Exception:
            _warn_once()

    try:
        worker = threading.Thread(target=stop, name="langfuse-shutdown", daemon=True)
        worker.start()
        worker.join(timeout=1.5)
    except Exception:
        _warn_once()
