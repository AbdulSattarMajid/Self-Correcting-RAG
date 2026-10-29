"""Streamlit chat UI for the Self-Correcting RAG pipeline.

Two knowledge sources:
  - Architecture library: the permanent ChromaDB/BM25 index (web fallback enabled)
  - My uploaded PDFs: a per-session, in-memory index (answers come from the PDFs only)

Launch (from the project root):  streamlit run app/ui.py
"""

import os
import streamlit as st

from uploads import build_session_index, UploadError, MAX_FILES, MAX_PAGES, MAX_FILE_MB
from tracing import flush_async, start_request_trace

# ── Page config (must be the first Streamlit call) ──────────────────
st.set_page_config(
    page_title="Knowledge Studio",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #64748b;
        --line: #e5eaf2;
        --accent: #5865d9;
        --accent-dark: #4653c2;
        --canvas: #f5f7fb;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 74% 0%, rgba(216, 224, 255, .38), transparent 34rem),
            var(--canvas);
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .main .block-container {
        max-width: 1120px;
        padding: 2rem 2.5rem 8rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        letter-spacing: -.035em;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(255, 255, 255, .08);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.15rem 2rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #d7deeb;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, .1);
        margin: 1.35rem 0;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: .75rem;
        margin: .15rem 0 2rem;
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 2.45rem;
        height: 2.45rem;
        border: 1px solid rgba(255, 255, 255, .2);
        border-radius: .85rem;
        background: linear-gradient(145deg, #7884f0, #4d59ca);
        color: white;
        font-size: 1.25rem;
        box-shadow: 0 6px 18px rgba(67, 82, 196, .35);
    }

    .sidebar-brand strong,
    .sidebar-brand small {
        display: block;
    }

    .sidebar-brand strong {
        color: white;
        font: 800 .78rem Inter, ui-sans-serif, system-ui, sans-serif;
        letter-spacing: .12em;
    }

    .sidebar-brand small {
        color: #9aa8be;
        font-size: .73rem;
        margin-top: .15rem;
    }

    .section-eyebrow {
        color: #91a0b7;
        font-size: .66rem;
        font-weight: 700;
        letter-spacing: .13em;
        margin: 0 0 .45rem;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: .35rem;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border: 1px solid rgba(255, 255, 255, .08);
        border-radius: .7rem;
        background: rgba(255, 255, 255, .035);
        padding: .55rem .65rem;
        transition: background .15s ease, border-color .15s ease;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, .08);
        border-color: rgba(255, 255, 255, .16);
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        border: 1px dashed #46536a;
        border-radius: .8rem;
        background: rgba(255, 255, 255, .035);
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section small {
        color: #b7c2d4;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border: 1px solid rgba(255, 255, 255, .12);
        border-radius: .7rem;
        background: #273249;
        color: #f8fafc;
        font-weight: 600;
        transition: background .15s ease, transform .15s ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #6976de;
        background: #34415d;
        color: white;
        transform: translateY(-1px);
    }

    [data-testid="stSidebar"] [data-testid="stToggle"] label {
        color: #d7deeb;
    }

    .workspace-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: .25rem 0 2.3rem;
    }

    .workspace-title {
        color: #78859b;
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .15em;
        text-transform: uppercase;
    }

    .workspace-subtitle {
        color: #344054;
        font: 700 1.02rem Inter, ui-sans-serif, system-ui, sans-serif;
        margin-top: .3rem;
    }

    .mode-pill {
        display: inline-flex;
        align-items: center;
        gap: .5rem;
        border: 1px solid #dfe5f1;
        border-radius: 999px;
        background: rgba(255, 255, 255, .75);
        color: #475467;
        font-size: .78rem;
        font-weight: 600;
        padding: .5rem .85rem;
        white-space: nowrap;
    }

    .mode-dot {
        width: .48rem;
        height: .48rem;
        border-radius: 50%;
        background: #6b74df;
        box-shadow: 0 0 0 3px rgba(107, 116, 223, .12);
    }

    .welcome-panel {
        position: relative;
        overflow: hidden;
        border: 1px solid #e4e8f3;
        border-radius: 1.15rem;
        background: linear-gradient(120deg, rgba(255,255,255,.98), rgba(247,248,255,.94));
        margin: 0 0 1.6rem;
        padding: 2rem 2.15rem 1.55rem;
        box-shadow: 0 12px 35px rgba(27, 39, 73, .045);
    }

    .welcome-panel:after {
        position: absolute;
        top: -5.8rem;
        right: -3.2rem;
        width: 15rem;
        height: 15rem;
        border: 1px solid rgba(88, 101, 217, .1);
        border-radius: 50%;
        box-shadow: 0 0 0 1.5rem rgba(88, 101, 217, .025), 0 0 0 3rem rgba(88, 101, 217, .02);
        content: "";
        pointer-events: none;
    }

    .welcome-eyebrow {
        color: var(--accent);
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
    }

    .welcome-panel h1 {
        max-width: 40rem;
        font-size: clamp(1.8rem, 3.2vw, 2.5rem);
        line-height: 1.17;
        margin: .55rem 0 .65rem;
    }

    .welcome-copy {
        max-width: 42rem;
        color: var(--muted);
        font-size: .96rem;
        line-height: 1.65;
    }

    .feature-row {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1.35rem;
    }

    .feature-chip {
        border: 1px solid #e5e9f2;
        border-radius: .55rem;
        background: #fff;
        color: #536176;
        font-size: .76rem;
        font-weight: 600;
        padding: .45rem .68rem;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid #e8ebf2;
        border-radius: 1rem;
        background: rgba(255, 255, 255, .9);
        box-shadow: 0 5px 18px rgba(28, 39, 70, .035);
        margin: .75rem 0;
        padding: 1.1rem 1.25rem;
    }

    [data-testid="stChatMessage"][data-testid*="user"] {
        background: #f0f2ff;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.7;
    }

    [data-testid="stChatInput"] {
        border: 1px solid #dfe4ee;
        border-radius: 1rem;
        background: rgba(255, 255, 255, .97);
        box-shadow: 0 10px 32px rgba(31, 42, 75, .1);
        padding: .45rem .65rem;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #8d96e8;
        box-shadow: 0 0 0 3px rgba(88, 101, 217, .12), 0 10px 32px rgba(31, 42, 75, .1);
    }

    [data-testid="stChatInput"] textarea {
        color: var(--ink);
    }

    [data-testid="stStatusWidget"] {
        border: 1px solid #e5e9f2;
        border-radius: .9rem;
        background: white;
    }

    [data-testid="stExpander"] {
        border-color: #e5e9f2;
        border-radius: .8rem;
        background: rgba(255, 255, 255, .72);
    }

    @media (max-width: 700px) {
        .main .block-container {
            padding: 1.25rem 1rem 7rem;
        }
        .workspace-header {
            align-items: flex-start;
            flex-direction: column;
            margin-bottom: 1.5rem;
        }
        .welcome-panel {
            padding: 1.5rem 1.25rem 1.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

LIBRARY, UPLOADS = "📚 Architecture library", "📄 My uploaded PDFs"

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><span class="brand-mark">✦</span>'
        '<div><strong>KNOWLEDGE STUDIO</strong><small>Self-correcting RAG</small></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-eyebrow">Workspace</div>', unsafe_allow_html=True)
    st.title("Your sources")

    source = st.radio("Knowledge source", [LIBRARY, UPLOADS])
    uploads_mode = source == UPLOADS

    uploaded_files, process_clicked, upload_status = [], False, None
    if uploads_mode:
        st.markdown(
            '<div class="section-eyebrow">Add your documents</div>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            f"Upload up to {MAX_FILES} PDFs (max {MAX_PAGES} pages, {MAX_FILE_MB} MB each)",
            type="pdf",
            accept_multiple_files=True,
        )
        process_clicked = st.button("⚙️ Process PDFs", disabled=not uploaded_files)
        upload_status = st.container()  # status messages appear right under the button

    st.divider()
    st.markdown('<div class="section-eyebrow">Preferences</div>', unsafe_allow_html=True)
    mock_mode = st.toggle(
        "🧪 Dev mock mode",
        value=os.environ.get("MOCK_LLM") == "1",
        help="Skip real LLM calls and return canned answers (saves API quota).",
    )
    # nodes.py reads MOCK_LLM at call time, so setting it here applies to the next run.
    os.environ["MOCK_LLM"] = "1" if mock_mode else "0"
    if mock_mode:
        st.info("Mock mode is **ON**: LLM calls return canned text.")

    if st.button("🗑️ Clear chat"):
        st.session_state.pop("messages", None)
        st.session_state.pop("cache", None)
        st.rerun()

    st.divider()
    st.subheader("How answers are built")
    st.markdown(
        "1. **Retrieve**: hybrid (vector + BM25) search with cross-encoder reranking\n"
        "2. **Grade**: the LLM judges relevance of each chunk\n"
        "3. **Web search**: Tavily fallback (library mode only)\n"
        "4. **Generate**: the LLM writes an answer from the context\n"
        "5. **Critic**: hallucination check; may regenerate once"
    )


# ── Load the pipeline once per Streamlit process ────────────────────
@st.cache_resource(show_spinner="Loading models & databases (first run only)…")
def load_pipeline():
    from graph import crag_app, initial_state  # importing runs nodes.py setup
    import nodes
    return crag_app, initial_state, nodes.embed_model.get_text_embedding_batch


crag_app, initial_state, embed_fn = load_pipeline()

# ── Session state ───────────────────────────────────────────────────
st.session_state.setdefault("messages", [])      # structured message dicts
st.session_state.setdefault("cache", {})         # (mock, source, question) -> assistant message
st.session_state.setdefault("upload_index", None)
st.session_state.setdefault("upload_sig", None)

# ── Build the PDF index when requested (kept in memory, per session) ─
if process_clicked:
    files = [(f.name, f.getvalue()) for f in uploaded_files]
    try:
        with st.spinner("Reading and indexing PDFs…"):
            st.session_state.upload_index = build_session_index(files, embed_fn)
        st.session_state.upload_sig = tuple((n, len(b)) for n, b in files)
    except UploadError as exc:
        st.session_state.upload_index = None
        st.session_state.upload_sig = None
        if upload_status is not None:
            upload_status.error(str(exc))

index = st.session_state.upload_index
if uploads_mode and index is not None and upload_status is not None:
    upload_status.success("Ready: " + ", ".join(f"{n} ({p} p.)" for n, p in index.files))
    current_sig = tuple((f.name, f.size) for f in uploaded_files)
    if current_sig != st.session_state.upload_sig:
        upload_status.warning("Uploaded files changed. Click **Process PDFs** to re-index.")

active_index = index if uploads_mode else None
source_sig = "library" if active_index is None else st.session_state.upload_sig

mode_label = "My uploaded PDFs" if uploads_mode else "Architecture library"
mode_description = (
    "Search the architecture library and use web search when the sources need a fresh perspective."
    if not uploads_mode
    else "Ask questions grounded in PDFs indexed for this session."
)
st.markdown(
    f"""
    <div class="workspace-header">
        <div>
            <div class="workspace-title">Research workspace</div>
            <div class="workspace-subtitle">A clearer path from question to evidence</div>
        </div>
        <div class="mode-pill"><span class="mode-dot"></span>{mode_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if not st.session_state.messages:
    st.markdown(
        f"""
        <section class="welcome-panel">
            <div class="welcome-eyebrow">Grounded answers, built to be checked</div>
            <h1>Ask a question. Follow the evidence.</h1>
            <div class="welcome-copy">{mode_description}</div>
            <div class="feature-row">
                <span class="feature-chip">Hybrid retrieval</span>
                <span class="feature-chip">Answer verification</span>
                <span class="feature-chip">Source excerpts</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


# ── Helpers ─────────────────────────────────────────────────────────
def redact(text: str) -> str:
    """Remove API key values from an error string before showing it."""
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY"):
        val = os.environ.get(var, "")
        if val and val in text:
            text = text.replace(val, "***")
    return text


def build_message(state: dict, from_uploads: bool) -> dict:
    """Turn the final graph state into a displayable assistant message."""
    sources = []
    for i, doc in enumerate(state.get("documents", []), 1):
        label = "🌐 Web search results" if doc.startswith("WEB SEARCH RESULTS") else f"📄 Chunk {i}"
        sources.append((label, doc[:800] + ("…" if len(doc) > 800 else "")))
    return {
        "role": "assistant",
        "content": state.get("generation") or "_No answer produced._",
        "grounded": state.get("grounded", "?"),
        "checks": state.get("retries", 0),
        "sources": sources,
        "from_uploads": from_uploads,
    }


def render_assistant(msg: dict):
    """Render an assistant message with native widgets (no raw HTML)."""
    if msg.get("error"):
        st.error(msg["content"])
        return
    if msg.get("cached"):
        st.caption("Cached result: no API calls made.")
    if msg.get("from_uploads"):
        st.caption("📄 Answered from your uploaded PDFs")
    color = "green" if msg.get("grounded") == "yes" else "red"
    st.markdown(f":{color}[Grounded: {msg.get('grounded', '?')}]  •  Critic checks: {msg.get('checks', 0)}")
    if msg.get("grounded") == "no":
        st.warning("The critic could not fully verify this answer against the sources. "
                   "Check the cited excerpts before relying on it.")
    st.markdown(msg["content"])
    if msg.get("sources"):
        with st.expander("Sources / retrieved context"):
            for label, text in msg["sources"]:
                st.markdown(f"**{label}**")
                st.code(text, language=None)


def run_pipeline(question: str, search_index) -> dict:
    """Stream the graph, show one line per finished node, return the merged final state."""
    state = {}
    mode = "uploads" if search_index is not None else "library"
    trace = start_request_trace(
        question, mode, len(search_index.files) if search_index is not None else 0
    )
    try:
        with trace:
            config = trace.graph_config({"configurable": {"index": search_index}})
            with st.status("Running Corrective-RAG pipeline…", expanded=True) as status:
                updates = crag_app.stream(
                    initial_state(question), config=config, stream_mode="updates"
                )
                for update in trace.iterate(updates):
                    for node_name, partial in update.items():
                        state.update(partial)

                        if node_name == "retrieve":
                            st.write(f"🔎 **Retrieve**: found {len(partial.get('documents', []))} parent chunk(s)")
                        elif node_name == "grade_documents":
                            kept = len(partial.get("documents", []))
                            ws = partial.get("web_search") == "Yes"
                            st.write(f"📝 **Grade documents**: {kept} chunk(s) kept" + (" → web search will run" if ws else ""))
                        elif node_name == "web_search":
                            st.write("🌐 **Web search**: Tavily fallback executed")
                        elif node_name == "generate":
                            st.write(f"✍️ **Generate**: attempt {state.get('retries', 0) + 1}")
                        elif node_name == "grade_generation":
                            st.write(f"🧐 **Critic**: grounded: {partial.get('grounded', '?')}")

                trace.finish(state)
                status.update(label="Pipeline finished ✅", state="complete", expanded=False)
    finally:
        flush_async()
    return state


# ── Render chat history ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)

# ── Chat input ──────────────────────────────────────────────────────
needs_upload = uploads_mode and index is None
if needs_upload:
    st.info("Upload your PDFs in the sidebar and click **Process PDFs** to start chatting with them.")

placeholder = "Ask a question about your PDFs…" if uploads_mode else "Ask a question about software architecture…"
if question := st.chat_input(placeholder, disabled=needs_upload):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Cache per (mock mode, source, question) so fake or other-source answers never leak across.
    cache_key = (mock_mode, source_sig, question.strip().lower())

    with st.chat_message("assistant"):
        if cache_key in st.session_state.cache:
            msg = {**st.session_state.cache[cache_key], "cached": True}
        else:
            try:
                final_state = run_pipeline(question, active_index)
                msg = build_message(final_state, from_uploads=active_index is not None)
                st.session_state.cache[cache_key] = msg
            except Exception as exc:
                msg = {"role": "assistant", "content": f"Pipeline error: {redact(str(exc))[:300]}", "error": True}
        render_assistant(msg)

    st.session_state.messages.append({k: v for k, v in msg.items() if k != "cached"})