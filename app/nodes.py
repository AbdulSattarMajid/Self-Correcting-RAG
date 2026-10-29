import os
import re
import json
import time
import pickle

# Offline mode: models must already be in the local HF cache.
# setdefault lets you override from the shell (e.g. HF_HUB_OFFLINE=0) for a first download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import chromadb
from google import genai
from google.genai import types
from tavily import TavilyClient
from groq import Groq
from langchain_core.runnables import RunnableConfig
from sentence_transformers import CrossEncoder
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from typing import Any

from state import GraphState
from common import CHROMA_DIR, BM25_PATH, COLLECTION_NAME, tokenize
from tracing import record_generation, warn_tracing_error

load_dotenv()

# 1. Initialize Clients & Models
print("Initializing Models and Databases for LangGraph...")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None

# Verify this name against the model list in Google AI Studio.
# You can override it in .env with GEMINI_MODEL=...
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

TOP_K = 5          # candidates pulled from each retriever
TOP_N_RERANK = 3   # leaf chunks kept after cross-encoder reranking
OVERVIEW_PAGES_PER_FILE = 4    # pages sampled per uploaded file for summary requests
OVERVIEW_CHARS_PER_PAGE = 1800 # keeps the prompt small enough for free-tier token limits
COMPARE_CANDIDATES_PER_FILE = 8  # per-file candidates; larger pool so 5 distinct pages remain after page-dedupe
COMPARE_PASSAGES_PER_FILE = 5    # best passages kept from EACH file after reranking (was 3)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# 2. Load Local Databases
db = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = db.get_collection(COLLECTION_NAME)

with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
leaf_ids = bm25_data["leaf_ids"]
leaf_texts = bm25_data["leaf_texts"]
leaf_to_text = dict(zip(leaf_ids, leaf_texts))
leaf_to_parent = dict(zip(leaf_ids, bm25_data["leaf_parent_ids"]))
parent_texts = bm25_data["parent_texts"]


# --- LLM HELPER (Gemini primary, Groq fallback) ---
# Verify the Groq model id at console.groq.com/docs/models; override in .env with GROQ_MODEL=...
GROQ_MODEL = os.getenv("GROQ_MODEL") or os.getenv("LLM_MODEL") or "openai/gpt-oss-20b"
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL") or os.getenv("LLM_FALLBACK_MODEL")
# LLM_PROVIDER=groq in .env (or shell) skips Gemini entirely, e.g. when its daily quota is gone.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
PERMANENT_ERROR_CODES = {400, 401, 403, 404}


class LLMResponse:
    """Tiny wrapper so Gemini and Groq results both expose `.text`."""
    def __init__(self, text: str, usage: dict[str, int] | None = None):
        self.text = text
        self.usage = usage


def _usage_counts(
    usage: Any, prompt_field: str, completion_field: str
) -> dict[str, int] | None:
    try:
        if usage is None:
            return None
        if isinstance(usage, dict):
            prompt_tokens = usage.get(prompt_field)
            completion_tokens = usage.get(completion_field)
        else:
            prompt_tokens = getattr(usage, prompt_field, None)
            completion_tokens = getattr(usage, completion_field, None)
        counts = {}
        if prompt_tokens is not None:
            counts["prompt_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            counts["completion_tokens"] = int(completion_tokens)
        return counts or None
    except Exception:
        warn_tracing_error()
        return None


def _record_llm_call(
    contents: str,
    config: Any,
    model: str,
    provider: str,
    started_at: float,
    *,
    response: Any = None,
    usage: dict[str, int] | None = None,
    failed: bool = False,
) -> None:
    try:
        record_generation(
            model=model,
            provider=provider,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            usage=usage,
            input_text=(
                f"System instruction: {getattr(config, 'system_instruction', '') or ''}"
                f"\n\n{contents}"
            ),
            output_text=getattr(response, "text", None),
            excerpt_count=len(re.findall(r"\[E\d+\]|\[Document \d+\]", contents)),
            status_message="Provider request failed" if failed else None,
        )
    except Exception:
        warn_tracing_error()


def _error_status_code(error: Exception) -> int | str:
    try:
        code = getattr(error, "status_code", None)
        if code is None:
            code = getattr(error, "code", None)
        return code if isinstance(code, int) and not isinstance(code, bool) else "unknown"
    except Exception:
        return "unknown"


def _is_quota_error(e: Exception) -> bool:
    msg = str(e)
    return getattr(e, "code", None) == 429 or "429" in msg[:40] or "RESOURCE_EXHAUSTED" in msg


def _call_groq(contents, config) -> LLMResponse:
    """Translate the Gemini-style config into a Groq chat completion."""
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    messages = []
    system = getattr(config, "system_instruction", None)
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": contents})

    kwargs = {}
    if getattr(config, "response_mime_type", None) == "application/json":
        kwargs["response_format"] = {"type": "json_object"}
    if getattr(config, "temperature", None) is not None:
        kwargs["temperature"] = config.temperature

    models = [GROQ_MODEL]
    if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != GROQ_MODEL:
        models.append(GROQ_FALLBACK_MODEL)

    last_error = None
    for model in models:
        started_at = time.perf_counter()
        try:
            resp = groq_client.chat.completions.create(model=model, messages=messages, **kwargs)
            usage = _usage_counts(
                getattr(resp, "usage", None), "prompt_tokens", "completion_tokens"
            )
            result = LLMResponse(resp.choices[0].message.content or "", usage=usage)
            _record_llm_call(
                contents, config, model, "groq", started_at, response=result, usage=usage
            )
            return result
        except Exception as e:
            _record_llm_call(contents, config, model, "groq", started_at, failed=True)
            last_error = e
            print(
                f"  -> Groq model {model} failed "
                f"({type(e).__name__}, code {_error_status_code(e)})."
            )
    raise last_error


def call_llm(contents, config, max_retries: int = 5):
    """Call Gemini with bounded retries; fall back to Groq if Gemini's quota is hit.

    - LLM_PROVIDER=groq: use Groq only.
    - 429 / quota exhausted: switch to Groq immediately (if GROQ_API_KEY is set).
    - Permanent errors (bad key, bad model name, malformed request): fail fast.
    - Other transient errors: exponential backoff, up to max_retries.
    """
    # --- DEV MOCK MODE ---
    # MOCK_LLM=1 skips real LLM calls and returns canned text (test the UI without API quota).
    if os.environ.get("MOCK_LLM") == "1":
        sysins = (getattr(config, "system_instruction", "") or "").lower()
        if "grader" in sysins:
            n = len(re.findall(r"\[Document \d+\]", contents))
            grades = ["yes" if i % 2 == 0 else "no" for i in range(max(n, 1))]
            return LLMResponse(json.dumps({"grades": grades}))
        if "you verify whether" in sysins:
            return LLMResponse("yes")
        question = contents.split("Question:")[-1].strip()
        return LLMResponse(f"[MOCK] Fake answer to: {question}")
    # --- END MOCK MODE ---

    if LLM_PROVIDER == "groq":
        return _call_groq(contents, config)

    last_error = None
    for attempt in range(max_retries):
        started_at = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            )
            usage = _usage_counts(
                getattr(response, "usage_metadata", None),
                "prompt_token_count",
                "candidates_token_count",
            )
            _record_llm_call(
                contents,
                config,
                getattr(response, "model_version", None) or MODEL_NAME,
                "gemini",
                started_at,
                response=response,
                usage=usage,
            )
            return response
        except Exception as e:
            _record_llm_call(
                contents, config, MODEL_NAME, "gemini", started_at, failed=True
            )
            last_error = e
            msg = str(e)

            if _is_quota_error(e):
                if groq_client is not None:
                    print(
                        f"  -> Gemini quota hit "
                        f"({type(e).__name__}, code {_error_status_code(e)}). "
                        "Falling back to Groq..."
                    )
                    return _call_groq(contents, config)
                print(
                    f"  -> Gemini quota hit "
                    f"({type(e).__name__}, code {_error_status_code(e)}); "
                    "no GROQ_API_KEY is set."
                )

            code = getattr(e, "code", None)
            if code in PERMANENT_ERROR_CODES or (
                code is None and any(c in msg[:40] for c in ("400", "401", "403", "404"))
            ):
                raise

            if attempt >= 1 and groq_client is not None:
                print("  -> Gemini keeps failing. Falling back to Groq...")
                return _call_groq(contents, config)

            wait = 2 ** attempt
            print(
                f"  -> Gemini request failed "
                f"({type(e).__name__}, code {_error_status_code(e)}). "
                f"Retrying in {wait}s "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


# --- RETRIEVAL HELPERS ---
def _get_index(config):
    """The per-session uploaded-PDF index passed via config, or None for library mode."""
    return ((config or {}).get("configurable") or {}).get("index")


_OVERVIEW_RE = re.compile(
    r"\b(summari[sz]e|summary|summaries|overview|tl;?dr|main points|key points|key takeaways"
    r"|what (?:is|are) (?:this|these|they|it) about)\b",
    re.I,
)


_COMPARE_RE = re.compile(
    r"\b(agree\w*|disagree\w*|differ\w*|difference\w*|compar\w+|contrast\w*|versus|vs"
    r"|conflict\w*|contradict\w*|similar\w*|in common)\b",
    re.I,
)


# NEW: "which of the documents...", "does any document...", "each/all documents..." questions
# need evidence from EVERY file, even though they contain no compare word.
_COVERAGE_RE = re.compile(
    r"\b(which|what)\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+)?(?:documents?|papers?|pdfs?|files?)\b"
    r"|\b(?:any|each|every|all|both)\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+)?(?:documents?|papers?|pdfs?|files?)\b",
    re.I,
)


# NEW: author / front-page questions. Names live on page 1, which topic search often misses
# (page 1 rarely contains the word "author"), so we read the first page of every file.
_FRONT_RE = re.compile(
    r"\b(authors?|written by|who wrote|affiliations?|abstract)\b", re.I,
)


def _request_kind(index, question: str):
    """'overview' / 'comparison' for requests plain top-3 search handles badly, else None.

    - overview: 'summarize these' has no topic words, so search can't help.
    - comparison: needs evidence from EACH file, but a global top-3 can come from one file.
    """
    if index is None:
        return None
    if _OVERVIEW_RE.search(question):
        return "overview"
    if _FRONT_RE.search(question):
        return "front"
    if len(index.files) > 1 and (_COMPARE_RE.search(question) or _COVERAGE_RE.search(question)):
        return "comparison"
    return None


# NEW: generic words that say "compare the files" but carry no topic, so they are
# removed from the SEARCH query only (the generator still sees the full question).
_QUERY_FILLER = {
    "do", "does", "did", "the", "a", "an", "that", "this", "these", "those", "they", "them",
    "it", "is", "are", "of", "to", "in", "on", "and", "or", "how", "what", "where", "which",
    "document", "documents", "doc", "docs", "paper", "papers", "pdf", "pdfs", "file", "files",
    "three", "two", "both", "all", "any", "between", "among", "each", "other",
    "every", "mention", "mentions", "discuss", "discusses", "give", "gives", "cover", "covers",
}


def _topic_query(question: str) -> str:
    """Strip comparison/filler words so search matches on the topic, not on 'documents agree'."""
    words = re.findall(r"\w+", question)
    kept = [w for w in words
            if w.lower() not in _QUERY_FILLER and not _COMPARE_RE.fullmatch(w)]
    cleaned = " ".join(kept)
    return cleaned if len(cleaned) >= 4 else question  # nothing useful left -> keep original


def _per_file_docs(index, question: str):
    """Best few passages from EACH uploaded file (hybrid search + rerank within each file)."""
    search_q = _topic_query(question)
    print(f"  -> Per-file search query: {search_q!r}")
    q = np.array(embed_model.get_query_embedding(search_q), dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    sims = index.embeddings @ q
    bm = index.bm25.get_scores(tokenize(search_q))

    by_file = {}
    for i, lid in enumerate(index.leaf_ids):             # leaf ids look like "f0-p3-c150"
        by_file.setdefault(lid.split("-p")[0], []).append(i)

    docs = []
    for idxs in by_file.values():
        vec_top = sorted(idxs, key=lambda i: sims[i], reverse=True)[:COMPARE_CANDIDATES_PER_FILE]
        kw_top = [i for i in sorted(idxs, key=lambda i: bm[i], reverse=True)[:COMPARE_CANDIDATES_PER_FILE] if bm[i] > 0]
        cand = list(dict.fromkeys(vec_top + kw_top))
        scores = np.array(cross_encoder.predict(
            [[search_q, index.leaf_to_text[index.leaf_ids[i]]] for i in cand]))
        seen_pages = set()  # one passage per page, so repeated chunks of one page don't fill the slots
        for j in np.argsort(scores)[::-1]:
            lid = index.leaf_ids[cand[j]]
            page_id = index.leaf_to_parent[lid]
            if page_id in seen_pages:
                continue
            seen_pages.add(page_id)
            header = index.parent_texts[page_id].split("\n", 1)[0]
            docs.append(f"{header}\n{index.leaf_to_text[lid]}")
            if len(seen_pages) >= COMPARE_PASSAGES_PER_FILE:
                break
    return docs


def _overview_docs(index):
    """Evenly sampled pages from EACH uploaded file, so a summary covers every document."""
    by_file = {}
    for pid in index.parent_texts:                      # ids look like "f0-p3"
        by_file.setdefault(pid.split("-p")[0], []).append(pid)
    docs = []
    for pids in by_file.values():
        k = min(OVERVIEW_PAGES_PER_FILE, len(pids))
        picks = sorted({round(i * (len(pids) - 1) / max(k - 1, 1)) for i in range(k)})
        docs += [index.parent_texts[pids[i]][:OVERVIEW_CHARS_PER_PAGE] for i in picks]
    return docs


def _front_docs(index):
    """First page of EACH uploaded file (title, authors, abstract live there)."""
    by_file = {}
    for pid in index.parent_texts:                      # ids look like "f0-p3"
        by_file.setdefault(pid.split("-p")[0], []).append(pid)
    return [index.parent_texts[pids[0]][:OVERVIEW_CHARS_PER_PAGE] for pids in by_file.values()]


def _library_candidates(question: str):
    """Vector (Chroma) + BM25 candidates from the permanent library index."""
    query_embedding = embed_model.get_query_embedding(question)
    chroma_results = chroma_collection.query(
        query_embeddings=[query_embedding], n_results=TOP_K
    )
    vector_ids = chroma_results["ids"][0] if chroma_results["ids"] else []

    bm25_scores = bm25.get_scores(tokenize(question))
    top_idx = np.argsort(bm25_scores)[::-1][:TOP_K]
    keyword_ids = [leaf_ids[i] for i in top_idx if bm25_scores[i] > 0]
    return vector_ids + keyword_ids


def _upload_candidates(index, question: str):
    """Vector (in-memory cosine) + BM25 candidates from the uploaded-PDF index."""
    q = np.array(embed_model.get_query_embedding(question), dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    sims = index.embeddings @ q
    vector_ids = [index.leaf_ids[i] for i in np.argsort(sims)[::-1][:TOP_K]]

    bm25_scores = index.bm25.get_scores(tokenize(question))
    top_idx = np.argsort(bm25_scores)[::-1][:TOP_K]
    keyword_ids = [index.leaf_ids[i] for i in top_idx if bm25_scores[i] > 0]
    return vector_ids + keyword_ids


# --- NODE 1: HYBRID RETRIEVER (leaf search -> rerank -> parent context) ---
def retrieve(state: GraphState, config: RunnableConfig):
    print("\n[Node] Running Hybrid Retrieval...")
    question = state["question"]
    index = _get_index(config)

    kind = _request_kind(index, question)
    if kind == "overview":
        docs = _overview_docs(index)
        print(f"  -> Overview request: sampled {len(docs)} pages across {len(index.files)} file(s).")
        return {"documents": docs, "question": question}
    if kind == "front":
        docs = _front_docs(index)
        print(f"  -> Front-page request: first page of {len(docs)} file(s).")
        return {"documents": docs, "question": question}
    if kind == "comparison":
        docs = _per_file_docs(index, question)
        print(f"  -> Comparison request: {len(docs)} passages from {len(index.files)} files.")
        return {"documents": docs, "question": question}

    if index is None:
        raw_ids = _library_candidates(question)
        text_of, parent_of, parents = leaf_to_text, leaf_to_parent, parent_texts
    else:
        print("  -> Searching your uploaded PDFs.")
        raw_ids = _upload_candidates(index, question)
        text_of, parent_of, parents = index.leaf_to_text, index.leaf_to_parent, index.parent_texts

    # Dedupe while preserving order; drop ids missing from the maps (stale index)
    candidate_ids = [i for i in dict.fromkeys(raw_ids) if i in text_of]
    if not candidate_ids:
        print("  -> No candidates found.")
        return {"documents": [], "question": question}

    # Rerank the small leaf chunks (they fit the cross-encoder's 512-token limit)
    pairs = [[question, text_of[i]] for i in candidate_ids]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, candidate_ids), key=lambda x: x[0], reverse=True)
    top_leaf_ids = [leaf_id for _, leaf_id in ranked[:TOP_N_RERANK]]

    # Swap each winning leaf for its parent (richer context), deduped by parent
    parent_ids = list(dict.fromkeys(parent_of[i] for i in top_leaf_ids))
    top_docs = [parents[p] for p in parent_ids]
    print(f"  -> {len(top_leaf_ids)} leaves -> {len(top_docs)} unique parent chunks.")

    return {"documents": top_docs, "question": question}


# --- NODE 2: CRAG GRADER (single batched call) ---
def grade_documents(state: GraphState, config: RunnableConfig):
    print("[Node] Grading retrieved documents...")
    question = state["question"]
    documents = state["documents"]
    # Web fallback only for the library; uploaded-PDF mode answers from the PDFs alone.
    index = _get_index(config)
    library_mode = index is None

    if _request_kind(index, question):
        print("  -> Summary/comparison request: skipping relevance grading.")
        return {"documents": documents, "web_search": "No", "question": question}

    if not documents:
        return {"documents": [], "web_search": "Yes" if library_mode else "No", "question": question}

    grader_config = types.GenerateContentConfig(
        system_instruction=(
            "You are a strict grader assessing relevance of retrieved documents to a user question. "
            "For each numbered document, grade it 'yes' if it contains keywords or meaning related "
            "to the question, otherwise 'no'. "
            'Respond ONLY with JSON in this exact shape: {"grades": ["yes", "no", ...]} '
            "with exactly one entry per document, in order."
        ),
        response_mime_type="application/json",
        temperature=0,
        http_options=types.HttpOptions(timeout=60000),  # 60s in ms
    )

    numbered = "\n\n".join(f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(documents))
    response = call_llm(f"{numbered}\n\nQuestion: {question}", grader_config)

    try:
        grades = json.loads(response.text or "{}").get("grades", [])
        grades = [str(g).strip().lower() for g in grades]
    except (json.JSONDecodeError, AttributeError):
        print("  -> Could not parse grader output; treating all documents as irrelevant.")
        grades = []

    valid_docs = [
        doc for doc, grade in zip(documents, grades) if grade.startswith("yes")
    ]
    for i, grade in enumerate(grades):
        print(f"  -> Chunk {i + 1}/{len(documents)}: {grade}")

    web_search_needed = "Yes" if (not valid_docs and library_mode) else "No"
    return {"documents": valid_docs, "web_search": web_search_needed, "question": question}


# --- NODE 3: WEB SEARCH ---
def web_search(state: GraphState):
    print("[Node] Local docs failed. Triggering Tavily Web Search...")
    question = state["question"]
    documents = list(state["documents"])  # copy; don't mutate state in place

    if os.environ.get("MOCK_LLM") == "1":
        print("  -> Mock mode is on; skipping Tavily web search.")
        return {"documents": documents, "question": question}

    try:
        results = tavily_client.search(query=question, max_results=3)
        web_text = "\n".join(d["content"] for d in results.get("results", []))
        if web_text:
            documents.append("WEB SEARCH RESULTS:\n" + web_text)
    except Exception as e:
        print(f"  -> Tavily search failed: {str(e)[:100]}")

    return {"documents": documents, "question": question}


def _label_upload_excerpts(documents):
    """Return model-safe excerpt labels and their source locations."""
    labeled_excerpts = []
    excerpt_sources = {}
    source_groups = {}
    source_header = re.compile(r"^\[Source: (.+), page (\d+)\]\s*")

    for number, document in enumerate(documents, 1):
        excerpt_id = f"E{number}"
        match = source_header.match(document)
        if match:
            file_name = match.group(1)
            excerpt_sources[excerpt_id] = (file_name, int(match.group(2)))
            source_group = source_groups.setdefault(file_name, len(source_groups) + 1)
            excerpt = document[match.end():]
        else:
            source_group = number
            excerpt = document
        labeled_excerpts.append(
            f"[{excerpt_id}] (source group {source_group})\n{excerpt}"
        )

    group_files = {number: name for name, number in source_groups.items()}
    return "\n\n".join(labeled_excerpts), excerpt_sources, group_files


_UPLOAD_TAG_RANGE = r"E\d+(?:\s*[-‐-―−]\s*E?\d+)?"
_UPLOAD_TAG_LIST = rf"{_UPLOAD_TAG_RANGE}(?:\s*,\s*{_UPLOAD_TAG_RANGE})*"
_UPLOAD_CITATION_RE = re.compile(
    rf"\[(?P<square>{_UPLOAD_TAG_LIST})\]"
    rf"|【(?P<wide>{_UPLOAD_TAG_LIST})】"
    rf"|\((?P<paren>{_UPLOAD_TAG_LIST})\)"
    rf"|(?<![A-Za-z0-9])(?P<bare>{_UPLOAD_TAG_LIST})(?![A-Za-z0-9])"
)
_RANGE_PARTS_RE = re.compile(r"E(\d+)(?:\s*[-‐-―−]\s*E?(\d+))?")
# "Source 1 (E1-E4):", "**Source 2 (E5-E8)**:", "(source group 3)"
_GROUP_LABEL_RE = re.compile(
    rf"\*{{0,2}}\bSource\s+\d+\s*\(\s*{_UPLOAD_TAG_LIST}\s*\)\*{{0,2}}\s*[:–—-]?\s*",
    re.I,
)
_GROUP_MARKER_RE = re.compile(r"\(\s*source group \d+\s*\)", re.I)


def _resolve_upload_citations(answer, excerpt_sources, group_files=None):
    """Resolve [E#] tags to real file/page, merge adjacent citations, drop leftovers."""
    answer = _GROUP_MARKER_RE.sub("", answer)
    answer = _GROUP_LABEL_RE.sub("", answer)

    source_by_number = {
        int(tag[1:]): src for tag, src in excerpt_sources.items()
        if re.fullmatch(r"E\d+", tag)
    }
    matches = list(_UPLOAD_CITATION_RE.finditer(answer))
    replacements = []
    i = 0
    while i < len(matches):
        cluster = [matches[i]]
        end = matches[i].end()
        i += 1
        # citations separated only by spaces/commas belong to one group
        while i < len(matches) and not answer[end:matches[i].start()].strip(" \t\r\n,"):
            cluster.append(matches[i])
            end = matches[i].end()
            i += 1

        pages_by_file = {}
        for m in cluster:
            tag_list = next(v for v in m.groupdict().values() if v is not None)
            for tag in re.findall(_UPLOAD_TAG_RANGE, tag_list):
                parts = _RANGE_PARTS_RE.fullmatch(tag)
                first = int(parts.group(1))
                last = int(parts.group(2) or first)
                lo, hi = min(first, last), max(first, last)
                for n in sorted(k for k in source_by_number if lo <= k <= hi):
                    file_name, page = source_by_number[n]
                    pages = pages_by_file.setdefault(file_name, [])
                    if page not in pages:
                        pages.append(page)

        cites = []
        for file_name, pages in pages_by_file.items():
            label = "page" if len(pages) == 1 else "pages"
            cites.append(f"[{file_name}, {label} {', '.join(map(str, sorted(pages)))}]")
        replacements.append((cluster[0].start(), end, " ".join(cites)))

    for start, end, text in reversed(replacements):
        if text and start > 0 and answer[start - 1].isalnum():
            text = " " + text  # keep "Ali [file.pdf, page 1]" instead of "Ali[file.pdf, page 1]"
        answer = answer[:start] + text + answer[end:]

    # safety net: only standalone uppercase tags (never touches base64, Route53, E2E ...)
    answer = re.sub(r"(?<![A-Za-z0-9])E\d+(?![A-Za-z0-9])", "", answer)
    # The model sometimes labels files "Source group 2" / "Group 2" (often in tables).
    gf = group_files or {}
    answer = re.sub(r"(?i)\s*\(\s*(?:source\s+)?groups?\s*\)", "", answer)
    # label directly followed by a citation: drop the label, the citation already names the file
    answer = re.sub(
        r"(?i)\b(?:source\s+)?group\s+(\d+)\b\s*[\u2013\u2014-]?\s*(?=\[)",
        lambda m: "" if int(m.group(1)) in gf else m.group(0),
        answer,
    )
    # any other use (e.g. "Group 2 | not in the excerpts"): show the real file name
    answer = re.sub(
        r"(?i)\b(?:source\s+)?group\s+(\d+)\b",
        lambda m: gf.get(int(m.group(1)), m.group(0)),
        answer,
    )
    return re.sub(r"\[\s*\]|【\s*】|\(\s*\)", "", answer)


def _strip_model_html(text: str) -> str:
    """Turn stray list/paragraph/bold HTML the model puts in tables into plain text.

    The prompt says 'plain Markdown only', but models still emit <ul><li>...</li></ul> inside table
    cells. Those show up as literal tags in Streamlit and React, so convert them to bullets.
    """
    text = re.sub(r"(?i)<li\b[^>]*>\s*", "\u2022 ", text)
    text = re.sub(r"(?i)</li\s*>", " ", text)
    return re.sub(r"(?i)</?(?:ul|ol|p|strong|em|b|i)\b[^>]*>", "", text)


# --- NODE 4: GENERATOR ---
def generate(state: GraphState, config: RunnableConfig):
    print("[Node] Generating final answer...")
    question = state["question"]
    documents = state["documents"]
    retries = state.get("retries", 0)

    # NEW: nothing relevant survived grading -> answer with a fixed message, no LLM call
    if not documents:
        where = "the library" if _get_index(config) is None else "your uploaded PDFs"
        print("  -> No documents to answer from; returning a fixed 'not found' message.")
        return {
            "generation": f"No relevant passage was found in {where} for this question.",
            "question": question,
        }

    uploads_mode = _get_index(config) is not None
    if uploads_mode:
        context, excerpt_sources, group_files = _label_upload_excerpts(documents)
    else:
        context = "\n\n".join(documents)
        excerpt_sources, group_files = {}, {}

    if not uploads_mode:
        instruction = (
            "You are a senior software architect. "
            "Answer the user's question comprehensively using only the provided context. "
            "If the context does not contain enough information, say so clearly instead of guessing. "
            "Use plain Markdown only; never use HTML tags such as <br>."
        )
    else:
        instruction = (
            "You are a careful research assistant. "
            "Answer the user's question using only the provided document excerpts. "
            "The source-group markers in the context identify excerpts from the same file. "
            "Use them only to track file coverage. Never reproduce source-group markers, group labels such as "
            "'Source 1', or excerpt tags except as citations. "
            "Cite key claims using only the excerpt tag [E#] shown above each excerpt. "
            "Never write source file names or page numbers yourself; citations will be resolved automatically. "
            "If the excerpts do not contain enough information, say so clearly instead of guessing. "
            "Never say a document lacks or does not mention something just because it is absent from the excerpts; "
            "say instead that it is not in the excerpts you saw. "
            "If asked to summarize, start with the note 'Based on selected excerpts, not the full documents.' "
            "then give a separate short summary for each source file, ending each summary with the excerpt tags that support it. "
            "If asked to compare or contrast documents, cover EVERY source file that appears in the excerpts, "
            "separating (1) points that at least two documents clearly state, "
            "citing each one, from (2) points where they differ or only one document covers the topic. "
            "If a document's excerpts say nothing about the point, write 'cannot be determined from the excerpts' for it, "
            "and never conclude that documents disagree just because one of them is silent. "
            "Never call something an agreement unless at least two documents support it, and if the excerpts show "
            "no disagreement, say so instead of inventing one. "
            "Only use quotation marks for text that appears verbatim in the excerpts. "
            "Be concise: do not repeat the same facts in several formats (for example a per-file list and a table), "
            "and keep the answer under about 400 words unless asked for more detail. "
            "When a point is missing, write 'not in the excerpts', never 'not explicitly discussed' or 'none'. "
            "For questions like 'which documents mention X', cover EVERY source file and mark each as "
            "'mentions it' with the relevant excerpt tag(s), or 'not in the excerpts'. "
            "Use plain Markdown only; never use HTML tags such as <br>."
        )
    if retries > 0:
        instruction += (
            " IMPORTANT: a previous draft contained claims not supported by the context. "
            "Include only statements that are directly supported by the context."
        )

    generator_config = types.GenerateContentConfig(
        system_instruction=instruction,
        temperature=0.2,
        http_options=types.HttpOptions(timeout=60000),  # 60s in ms
    )

    response = call_llm(
        f"Context: {context}\n\nQuestion: {question}", generator_config
    )
    text = re.sub(r"\s*<br\s*/?>\s*", " · ", response.text or "")
    text = _strip_model_html(text)
    if uploads_mode:
        text = _resolve_upload_citations(text, excerpt_sources, group_files)
    return {"generation": text, "question": question}


# --- NODE 5: HALLUCINATION CRITIC ---
def grade_generation(state: GraphState):
    print("[Node] Checking answer for hallucinations...")

    # NEW: the fixed 'not found' message makes no claims, so there is nothing to verify
    if not state["documents"]:
        print("  -> No documents; skipping critic call.")
        return {"grounded": "yes", "retries": state.get("retries", 0) + 1}

    config = types.GenerateContentConfig(
        system_instruction=(
            "You verify whether an answer is fully supported by the provided context. "
            "Reply with a single word: 'yes' if every claim is supported, otherwise 'no'."
        ),
        temperature=0,
        http_options=types.HttpOptions(timeout=60000),
    )

    context = "\n\n".join(state["documents"])
    response = call_llm(
        f"Context: {context}\n\nAnswer: {state['generation']}", config
    )
    verdict = (response.text or "").strip().lower()
    grounded = "yes" if verdict.startswith("yes") else "no"
    print(f"  -> Grounded: {grounded}")

    return {"grounded": grounded, "retries": state.get("retries", 0) + 1}