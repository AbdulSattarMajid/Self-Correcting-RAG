"""Build a session-scoped, in-memory search index from user-uploaded PDFs.

Nothing here is written to disk: the index lives in Streamlit's session state and
disappears with the session, so uploads never pollute the permanent library index.

Chunking mirrors the library index: each PDF page is a "parent" chunk (given to the
generator, so answers can cite file + page), split into small overlapping "leaf" chunks
(used for vector + BM25 search and reranking).
"""
import io
from dataclasses import dataclass

import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from common import tokenize

MAX_FILES = 3
MAX_PAGES = 20
MAX_FILE_MB = 10
CHILD_WORDS = 180   # roughly 256 tokens
CHILD_STEP = 150    # 30-word overlap between neighbouring leaves


class UploadError(ValueError):
    """Raised with a user-friendly message when an upload can't be accepted."""


@dataclass
class SessionIndex:
    leaf_ids: list
    leaf_to_text: dict
    leaf_to_parent: dict
    parent_texts: dict
    embeddings: np.ndarray   # (n_leaves, dim), L2-normalised
    bm25: BM25Okapi
    files: list              # [(filename, page_count), ...]


def build_session_index(files, embed_fn) -> SessionIndex:
    """files: list of (filename, raw_bytes). embed_fn: list[str] -> list[list[float]]."""
    if not files:
        raise UploadError("Upload at least one PDF.")
    if len(files) > MAX_FILES:
        raise UploadError(f"You can upload at most {MAX_FILES} PDFs at a time.")

    leaf_ids, leaf_texts = [], []
    leaf_to_parent, parent_texts, file_info = {}, {}, []

    for f_idx, (name, data) in enumerate(files):
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            raise UploadError(f"'{name}' is larger than {MAX_FILE_MB} MB.")
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise UploadError(f"'{name}' is password-protected.")
            n_pages = len(reader.pages)
        except UploadError:
            raise
        except Exception:
            raise UploadError(f"'{name}' could not be read as a PDF.")
        if n_pages > MAX_PAGES:
            raise UploadError(f"'{name}' has {n_pages} pages; the limit is {MAX_PAGES}.")

        for page_no, page in enumerate(reader.pages, 1):
            try:
                text = " ".join((page.extract_text() or "").split())
            except Exception:
                text = ""
            if not text:
                continue

            parent_id = f"f{f_idx}-p{page_no}"
            parent_texts[parent_id] = f"[Source: {name}, page {page_no}]\n{text}"

            words = text.split()
            for start in range(0, len(words), CHILD_STEP):
                leaf_id = f"{parent_id}-c{start}"
                leaf_ids.append(leaf_id)
                leaf_texts.append(" ".join(words[start:start + CHILD_WORDS]))
                leaf_to_parent[leaf_id] = parent_id
                if start + CHILD_WORDS >= len(words):
                    break
        file_info.append((name, n_pages))

    if not leaf_ids:
        raise UploadError(
            "No extractable text found. Scanned/image-only PDFs need OCR, which isn't supported."
        )

    emb = np.array(embed_fn(leaf_texts), dtype=float)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    return SessionIndex(
        leaf_ids=leaf_ids,
        leaf_to_text=dict(zip(leaf_ids, leaf_texts)),
        leaf_to_parent=leaf_to_parent,
        parent_texts=parent_texts,
        embeddings=emb,
        bm25=BM25Okapi([tokenize(t) for t in leaf_texts]),
        files=file_info,
    )