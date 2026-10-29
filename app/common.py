"""Shared paths and helpers used by the ingest, BM25 and graph code."""
import os
import re

# Resolve everything relative to this file, so scripts work from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../app
ROOT_DIR = os.path.dirname(BASE_DIR)                    # project root

DATA_DIR = os.path.join(ROOT_DIR, "data")
CHROMA_DIR = os.path.join(ROOT_DIR, "chroma_db")
STORAGE_DIR = os.path.join(ROOT_DIR, "local_storage")
BM25_PATH = os.path.join(STORAGE_DIR, "bm25_index.pkl")

COLLECTION_NAME = "architecture_docs"


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with punctuation stripped.

    Used for BOTH building the BM25 index and querying it, so they always match
    ("microservices," and "microservices" become the same token).
    """
    return re.findall(r"\w+", text.lower())