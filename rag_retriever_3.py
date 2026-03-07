# rag_retriever_3.py
# SentenceTransformer (all-MiniLM-L6-v2) + FAISS
# Real semantic search — "nervous" finds "anxiety", "freaking out" finds "panic"
# Requires: pip install sentence-transformers faiss-cpu
#           pip install torch --index-url https://download.pytorch.org/whl/cpu
# Fails silently — never breaks the chat flow.

import os
import pickle
from typing import Optional

DOCS_PATH = "rag_docs/"
INDEX_PATH = "rag_docs/faiss_index.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.30  # cosine similarity: 0 = unrelated, 1 = identical

_model = None
_index = None
_chunks = []


# ─────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────

def _get_model():
    """Lazy-load SentenceTransformer once per session."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("[RAG] Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[RAG] Model loaded.")
    return _model


def _extract_text_from_pdf(filepath: str) -> str:
    """Extract plain text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        print(f"[RAG] PDF read failed for {filepath}: {e}")
        return ""


def _collect_chunks() -> list:
    """Read all PDFs in rag_docs/ and return paragraph chunks."""
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)

    chunks = []
    for fname in sorted(os.listdir(DOCS_PATH)):
        if not fname.endswith(".pdf"):
            continue
        fpath = os.path.join(DOCS_PATH, fname)
        text = _extract_text_from_pdf(fpath)
        if not text:
            continue
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        chunks.extend(paragraphs)
        print(f"[RAG] Loaded {len(paragraphs)} chunks from {fname}")
    return chunks


def _load_or_build():
    """Load cached FAISS index or build from PDFs."""
    global _index, _chunks

    if _index is not None:
        return  # Already loaded this session

    # Try loading from cache
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            _index = data["index"]
            _chunks = data["chunks"]
            print(f"[RAG] Loaded index from cache: {len(_chunks)} chunks.")
            return
        except Exception as e:
            print(f"[RAG] Cache load failed, rebuilding: {e}")

    _build_index()


def _build_index():
    """Embed all chunks and build FAISS index."""
    global _index, _chunks

    import faiss
    import numpy as np

    _chunks = _collect_chunks()

    if not _chunks:
        print("[RAG] No content found in rag_docs/. Retriever dormant.")
        return

    model = _get_model()
    print(f"[RAG] Embedding {len(_chunks)} chunks...")
    embeddings = model.encode(_chunks, convert_to_numpy=True, show_progress_bar=False)

    # Normalise so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)  # Inner Product on normalised vectors = cosine
    _index.add(embeddings)

    # Cache to disk
    try:
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"index": _index, "chunks": _chunks}, f)
        print(f"[RAG] Index built and cached: {len(_chunks)} chunks.")
    except Exception as e:
        print(f"[RAG] Cache save failed (will rebuild next run): {e}")


# ─────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────

def rebuild_index():
    """
    Call this after adding or replacing PDFs in rag_docs/.
    Deletes the cache and rebuilds from scratch.
    """
    global _index, _chunks
    _index = None
    _chunks = []
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
        print("[RAG] Cache cleared.")
    _build_index()
    print("[RAG] Index rebuilt successfully.")


def retrieve(query: str, k: int = 2) -> Optional[str]:
    """
    Returns up to k semantically relevant chunks as a single string, or None.
    Always fails silently — never breaks the chat flow.
    """
    try:
        _load_or_build()

        if _index is None or not _chunks:
            return None

        import faiss
        import numpy as np

        model = _get_model()
        query_vec = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        scores, indices = _index.search(query_vec, k)

        # scores[0] are cosine similarities (0 to 1 after normalisation)
        relevant = [
            _chunks[idx]
            for score, idx in zip(scores[0], indices[0])
            if idx < len(_chunks) and score >= SIMILARITY_THRESHOLD
        ]

        if not relevant:
            return None

        return "\n".join(relevant[:2])

    except Exception as e:
        print(f"[RAG] Retrieval error (silent): {e}")
        return None