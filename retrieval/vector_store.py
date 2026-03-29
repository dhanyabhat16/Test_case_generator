"""
vector_store.py
---------------
Wrapper around Person 1's FAISS store (testgen_store.index + testgen_store_meta.pkl).
Provides load and query operations with optional metadata filtering by input type.

Usage:
    from retrieval.vector_store import VectorStore
    vs = VectorStore()
    results = vs.query("validate email input", n_results=5, filter_type="code")
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Mirror Person 1's path constants exactly ──────────────────────────────────
# embedder.py sets FAISS_DB_PATH relative to the ingestion/ folder.
# From retrieval/, we go up one level to the project root, then into faiss_db/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

FAISS_DB_PATH = os.path.join(_PROJECT_ROOT, "faiss_db")
INDEX_FILE    = os.path.join(FAISS_DB_PATH, "testgen_store.index")
META_FILE     = os.path.join(FAISS_DB_PATH, "testgen_store_meta.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # must match Person 1's embedder
VECTOR_DIM      = 384                   # all-MiniLM-L6-v2 output dimension


class VectorStore:
    """
    Read-only interface to the FAISS vector store built by Person 1's embedder.

    Attributes:
        index    : The loaded FAISS index (IndexFlatIP).
        records  : List of dicts — each has 'id', 'content', 'metadata'.
        embedder : SentenceTransformer model for encoding query strings.
    """

    def __init__(self) -> None:
        self.index, self.records = self._load_store()
        self.embedder = self._load_embedder()
        print(
            f"[VectorStore] Loaded {len(self.records)} chunk(s) "
            f"from '{FAISS_DB_PATH}'"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        n_results: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the FAISS store for chunks similar to `text`.

        Args:
            text        : The query string (e.g. a function body or requirement).
            n_results   : How many results to return (default 5).
            filter_type : If set, only return chunks whose metadata['type']
                          matches this value ("code", "requirements", "api").

        Returns:
            List of result dicts, each containing:
                {
                    "id"       : str,   # chunk ID from embedder
                    "content"  : str,   # raw text of the chunk
                    "metadata" : dict,  # flat metadata (lists parsed back from JSON)
                    "score"    : float, # cosine similarity score (higher = better)
                }
            Sorted by score descending.
        """
        if not text or not text.strip():
            return []

        if len(self.records) == 0:
            print("[VectorStore] Store is empty — run the embedder first.")
            return []

        # Encode and normalise the query vector (must match Person 1's normalisation)
        query_vec = self._encode_and_normalise(text)

        # ── If filtering by type, we can't use FAISS's built-in filter, so we
        #    fetch more candidates and filter afterwards ──────────────────────
        fetch_k = n_results * 10 if filter_type else n_results
        fetch_k = min(fetch_k, len(self.records))  # can't fetch more than we have

        scores, indices = self.index.search(query_vec, fetch_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.records):
                continue  # FAISS returns -1 for empty slots

            record   = self.records[idx]
            metadata = self._parse_metadata(record["metadata"])

            # Apply type filter
            if filter_type and metadata.get("type") != filter_type:
                continue

            results.append(
                {
                    "id"      : record["id"],
                    "content" : record["content"],
                    "metadata": metadata,
                    "score"   : float(score),
                }
            )

            if len(results) >= n_results:
                break

        return results

    def get_all_by_type(self, filter_type: str) -> List[Dict[str, Any]]:
        """
        Return every chunk of a given type without vector search.
        Useful for evaluation or bulk processing.

        Args:
            filter_type: "code", "requirements", or "api"

        Returns:
            List of record dicts (id, content, metadata).
        """
        results = []
        for record in self.records:
            meta = self._parse_metadata(record["metadata"])
            if meta.get("type") == filter_type:
                results.append(
                    {
                        "id"      : record["id"],
                        "content" : record["content"],
                        "metadata": meta,
                    }
                )
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary of what's currently in the store."""
        type_counts: Dict[str, int] = {}
        file_counts: Dict[str, int] = {}

        for record in self.records:
            meta      = record.get("metadata", {})
            rtype     = str(meta.get("type", "unknown"))
            file_name = str(meta.get("file_name", "unknown"))

            type_counts[rtype]     = type_counts.get(rtype, 0) + 1
            file_counts[file_name] = file_counts.get(file_name, 0) + 1

        return {
            "total_chunks": len(self.records),
            "by_type"     : type_counts,
            "by_file"     : file_counts,
            "index_size"  : self.index.ntotal if self.index else 0,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_store(self) -> Tuple[Any, List[Dict[str, Any]]]:
        """Load FAISS index + pickle metadata from disk."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Run: pip install faiss-cpu"
            ) from exc

        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError(
                f"FAISS index not found at '{INDEX_FILE}'.\n"
                "Run Person 1's embedder first:\n"
                "  python -m ingestion.embedder ingest <file>"
            )

        if not os.path.exists(META_FILE):
            raise FileNotFoundError(
                f"Metadata file not found at '{META_FILE}'.\n"
                "Run Person 1's embedder first."
            )

        index = faiss.read_index(INDEX_FILE)

        with open(META_FILE, "rb") as f:
            records = pickle.load(f)

        return index, records

    def _load_embedder(self):
        """Load the same sentence-transformer model Person 1 used."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Run: pip install sentence-transformers"
            ) from exc

        return SentenceTransformer(EMBEDDING_MODEL)

    def _encode_and_normalise(self, text: str) -> np.ndarray:
        """Encode a query string and normalise to unit length (matches IndexFlatIP)."""
        vector = self.embedder.encode(text, convert_to_numpy=True).astype("float32")
        norm   = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.reshape(1, -1)  # FAISS expects shape (1, dim)

    @staticmethod
    def _parse_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Person 1's embedder stores lists/dicts as JSON strings.
        Parse them back so callers get real Python objects.
        """
        parsed: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                # Try to decode JSON strings back to lists/dicts
                stripped = value.strip()
                if stripped.startswith(("{", "[")):
                    try:
                        parsed[key] = json.loads(stripped)
                        continue
                    except (json.JSONDecodeError, ValueError):
                        pass
            parsed[key] = value
        return parsed


# ── Module-level convenience singleton ────────────────────────────────────────
# Import this in retriever.py so the model is only loaded once.
_store_instance: Optional[VectorStore] = None


def get_store() -> VectorStore:
    """Return a module-level singleton VectorStore (lazy init)."""
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore()
    return _store_instance


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    store = VectorStore()
    stats = store.get_stats()
    print("\n=== FAISS Store Stats ===")
    print(f"  Total chunks : {stats['total_chunks']}")
    print(f"  By type      : {stats['by_type']}")
    print(f"  By file      : {stats['by_file']}")
    print(f"  Index size   : {stats['index_size']}")

    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        print(f"\n=== Query: '{query_text}' ===")
        results = store.query(query_text, n_results=3)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] ID    : {r['id']}")
            print(f"    Score : {r['score']:.4f}")
            print(f"    Type  : {r['metadata'].get('type')}")
            print(f"    File  : {r['metadata'].get('file_name')}")
            print(f"    Content preview: {r['content'][:120]}...")
