"""
embedder.py
-----------
Embeds parsed chunks using sentence-transformers and stores them
in a persistent FAISS vector store.

Usage:
    from ingestion.embedder import embed_and_store, clear_store, ingest_file
    chunk_ids = embed_and_store(chunks)
    chunk_ids = ingest_file("path/to/file.py")
    clear_store()
"""

from __future__ import annotations

import json
import os
import pickle
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

# Persistent FAISS directory
FAISS_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_db")
INDEX_FILE = os.path.join(FAISS_DB_PATH, "testgen_store.index")
META_FILE = os.path.join(FAISS_DB_PATH, "testgen_store_meta.pkl")
COLLECTION_NAME = "testgen_store"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def embed_and_store(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Embed a list of chunk dicts and persist them in FAISS.

    Args:
        chunks: List of parser chunks that match SCHEMA.md.

    Returns:
        List of generated chunk IDs.
    """
    if not chunks:
        print("No chunks provided. Nothing stored.")
        return []

    _ensure_db_dir()
    embedder = _get_embedder()
    index, records = _load_store()

    ids: List[str] = []
    new_vectors: List[np.ndarray] = []

    print(f"Embedding {len(chunks)} chunk(s) using {EMBEDDING_MODEL}...")

    for chunk in chunks:
        content = str(chunk.get("content", "")).strip()
        metadata = chunk.get("metadata", {})

        if not content:
            continue

        chunk_id = _make_chunk_id(metadata)
        flat_metadata = _flatten_metadata(metadata)

        vector = embedder.encode(content, convert_to_numpy=True)
        vector = _normalize_vector(vector)

        ids.append(chunk_id)
        new_vectors.append(vector)
        records.append(
            {
                "id": chunk_id,
                "content": content,
                "metadata": flat_metadata,
            }
        )

    if new_vectors:
        matrix = np.vstack(new_vectors).astype("float32")
        index.add(matrix)
        _save_store(index, records)

        print(f"Stored {len(new_vectors)} chunk(s) in FAISS store '{COLLECTION_NAME}'")
        print(f"DB path: {os.path.abspath(FAISS_DB_PATH)}")

    return ids


def clear_store() -> None:
    """Delete all vectors and metadata from the FAISS store."""
    _ensure_db_dir()
    index = _create_index()
    records: List[Dict[str, Any]] = []
    _save_store(index, records)
    print(f"Cleared all chunk(s) from '{COLLECTION_NAME}'")


def get_store_stats() -> Dict[str, Any]:
    """Return aggregate stats for the FAISS store."""
    _ensure_db_dir()
    _, records = _load_store()

    total = len(records)
    type_counts: Dict[str, int] = {}
    file_counts: Dict[str, int] = {}

    for item in records:
        meta = item.get("metadata", {})
        input_type = str(meta.get("type", "unknown"))
        file_name = str(meta.get("file_name", "unknown"))

        type_counts[input_type] = type_counts.get(input_type, 0) + 1
        file_counts[file_name] = file_counts.get(file_name, 0) + 1

    return {
        "total_chunks": total,
        "by_type": type_counts,
        "by_file": file_counts,
        "collection_name": COLLECTION_NAME,
        "db_path": os.path.abspath(FAISS_DB_PATH),
    }


def ingest_file(file_path: str) -> List[str]:
    """
    Auto-detect file type, parse, embed, and store in one call.

    Args:
        file_path: Path to a supported input file.

    Returns:
        List of stored chunk IDs.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".py", ".java"):
        from ingestion.code_parser import parse_code_file

        chunks = parse_code_file(file_path)
        input_type = "code"
    elif ext in (".pdf", ".docx"):
        from ingestion.doc_parser import parse_doc_file

        chunks = parse_doc_file(file_path)
        input_type = "requirements"
    elif ext in (".yaml", ".yml", ".json"):
        from ingestion.api_parser import parse_api_file

        chunks = parse_api_file(file_path)
        input_type = "api"
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            "Supported: .py, .java, .pdf, .docx, .yaml, .yml, .json"
        )

    print(f"Parsed {len(chunks)} chunk(s) from '{os.path.basename(file_path)}' [{input_type}]")
    return embed_and_store(chunks)


def _ensure_db_dir() -> None:
    os.makedirs(FAISS_DB_PATH, exist_ok=True)


def _create_index():
    """Create an empty cosine-similarity FAISS index."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required. Run: pip install faiss-cpu") from exc

    # all-MiniLM-L6-v2 outputs 384-dimensional vectors
    return faiss.IndexFlatIP(384)


def _load_store() -> Tuple[Any, List[Dict[str, Any]]]:
    """Load FAISS index and metadata records from disk."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required. Run: pip install faiss-cpu") from exc

    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = _create_index()

    if os.path.exists(META_FILE):
        with open(META_FILE, "rb") as f:
            records = pickle.load(f)
    else:
        records = []

    return index, records


def _save_store(index: Any, records: List[Dict[str, Any]]) -> None:
    """Persist FAISS index and metadata records to disk."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required. Run: pip install faiss-cpu") from exc

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(records, f)


def _get_embedder():
    """Load the sentence-transformer embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Run: pip install sentence-transformers"
        ) from exc

    return SentenceTransformer(EMBEDDING_MODEL)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector for cosine similarity with IndexFlatIP."""
    vector = vector.astype("float32")
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _make_chunk_id(metadata: Dict[str, Any]) -> str:
    """
    Generate a unique ID for a chunk.
    Format: {type}_{file}_{label}_{uuid8}
    """
    input_type = metadata.get("type", "unknown")
    file_name = str(metadata.get("file_name", "file")).replace(".", "_")

    if input_type == "code":
        label = str(metadata.get("function_name", "fn"))
    elif input_type == "requirements":
        label = str(metadata.get("chunk_index", "0"))
    elif input_type == "api":
        endpoint = str(metadata.get("endpoint", "ep")).replace("/", "_").strip("_")
        method = str(metadata.get("method", "GET"))
        label = f"{method}_{endpoint}"
    else:
        label = "chunk"

    short_uuid = str(uuid.uuid4())[:8]
    return f"{input_type}_{file_name}_{label}_{short_uuid}"


def _flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize metadata into JSON-serializable scalar values.
    Lists/dicts are stored as JSON strings.
    """
    flat: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif value is None:
            flat[key] = ""
        else:
            flat[key] = json.dumps(value)
    return flat


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m ingestion.embedder ingest <file_path>")
        print("  python -m ingestion.embedder stats")
        print("  python -m ingestion.embedder clear")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Please provide a file path.")
            sys.exit(1)
        stored_ids = ingest_file(sys.argv[2])
        print(f"Ingested {len(stored_ids)} chunk(s).")
    elif command == "stats":
        stats = get_store_stats()
        print("FAISS Store Stats:")
        print(f"  Total chunks : {stats['total_chunks']}")
        print(f"  By type      : {stats['by_type']}")
        print(f"  By file      : {stats['by_file']}")
        print(f"  Collection   : {stats['collection_name']}")
        print(f"  DB path      : {stats['db_path']}")
    elif command == "clear":
        clear_store()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
