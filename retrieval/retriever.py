"""
retriever.py
------------
Detects the input type from a chunk's metadata and retrieves the top-k
most relevant chunks from the FAISS store for use in prompt construction.

Usage:
    from retrieval.retriever import retrieve
    results = retrieve(query_text="def validate_email(email):", input_type="code")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from retrieval.vector_store import get_store

# Default number of similar chunks to retrieve
DEFAULT_TOP_K = 5


def detect_input_type(metadata: Dict[str, Any]) -> str:
    """
    Detect the input mode from a chunk's metadata 'type' field.

    Args:
        metadata: A chunk metadata dict as produced by Person 1's parsers.

    Returns:
        One of: "code", "requirements", "api", or "unknown".
    """
    return str(metadata.get("type", "unknown")).lower()


def retrieve(
    query_text: str,
    input_type: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks from the FAISS store,
    filtered to the same input type as the query.

    Args:
        query_text : The text to search with (e.g. a function body,
                     a requirement paragraph, or an endpoint description).
        input_type : One of "code", "requirements", "api".
        top_k      : How many chunks to return (default 5).

    Returns:
        List of result dicts, each containing:
            {
                "id"       : str,
                "content"  : str,
                "metadata" : dict,
                "score"    : float,
                "rank"     : int,    # 1-indexed rank (1 = most relevant)
                "source"   : str,    # human-readable source label for traceability
            }
        Sorted by relevance score descending.
    """
    if not query_text or not query_text.strip():
        return []

    store = get_store()
    raw_results = store.query(
        text=query_text,
        n_results=top_k,
        filter_type=input_type,
    )

    # Annotate with rank + source label for traceability (used by test_formatter)
    enriched = []
    for rank, result in enumerate(raw_results, start=1):
        result["rank"]   = rank
        result["source"] = _build_source_label(result["metadata"])
        enriched.append(result)

    return enriched


def retrieve_for_chunk(
    chunk: Dict[str, Any],
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: takes a parsed chunk dict (as produced by Person 1's
    parsers) and retrieves similar chunks of the same type.

    Args:
        chunk : A chunk dict with 'content' and 'metadata' keys.
        top_k : How many results to return.

    Returns:
        Same format as retrieve().
    """
    content    = chunk.get("content", "")
    metadata   = chunk.get("metadata", {})
    input_type = detect_input_type(metadata)

    return retrieve(query_text=content, input_type=input_type, top_k=top_k)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_source_label(metadata: Dict[str, Any]) -> str:
    """
    Build a human-readable source label for traceability.
    Format varies by type so Person 3's UI can display it clearly.
    """
    input_type = metadata.get("type", "unknown")
    file_name  = metadata.get("file_name", "unknown")

    if input_type == "code":
        fn = metadata.get("function_name", "unknown_function")
        return f"{file_name} :: {fn}()"

    elif input_type == "requirements":
        section = metadata.get("section", "Unknown Section")
        page    = metadata.get("page", "?")
        idx     = metadata.get("chunk_index", "?")
        return f"{file_name} :: {section} (page {page}, chunk {idx})"

    elif input_type == "api":
        method   = metadata.get("method", "?")
        endpoint = metadata.get("endpoint", "?")
        return f"{file_name} :: {method} {endpoint}"

    return f"{file_name} :: chunk"


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m retrieval.retriever <input_type> <query>")
        print("  input_type: code | requirements | api")
        print("  Example: python -m retrieval.retriever code 'def add(a, b)'")
        sys.exit(1)

    itype = sys.argv[1]
    query = " ".join(sys.argv[2:])

    print(f"\nRetrieving top-{DEFAULT_TOP_K} '{itype}' chunks for: '{query}'\n")
    results = retrieve(query_text=query, input_type=itype)

    if not results:
        print("No results found. Make sure the FAISS store has been populated.")
    else:
        for r in results:
            print(f"[Rank {r['rank']}] Score: {r['score']:.4f}")
            print(f"  Source : {r['source']}")
            print(f"  Preview: {r['content'][:150]}...")
            print()
