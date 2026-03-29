"""
pipeline.py
-----------
End-to-end orchestrator for the test case generation pipeline.

Flow:
    input_file → parser → embedder → retriever → prompt_engine → llm_client → formatter → output

Usage (CLI):
    python pipeline.py path/to/file.py
    python pipeline.py path/to/requirements.pdf
    python pipeline.py path/to/api_spec.yaml

Usage (Python API):
    from pipeline import run_pipeline
    results = run_pipeline("path/to/file.py")
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

# Load .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def run_pipeline(
    input_file: str,
    output_dir: str = "outputs",
    top_k: int = 5,
    skip_embed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run the full test generation pipeline on a single input file.

    Args:
        input_file  : Path to a .py / .java / .pdf / .docx / .yaml / .json file.
        output_dir  : Directory to save generated test files (default: 'outputs').
        top_k       : Number of similar chunks to retrieve per query (default 5).
        skip_embed  : If True, skip re-embedding and use whatever is in the store.
                      Useful for re-running generation without re-parsing.

    Returns:
        List of result dicts, one per chunk processed:
        {
            "chunk_id"    : str,   # ID of the primary chunk
            "input_type"  : str,   # "code" | "requirements" | "api"
            "source"      : str,   # human-readable source label
            "output_file" : str,   # path to the saved test file
            "status"      : str,   # "ok" | "error"
            "error"       : str,   # error message if status == "error"
        }
    """
    _print_banner(input_file)

    # ── Step 1: Parse input file ───────────────────────────────────────────────
    print("\n[Pipeline] Step 1/5 — Parsing input file...")
    chunks = _parse_file(input_file)
    if not chunks:
        print("[Pipeline] ❌ No chunks parsed. Aborting.")
        return []
    print(f"[Pipeline] ✅ Parsed {len(chunks)} chunk(s)")

    # ── Step 2: Embed and store chunks ────────────────────────────────────────
    if not skip_embed:
        print("\n[Pipeline] Step 2/5 — Embedding and storing chunks...")
        from ingestion.embedder import embed_and_store
        chunk_ids = embed_and_store(chunks)
        print(f"[Pipeline] ✅ Stored {len(chunk_ids)} chunk(s) in FAISS store")
    else:
        print("\n[Pipeline] Step 2/5 — Skipping embed (skip_embed=True)")

    # ── Step 3: Load vector store ─────────────────────────────────────────────
    print("\n[Pipeline] Step 3/5 — Loading vector store...")
    from retrieval.vector_store import VectorStore
    store = VectorStore()
    stats = store.get_stats()
    print(f"[Pipeline] ✅ Store ready — {stats['total_chunks']} total chunks")

    # ── Step 4 + 5: For each chunk — retrieve, prompt, generate, format ───────
    print(f"\n[Pipeline] Steps 4-5/5 — Generating tests for {len(chunks)} chunk(s)...")
    print("-" * 60)

    from retrieval.retriever   import retrieve_for_chunk
    from retrieval.prompt_engine import build_prompt
    from generation.llm_client  import generate
    from generation.test_formatter import format_and_save

    results = []
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks, start=1):
        meta       = chunk.get("metadata", {})
        input_type = meta.get("type", "unknown")
        label      = _get_chunk_label(meta)

        print(f"\n[{i}/{len(chunks)}] Processing: {label} ({input_type})")

        try:
            # Step 4a: Retrieve similar chunks
            retrieved = retrieve_for_chunk(chunk, top_k=top_k)
            print(f"          Retrieved {len(retrieved)} similar chunk(s)")

            # Step 4b: Build prompt
            prompt = build_prompt(chunk, retrieved)
            print(f"          Prompt built ({len(prompt)} chars)")

            # Step 5a: Generate with LLM
            print("          Calling LLM...")
            t0          = time.time()
            raw_output  = generate(prompt)
            elapsed     = time.time() - t0
            print(f"          LLM responded in {elapsed:.1f}s ({len(raw_output)} chars)")

            # Step 5b: Format and save
            output_path = format_and_save(
                raw_output    = raw_output,
                primary_chunk = chunk,
                retrieved_chunks = retrieved,
                output_dir    = output_dir,
            )

            results.append(
                {
                    "chunk_id"   : meta.get("chunk_index", str(i)),
                    "input_type" : input_type,
                    "source"     : label,
                    "output_file": output_path,
                    "status"     : "ok",
                    "error"      : None,
                }
            )
            print(f"          ✅ Saved → {output_path}")

        except Exception as exc:
            print(f"          ❌ Error: {exc}")
            results.append(
                {
                    "chunk_id"   : meta.get("chunk_index", str(i)),
                    "input_type" : input_type,
                    "source"     : label,
                    "output_file": None,
                    "status"     : "error",
                    "error"      : str(exc),
                }
            )

    # ── Summary ────────────────────────────────────────────────────────────────
    _print_summary(results, output_dir)
    return results


# ── File parser dispatcher ─────────────────────────────────────────────────────

def _parse_file(file_path: str) -> List[Dict[str, Any]]:
    """Dispatch to the correct parser based on file extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: '{file_path}'")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".py", ".java"):
        from ingestion.code_parser import parse_code_file
        return parse_code_file(file_path)

    elif ext in (".pdf", ".docx"):
        from ingestion.doc_parser import parse_doc_file
        return parse_doc_file(file_path)

    elif ext in (".yaml", ".yml", ".json"):
        from ingestion.api_parser import parse_api_file
        return parse_api_file(file_path)

    else:
        raise ValueError(
            f"Unsupported file extension: '{ext}'. "
            "Supported: .py, .java, .pdf, .docx, .yaml, .yml, .json"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_chunk_label(metadata: Dict[str, Any]) -> str:
    """Build a short human-readable label for a chunk."""
    input_type = metadata.get("type", "unknown")
    file_name  = metadata.get("file_name", "?")

    if input_type == "code":
        return f"{file_name} :: {metadata.get('function_name', '?')}()"
    elif input_type == "requirements":
        return f"{file_name} :: {metadata.get('section', '?')}"
    elif input_type == "api":
        return f"{file_name} :: {metadata.get('method', '?')} {metadata.get('endpoint', '?')}"
    return f"{file_name} :: chunk"


def _print_banner(input_file: str) -> None:
    print("=" * 60)
    print("  TestGen RAG — Automated Test Case Generator")
    print("=" * 60)
    print(f"  Input file : {input_file}")
    print(f"  File exists: {os.path.exists(input_file)}")


def _print_summary(results: List[Dict[str, Any]], output_dir: str) -> None:
    ok_count  = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total chunks processed : {len(results)}")
    print(f"  ✅ Successful           : {ok_count}")
    print(f"  ❌ Failed               : {err_count}")
    print(f"  Output directory        : {os.path.abspath(output_dir)}")

    if ok_count > 0:
        print("\n  Generated files:")
        for r in results:
            if r["status"] == "ok":
                print(f"    → {r['output_file']}")

    if err_count > 0:
        print("\n  Errors:")
        for r in results:
            if r["status"] == "error":
                print(f"    ✗ {r['source']}: {r['error']}")

    print("=" * 60)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <input_file> [output_dir]")
        print("")
        print("Examples:")
        print("  python pipeline.py data/sample_code.py")
        print("  python pipeline.py data/sample_requirements.pdf")
        print("  python pipeline.py data/sample_api.yaml outputs/")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "outputs"

    run_pipeline(input_file=input_path, output_dir=output_path)
