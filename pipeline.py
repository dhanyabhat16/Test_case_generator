"""
pipeline.py  (UPGRADED)
-----------------------
End-to-end orchestrator with:
  - Hybrid BM25+SBERT retrieval (Module 3: Advanced RAG)
  - Named prompting techniques: few-shot, CoT, zero-shot (Module 2)
  - LLM evaluation via DeepEval-style metrics (Module 4)
  - RAG vs No-RAG comparison mode

Usage (CLI):
    python pipeline.py path/to/file.py
    python pipeline.py path/to/file.py --technique cot
    python pipeline.py path/to/file.py --retrieval hybrid
    python pipeline.py path/to/file.py --compare   # RAG vs No-RAG comparison

Usage (Python):
    from pipeline import run_pipeline
    results = run_pipeline("path/to/file.py", technique="few_shot", retrieval="hybrid")
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def run_pipeline(
    input_file   : str,
    output_dir   : str = "outputs",
    top_k        : int = 5,
    skip_embed   : bool = False,
    technique    : str = "few_shot",     # zero_shot | one_shot | few_shot | chain_of_thought
    retrieval    : str = "hybrid",       # hybrid | dense | bm25
    alpha        : float = 0.5,          # hybrid fusion weight (0=BM25 only, 1=dense only)
    run_llm_eval : bool = True,          # run DeepEval-style evaluation
) -> List[Dict[str, Any]]:
    """
    Run the full upgraded test generation pipeline.

    Args:
        input_file   : Path to .py / .java / .pdf / .docx / .yaml / .json
        output_dir   : Where to save generated test files
        top_k        : Number of chunks to retrieve per query
        skip_embed   : Skip re-embedding (use existing store)
        technique    : Prompting technique (zero_shot/one_shot/few_shot/chain_of_thought)
        retrieval    : Retrieval method (hybrid/dense/bm25)
        alpha        : Hybrid fusion weight for BM25 vs SBERT
        run_llm_eval : Whether to run LLM-based evaluation after generation

    Returns:
        List of result dicts with generation + evaluation metrics.
    """
    _print_banner(input_file, technique, retrieval)

    # ── Step 1: Parse ──────────────────────────────────────────────────────────
    print("\n[Pipeline] Step 1/6 — Parsing input file...")
    chunks = _parse_file(input_file)
    if not chunks:
        print("[Pipeline] ❌ No chunks parsed. Aborting.")
        return []
    print(f"[Pipeline] ✅ Parsed {len(chunks)} chunk(s)")

    # ── Step 2: Embed ──────────────────────────────────────────────────────────
    if not skip_embed:
        print("\n[Pipeline] Step 2/6 — Embedding and storing chunks...")
        from ingestion.embedder import embed_and_store
        chunk_ids = embed_and_store(chunks)
        print(f"[Pipeline] ✅ Stored {len(chunk_ids)} chunk(s) in FAISS store")
    else:
        print("\n[Pipeline] Step 2/6 — Skipping embed (skip_embed=True)")

    # ── Step 3: Load retriever ─────────────────────────────────────────────────
    print(f"\n[Pipeline] Step 3/6 — Loading {retrieval} retriever...")
    retriever_fn = _load_retriever(retrieval, alpha)
    print(f"[Pipeline] ✅ {retrieval.upper()} retriever ready")

    # ── Step 4: Generate ───────────────────────────────────────────────────────
    print(f"\n[Pipeline] Steps 4-5/6 — Generating tests ({technique} prompting)...")
    print("-" * 60)

    from retrieval.prompt_engine import build_prompt
    from generation.llm_client    import generate
    from generation.test_formatter import format_and_save

    results     = []
    eval_cases  = []
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks, start=1):
        meta       = chunk.get("metadata", {})
        input_type = meta.get("type", "unknown")
        label      = _get_chunk_label(meta)

        print(f"\n[{i}/{len(chunks)}] {label} ({input_type})")

        try:
            # Retrieve with chosen method
            retrieved = retriever_fn(
                query_text=chunk.get("content", ""),
                input_type=input_type,
                top_k=top_k,
            )
            method_used = retrieved[0].get("retrieval_method", retrieval) if retrieved else retrieval
            print(f"          Retrieved {len(retrieved)} chunks via {method_used}")

            # Build prompt with named technique
            prompt = build_prompt(chunk, retrieved, technique=technique)
            print(f"          Prompt: {len(prompt)} chars | Technique: {technique}")

            # Generate
            t0         = time.time()
            raw_output = generate(prompt)
            elapsed    = time.time() - t0
            print(f"          LLM: {elapsed:.1f}s | {len(raw_output)} chars")

            # Save
            output_path = format_and_save(
                raw_output       = raw_output,
                primary_chunk    = chunk,
                retrieved_chunks = retrieved,
                output_dir       = output_dir,
            )

            results.append({
                "chunk_id"         : meta.get("chunk_index", str(i)),
                "input_type"       : input_type,
                "source"           : label,
                "output_file"      : output_path,
                "status"           : "ok",
                "error"            : None,
                "retrieval_method" : method_used,
                "prompt_technique" : technique,
            })

            # Collect for LLM evaluation
            if run_llm_eval:
                eval_cases.append({
                    "source_chunk"    : chunk,
                    "retrieved_chunks": retrieved,
                    "generated_test"  : raw_output,
                    "test_file"       : output_path,
                })

            print(f"          ✅ Saved → {output_path}")

        except Exception as exc:
            print(f"          ❌ Error: {exc}")
            results.append({
                "chunk_id"         : meta.get("chunk_index", str(i)),
                "input_type"       : input_type,
                "source"           : label,
                "output_file"      : None,
                "status"           : "error",
                "error"            : str(exc),
                "retrieval_method" : retrieval,
                "prompt_technique" : technique,
            })

    # ── Step 5: Structural evaluation ─────────────────────────────────────────
    print("\n[Pipeline] Step 5/6 — Structural evaluation (parse/exec/coverage)...")
    from evaluation.evaluator import evaluate_all
    eval_results = evaluate_all(output_dir=output_dir, source_file=input_file)
    from evaluation.metrics import compute_metrics
    compute_metrics(eval_results, label=f"RAG_{technique}")

    # ── Step 6: LLM evaluation ─────────────────────────────────────────────────
    if run_llm_eval and eval_cases:
        print("\n[Pipeline] Step 6/6 — LLM evaluation (DeepEval-style metrics)...")
        from evaluation.llm_evaluator import LLMEvaluator
        llm_eval = LLMEvaluator()
        llm_results = llm_eval.evaluate_batch(eval_cases)
        agg = llm_eval.aggregate(llm_results)
        print(f"\n[Pipeline] LLM Eval Summary:")
        print(f"  Faithfulness      : {agg.get('avg_faithfulness', 0):.3f}")
        print(f"  Answer Relevance  : {agg.get('avg_answer_relevance', 0):.3f}")
        print(f"  Ctx Precision     : {agg.get('avg_ctx_precision', 0):.3f}")
        print(f"  Overall           : {agg.get('avg_overall', 0):.3f}")

        # Attach to results
        for r, lr in zip(results, llm_results):
            if r["status"] == "ok":
                r["llm_eval"] = lr.to_dict()
    else:
        print("\n[Pipeline] Step 6/6 — Skipping LLM evaluation")

    _print_summary(results, output_dir)
    return results


def run_comparison(
    input_file : str,
    output_dir : str = "outputs",
    technique  : str = "few_shot",
) -> Dict[str, Any]:
    """
    Run RAG vs No-RAG comparison — fills in the results table from README.

    Runs the pipeline twice:
      1. With hybrid retrieval (top_k=5)
      2. Without retrieval (top_k=0 = zero-shot baseline)

    Then compares structural + LLM evaluation metrics side-by-side.
    """
    from evaluation.evaluator import evaluate_all
    from evaluation.metrics   import compute_metrics, compare_rag_vs_baseline

    print("\n" + "=" * 60)
    print("  RAG vs No-RAG Full Comparison")
    print("=" * 60)

    # Run 1: With RAG
    print("\n[Comparison] Run 1: WITH hybrid RAG retrieval...")
    rag_out = os.path.join(output_dir, "rag_run")
    run_pipeline(
        input_file=input_file, output_dir=rag_out,
        top_k=5, technique=technique, retrieval="hybrid",
        run_llm_eval=True,
    )

    # Run 2: Without RAG
    print("\n[Comparison] Run 2: WITHOUT retrieval (zero-shot baseline)...")
    norag_out = os.path.join(output_dir, "norag_run")
    run_pipeline(
        input_file=input_file, output_dir=norag_out,
        top_k=0, technique="zero_shot", retrieval="dense",
        run_llm_eval=True,
    )

    # Structural comparison
    compare_rag_vs_baseline(
        input_file=input_file,
        output_dir=output_dir,
    )

    return {"rag_dir": rag_out, "norag_dir": norag_out}


# ── Retriever loader ──────────────────────────────────────────────────────────

def _load_retriever(method: str, alpha: float = 0.5):
    """Return a callable retriever function based on chosen method."""

    if method == "hybrid":
        try:
            from retrieval.hybrid_retriever import retrieve_hybrid
            def hybrid_fn(query_text, input_type, top_k):
                return retrieve_hybrid(query_text, input_type, top_k=top_k, alpha=alpha)
            return hybrid_fn
        except Exception as e:
            print(f"[Pipeline] Hybrid retriever failed ({e}), falling back to dense")

    # Fallback: original dense retriever
    from retrieval.retriever import retrieve
    def dense_fn(query_text, input_type, top_k):
        return retrieve(query_text, input_type, top_k)
    return dense_fn


# ── Helpers (same as original) ────────────────────────────────────────────────

def _parse_file(file_path: str) -> List[Dict[str, Any]]:
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
        raise ValueError(f"Unsupported file extension: '{ext}'")


def _get_chunk_label(metadata: Dict[str, Any]) -> str:
    input_type = metadata.get("type", "unknown")
    file_name  = metadata.get("file_name", "?")
    if input_type == "code":
        return f"{file_name} :: {metadata.get('function_name', '?')}()"
    elif input_type == "requirements":
        return f"{file_name} :: {metadata.get('section', '?')}"
    elif input_type == "api":
        return f"{file_name} :: {metadata.get('method','?')} {metadata.get('endpoint','?')}"
    return f"{file_name} :: chunk"


def _print_banner(input_file: str, technique: str, retrieval: str) -> None:
    print("=" * 60)
    print("  TestGen RAG — Automated Test Case Generator (UPGRADED)")
    print("=" * 60)
    print(f"  Input        : {input_file}")
    print(f"  Retrieval    : {retrieval.upper()} (BM25+SBERT hybrid)")
    print(f"  Prompting    : {technique}")
    print(f"  File exists  : {os.path.exists(input_file)}")


def _print_summary(results: List[Dict[str, Any]], output_dir: str) -> None:
    ok  = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] == "error")
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Chunks processed : {len(results)}")
    print(f"  ✅ Successful    : {ok}")
    print(f"  ❌ Failed        : {err}")
    print(f"  Output directory : {os.path.abspath(output_dir)}")
    if ok > 0:
        print("\n  Generated files:")
        for r in results:
            if r["status"] == "ok":
                method = r.get("retrieval_method", "?")
                tech   = r.get("prompt_technique", "?")
                print(f"    → {r['output_file']}  [{method} | {tech}]")
    print("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TestGen RAG — Test Case Generator")
    parser.add_argument("input_file",  help="Input file (.py, .java, .pdf, .docx, .yaml, .json)")
    parser.add_argument("output_dir",  nargs="?", default="outputs", help="Output directory")
    parser.add_argument("--technique", default="few_shot",
                        choices=["zero_shot", "one_shot", "few_shot", "chain_of_thought"],
                        help="Prompting technique (default: few_shot)")
    parser.add_argument("--retrieval", default="hybrid",
                        choices=["hybrid", "dense", "bm25"],
                        help="Retrieval method (default: hybrid)")
    parser.add_argument("--alpha",     type=float, default=0.5,
                        help="Hybrid fusion weight 0=BM25 only, 1=dense only (default: 0.5)")
    parser.add_argument("--compare",   action="store_true",
                        help="Run RAG vs No-RAG comparison")
    parser.add_argument("--no-eval",   action="store_true",
                        help="Skip LLM evaluation")

    args = parser.parse_args()

    if args.compare:
        run_comparison(
            input_file=args.input_file,
            output_dir=args.output_dir,
            technique=args.technique,
        )
    else:
        run_pipeline(
            input_file   = args.input_file,
            output_dir   = args.output_dir,
            technique    = args.technique,
            retrieval    = args.retrieval,
            alpha        = args.alpha,
            run_llm_eval = not args.no_eval,
        )
