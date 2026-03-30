"""
metrics.py
----------
Aggregates evaluation results from evaluator.py into a summary table,
runs RAG vs No-RAG comparison, and exports results to CSV.

Usage:
    from evaluation.metrics import compute_metrics, compare_rag_vs_baseline
    df = compute_metrics(results)          # results from evaluator.evaluate_all()
    compare_rag_vs_baseline("data/sample_code.py")
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_metrics(
    results: List[Dict[str, Any]],
    output_csv: str = "outputs/metrics_report.csv",
    label: str = "RAG",
) -> Dict[str, Any]:
    """
    Aggregate a list of evaluate_file() results into summary metrics.

    Args:
        results    : List of dicts from evaluator.evaluate_all().
        output_csv : Where to save the CSV report.
        label      : Label for this run (e.g. "RAG" or "No-RAG").

    Returns:
        {
            "label"         : str,
            "n_files"       : int,
            "parse_rate"    : float,   # 0–100
            "exec_rate"     : float,   # 0–100
            "coverage"      : float,   # 0–100 or None
            "ok_count"      : int,
            "parse_errors"  : int,
            "exec_errors"   : int,
            "csv_path"      : str,
        }
    """
    if not results:
        print("[Metrics] No results to aggregate.")
        return {}

    n          = len(results)
    ok_count   = sum(1 for r in results if r["status"] == "ok")
    pr_errors  = sum(1 for r in results if r["parse_error"])
    ex_errors  = sum(1 for r in results if r["exec_error"])

    avg_pr     = sum(r["parse_rate"] for r in results) / n * 100
    avg_ex     = sum(r["exec_rate"]  for r in results) / n * 100

    cvs        = [r["coverage"] for r in results if r.get("coverage") is not None]
    avg_cv     = round(sum(cvs) / len(cvs), 2) if cvs else None

    summary = {
        "label"        : label,
        "n_files"      : n,
        "parse_rate"   : round(avg_pr, 2),
        "exec_rate"    : round(avg_ex, 2),
        "coverage"     : avg_cv,
        "ok_count"     : ok_count,
        "parse_errors" : pr_errors,
        "exec_errors"  : ex_errors,
        "csv_path"     : output_csv,
    }

    _print_summary_table(summary)
    _save_csv(results, output_csv, label)

    return summary


def compare_rag_vs_baseline(
    input_file: str,
    output_dir: str = "outputs",
    comparison_csv: str = "outputs/rag_vs_baseline.csv",
) -> None:
    """
    Run the pipeline twice — once with RAG retrieval (normal) and once
    without (zero-shot, skip_embed=True with empty store simulation) —
    then print a side-by-side comparison table.

    Args:
        input_file     : The source file to generate tests for.
        output_dir     : Where generated tests are saved.
        comparison_csv : Output path for the comparison CSV.
    """
    from evaluation.evaluator import evaluate_all
    from pipeline import run_pipeline

    print("\n" + "=" * 60)
    print("  RAG vs No-RAG Comparison")
    print("=" * 60)

    # ── Run 1: With RAG ────────────────────────────────────────────────────────
    print("\n[Metrics] Run 1: WITH RAG retrieval...")
    rag_out = os.path.join(output_dir, "rag_run")
    run_pipeline(input_file=input_file, output_dir=rag_out, top_k=5)
    rag_results = evaluate_all(output_dir=rag_out, source_file=input_file)
    rag_summary = compute_metrics(rag_results, label="RAG",
                                  output_csv=os.path.join(output_dir, "metrics_rag.csv"))

    # ── Run 2: Without RAG (top_k=0 → retriever returns empty context) ────────
    print("\n[Metrics] Run 2: WITHOUT RAG (zero-shot baseline)...")
    norag_out = os.path.join(output_dir, "norag_run")
    run_pipeline(input_file=input_file, output_dir=norag_out, top_k=0)
    norag_results = evaluate_all(output_dir=norag_out, source_file=input_file)
    norag_summary = compute_metrics(norag_results, label="No-RAG",
                                    output_csv=os.path.join(output_dir, "metrics_norag.csv"))

    # ── Print comparison table ─────────────────────────────────────────────────
    _print_comparison(rag_summary, norag_summary)
    _save_comparison_csv(rag_summary, norag_summary, comparison_csv)


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_summary_table(summary: Dict[str, Any]) -> None:
    """Print a formatted summary table to console."""
    cv = f"{summary['coverage']:.1f}%" if summary["coverage"] is not None else "N/A"

    print("\n" + "=" * 50)
    print(f"  METRICS SUMMARY — {summary['label']}")
    print("=" * 50)
    print(f"  Files evaluated : {summary['n_files']}")
    print(f"  Parse Rate (PR) : {summary['parse_rate']:.1f}%")
    print(f"  Exec Rate  (EX) : {summary['exec_rate']:.1f}%")
    print(f"  Coverage   (CV) : {cv}")
    print(f"  ✅ Fully OK     : {summary['ok_count']}")
    print(f"  ❌ Parse errors : {summary['parse_errors']}")
    print(f"  ❌ Exec errors  : {summary['exec_errors']}")
    print("=" * 50)


def _print_comparison(rag: Dict[str, Any], norag: Dict[str, Any]) -> None:
    """Print a side-by-side comparison table."""
    rag_cv   = f"{rag['coverage']:.1f}%"   if rag.get("coverage")   is not None else "N/A"
    norag_cv = f"{norag['coverage']:.1f}%" if norag.get("coverage") is not None else "N/A"

    print("\n" + "=" * 60)
    print("  RAG vs No-RAG Comparison Table")
    print("=" * 60)
    print(f"  {'Metric':<22} {'With RAG':>12} {'No RAG':>12}")
    print("-" * 50)
    print(f"  {'Parse Rate (PR%)' :<22} {rag['parse_rate']:>11.1f}% {norag['parse_rate']:>11.1f}%")
    print(f"  {'Exec Rate (EX%)' :<22} {rag['exec_rate']:>11.1f}% {norag['exec_rate']:>11.1f}%")
    print(f"  {'Coverage (CV%)' :<22} {rag_cv:>12} {norag_cv:>12}")
    print(f"  {'Files OK' :<22} {rag['ok_count']:>12} {norag['ok_count']:>12}")
    print("=" * 60)


# ── CSV export ─────────────────────────────────────────────────────────────────

def _save_csv(
    results: List[Dict[str, Any]],
    output_csv: str,
    label: str,
) -> None:
    """Save per-file results to a CSV file."""
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    fieldnames = [
        "label", "test_file", "status",
        "parse_rate", "exec_rate", "coverage",
        "parse_error", "exec_error",
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "label"      : label,
                    "test_file"  : os.path.basename(r.get("test_file", "")),
                    "status"     : r.get("status", ""),
                    "parse_rate" : r.get("parse_rate", 0.0),
                    "exec_rate"  : r.get("exec_rate",  0.0),
                    "coverage"   : r.get("coverage",   ""),
                    "parse_error": r.get("parse_error","") or "",
                    "exec_error" : r.get("exec_error", "") or "",
                })
        print(f"[Metrics] 💾 CSV saved → {os.path.abspath(output_csv)}")

    except Exception as e:
        print(f"[Metrics] ⚠️  Could not save CSV: {e}")


def _save_comparison_csv(
    rag: Dict[str, Any],
    norag: Dict[str, Any],
    output_csv: str,
) -> None:
    """Save the RAG vs No-RAG summary comparison to a CSV file."""
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    rows = [
        {"metric": "Parse Rate (%)",   "RAG": rag["parse_rate"],  "No-RAG": norag["parse_rate"]},
        {"metric": "Exec Rate (%)",    "RAG": rag["exec_rate"],   "No-RAG": norag["exec_rate"]},
        {"metric": "Coverage (%)",     "RAG": rag.get("coverage") or "N/A", "No-RAG": norag.get("coverage") or "N/A"},
        {"metric": "Files OK",         "RAG": rag["ok_count"],    "No-RAG": norag["ok_count"]},
        {"metric": "Parse Errors",     "RAG": rag["parse_errors"],"No-RAG": norag["parse_errors"]},
        {"metric": "Exec Errors",      "RAG": rag["exec_errors"], "No-RAG": norag["exec_errors"]},
    ]

    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "RAG", "No-RAG"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Metrics] 💾 Comparison CSV saved → {os.path.abspath(output_csv)}")

    except Exception as e:
        print(f"[Metrics] ⚠️  Could not save comparison CSV: {e}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from evaluation.evaluator import evaluate_all

    output_dir  = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    source_file = sys.argv[2] if len(sys.argv) > 2 else None
    label       = sys.argv[3] if len(sys.argv) > 3 else "RAG"

    print(f"[Metrics] Computing metrics for '{output_dir}' (label={label})")
    results = evaluate_all(output_dir=output_dir, source_file=source_file)
    compute_metrics(results, label=label,
                    output_csv=os.path.join(output_dir, "metrics_report.csv"))