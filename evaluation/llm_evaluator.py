"""
llm_evaluator.py
----------------
LLM-based evaluation using DeepEval-style metrics.

Syllabus coverage:
  - LLM Evaluation Techniques using DeepEval and TruLens (Module 4)

Metrics implemented:
  1. Faithfulness         : Are the generated tests grounded in the source code/spec?
  2. Answer Relevance     : Do the tests actually test what was asked?
  3. Contextual Precision : Are the retrieved chunks relevant to the generation?
  4. Contextual Recall    : Did retrieval find all necessary information?
  5. Hallucination Score  : Did the LLM invent functions/imports that don't exist?

Design:
  We implement these metrics both:
    a) Using DeepEval library (if installed) — full LLM-judge evaluation
    b) Using heuristic fallback (no extra dependencies) — fast approximation

Usage:
    from evaluation.llm_evaluator import LLMEvaluator
    evaluator = LLMEvaluator()
    result = evaluator.evaluate_single(
        source_chunk=chunk,
        retrieved_chunks=retrieved,
        generated_test=test_code,
    )
    print(result.summary())
"""

from __future__ import annotations

import re
import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class LLMEvalResult:
    """Holds all evaluation scores for one generated test file."""
    test_file            : str  = ""
    faithfulness         : float = 0.0  # 0-1: tests reference real code entities
    answer_relevance     : float = 0.0  # 0-1: tests cover the right scenarios
    contextual_precision : float = 0.0  # 0-1: retrieved chunks were on-topic
    contextual_recall    : float = 0.0  # 0-1: retrieval found needed context
    hallucination_score  : float = 0.0  # 0-1: LOW is GOOD (1 = lots of hallucination)
    overall_score        : float = 0.0  # weighted average
    details              : Dict[str, Any] = field(default_factory=dict)
    method               : str   = "heuristic"  # "deepeval" or "heuristic"

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  LLM EVALUATION REPORT — {self.method.upper()}",
            f"{'='*55}",
            f"  File               : {self.test_file or 'N/A'}",
            f"  Faithfulness       : {self.faithfulness:.3f}  {'✅' if self.faithfulness >= 0.7 else '⚠️'}",
            f"  Answer Relevance   : {self.answer_relevance:.3f}  {'✅' if self.answer_relevance >= 0.7 else '⚠️'}",
            f"  Contextual Prec.   : {self.contextual_precision:.3f}  {'✅' if self.contextual_precision >= 0.7 else '⚠️'}",
            f"  Contextual Recall  : {self.contextual_recall:.3f}  {'✅' if self.contextual_recall >= 0.7 else '⚠️'}",
            f"  Hallucination      : {self.hallucination_score:.3f}  {'✅' if self.hallucination_score <= 0.3 else '⚠️'} (lower is better)",
            f"  {'─'*45}",
            f"  Overall Score      : {self.overall_score:.3f}",
            f"{'='*55}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_file"           : self.test_file,
            "faithfulness"        : round(self.faithfulness, 4),
            "answer_relevance"    : round(self.answer_relevance, 4),
            "contextual_precision": round(self.contextual_precision, 4),
            "contextual_recall"   : round(self.contextual_recall, 4),
            "hallucination_score" : round(self.hallucination_score, 4),
            "overall_score"       : round(self.overall_score, 4),
            "method"              : self.method,
        }


# ── Main evaluator ─────────────────────────────────────────────────────────────

class LLMEvaluator:
    """
    Evaluates generated test cases using DeepEval-style metrics.

    Automatically uses DeepEval if installed, otherwise falls back to
    fast heuristic evaluation (no extra dependencies needed).

    Syllabus: LLM Evaluation Techniques using DeepEval (Module 4)
    """

    METRIC_WEIGHTS = {
        "faithfulness"         : 0.30,
        "answer_relevance"     : 0.25,
        "contextual_precision" : 0.20,
        "contextual_recall"    : 0.15,
        "hallucination"        : 0.10,   # inverted: lower hallucination = better
    }

    def __init__(self, use_deepeval: bool = True, llm_judge: str = "groq"):
        """
        Args:
            use_deepeval : Try to use DeepEval library first (default True).
            llm_judge    : LLM provider for DeepEval judge ('groq' or 'gemini').
        """
        self._deepeval_available = False
        self._llm_judge = llm_judge

        if use_deepeval:
            self._deepeval_available = self._check_deepeval()

        method = "DeepEval" if self._deepeval_available else "heuristic"
        print(f"[LLMEvaluator] Initialized — using {method} evaluation")

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate_single(
        self,
        source_chunk     : Dict[str, Any],
        retrieved_chunks : List[Dict[str, Any]],
        generated_test   : str,
        test_file        : str = "",
    ) -> LLMEvalResult:
        """
        Evaluate a single generated test against source and context.

        Args:
            source_chunk     : The primary chunk that was being tested.
            retrieved_chunks : Chunks retrieved from the vector store.
            generated_test   : The LLM-generated test code string.
            test_file        : Optional path label for the result.

        Returns:
            LLMEvalResult with all metric scores.
        """
        if self._deepeval_available:
            try:
                return self._evaluate_deepeval(
                    source_chunk, retrieved_chunks, generated_test, test_file
                )
            except Exception as e:
                print(f"[LLMEvaluator] DeepEval failed ({e}), falling back to heuristic")

        return self._evaluate_heuristic(
            source_chunk, retrieved_chunks, generated_test, test_file
        )

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> List[LLMEvalResult]:
        """
        Evaluate a batch of test cases.

        Args:
            test_cases: List of dicts, each with keys:
                'source_chunk', 'retrieved_chunks', 'generated_test', 'test_file'

        Returns:
            List of LLMEvalResult.
        """
        results = []
        for i, tc in enumerate(test_cases, 1):
            print(f"[LLMEvaluator] Evaluating {i}/{len(test_cases)}...")
            result = self.evaluate_single(
                source_chunk     = tc["source_chunk"],
                retrieved_chunks = tc.get("retrieved_chunks", []),
                generated_test   = tc["generated_test"],
                test_file        = tc.get("test_file", f"test_{i}"),
            )
            results.append(result)
            print(result.summary())
        return results

    def aggregate(self, results: List[LLMEvalResult]) -> Dict[str, Any]:
        """Compute aggregate statistics across all evaluated files."""
        if not results:
            return {}
        n = len(results)
        return {
            "n_files"              : n,
            "avg_faithfulness"     : round(sum(r.faithfulness for r in results) / n, 4),
            "avg_answer_relevance" : round(sum(r.answer_relevance for r in results) / n, 4),
            "avg_ctx_precision"    : round(sum(r.contextual_precision for r in results) / n, 4),
            "avg_ctx_recall"       : round(sum(r.contextual_recall for r in results) / n, 4),
            "avg_hallucination"    : round(sum(r.hallucination_score for r in results) / n, 4),
            "avg_overall"          : round(sum(r.overall_score for r in results) / n, 4),
            "method"               : results[0].method,
        }

    # ── DeepEval evaluation ────────────────────────────────────────────────────

    def _check_deepeval(self) -> bool:
        try:
            import deepeval  # noqa: F401
            return True
        except ImportError:
            print("[LLMEvaluator] deepeval not installed. Run: pip install deepeval")
            return False

    def _evaluate_deepeval(
        self,
        source_chunk     : Dict[str, Any],
        retrieved_chunks : List[Dict[str, Any]],
        generated_test   : str,
        test_file        : str,
    ) -> LLMEvalResult:
        """
        Full DeepEval evaluation using LLM-as-judge.
        Uses DeepEval's built-in metrics with an LLM judge.
        """
        from deepeval import evaluate
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            HallucinationMetric,
        )
        from deepeval.test_case import LLMTestCase

        source_content = source_chunk.get("content", "")
        context_list   = [c.get("content", "") for c in retrieved_chunks]
        input_query    = f"Generate tests for: {source_content[:500]}"

        test_case = LLMTestCase(
            input            = input_query,
            actual_output    = generated_test,
            expected_output  = f"Comprehensive tests for {source_chunk.get('metadata', {}).get('function_name', 'the function')}",
            context          = context_list,
            retrieval_context= context_list,
        )

        # Configure LLM judge
        llm = self._get_deepeval_llm()

        metrics = [
            FaithfulnessMetric(threshold=0.7, model=llm, include_reason=True),
            AnswerRelevancyMetric(threshold=0.7, model=llm, include_reason=True),
            ContextualPrecisionMetric(threshold=0.7, model=llm, include_reason=True),
            ContextualRecallMetric(threshold=0.7, model=llm, include_reason=True),
            HallucinationMetric(threshold=0.3, model=llm, include_reason=True),
        ]

        for metric in metrics:
            metric.measure(test_case)

        result = LLMEvalResult(
            test_file            = test_file,
            faithfulness         = metrics[0].score or 0.0,
            answer_relevance     = metrics[1].score or 0.0,
            contextual_precision = metrics[2].score or 0.0,
            contextual_recall    = metrics[3].score or 0.0,
            hallucination_score  = metrics[4].score or 0.0,
            method               = "deepeval",
        )
        result.overall_score = self._compute_overall(result)
        return result

    def _get_deepeval_llm(self):
        """Configure DeepEval LLM judge from environment variables."""
        import os
        try:
            from deepeval.models import DeepEvalBaseLLM
            # Use Groq as the judge model
            groq_key = os.getenv("GROQ_API_KEY", "")
            if groq_key:
                # DeepEval supports custom LLMs — use GPT-compatible Groq endpoint
                from deepeval.models.gpt_model import GPTModel
                return GPTModel(model="llama-3.3-70b-versatile", api_key=groq_key)
        except Exception:
            pass
        return None  # DeepEval will use its default

    # ── Heuristic evaluation ───────────────────────────────────────────────────

    def _evaluate_heuristic(
        self,
        source_chunk     : Dict[str, Any],
        retrieved_chunks : List[Dict[str, Any]],
        generated_test   : str,
        test_file        : str,
    ) -> LLMEvalResult:
        """
        Fast heuristic evaluation — no LLM judge required.
        Approximates DeepEval metrics using structural analysis.

        These are well-defined proxy metrics:
        - Faithfulness      ≈ overlap of identifiers between source and tests
        - Answer Relevance  ≈ proportion of test_* functions with assert statements
        - Ctx Precision     ≈ retrieved chunks whose type matches source type
        - Ctx Recall        ≈ whether any retrieved chunk is referenced in the test
        - Hallucination     ≈ imports that don't appear in source or context
        """
        source_content = source_chunk.get("content", "")
        source_meta    = source_chunk.get("metadata", {})
        input_type     = source_meta.get("type", "unknown")

        # ── Metric 1: Faithfulness ─────────────────────────────────────────────
        faithfulness = self._heuristic_faithfulness(source_content, generated_test, source_meta)

        # ── Metric 2: Answer Relevance ────────────────────────────────────────
        answer_relevance = self._heuristic_answer_relevance(generated_test, input_type)

        # ── Metric 3: Contextual Precision ───────────────────────────────────
        contextual_precision = self._heuristic_ctx_precision(retrieved_chunks, input_type)

        # ── Metric 4: Contextual Recall ───────────────────────────────────────
        contextual_recall = self._heuristic_ctx_recall(retrieved_chunks, generated_test)

        # ── Metric 5: Hallucination ───────────────────────────────────────────
        hallucination = self._heuristic_hallucination(
            source_content,
            [c.get("content", "") for c in retrieved_chunks],
            generated_test,
        )

        result = LLMEvalResult(
            test_file            = test_file,
            faithfulness         = faithfulness,
            answer_relevance     = answer_relevance,
            contextual_precision = contextual_precision,
            contextual_recall    = contextual_recall,
            hallucination_score  = hallucination,
            method               = "heuristic",
            details              = {
                "input_type"         : input_type,
                "source_length"      : len(source_content),
                "generated_length"   : len(generated_test),
                "n_retrieved"        : len(retrieved_chunks),
            },
        )
        result.overall_score = self._compute_overall(result)
        return result

    def _heuristic_faithfulness(
        self,
        source: str,
        generated: str,
        meta: Dict[str, Any],
    ) -> float:
        """
        Faithfulness: the generated test references real identifiers from the source.

        Approach:
        - Extract all identifiers from the source code
        - Check what fraction appear in the generated test
        """
        # Get key identifiers from source
        source_identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]+\b", source))
        # Add function/class names from metadata
        fn_name = meta.get("function_name", "")
        if fn_name:
            source_identifiers.add(fn_name)

        # Filter out very short/common Python keywords
        stop_words = {
            "def", "return", "if", "else", "for", "in", "not", "and", "or",
            "True", "False", "None", "import", "from", "class", "self",
            "int", "str", "float", "list", "dict", "bool", "any", "len",
        }
        source_identifiers -= stop_words
        source_identifiers = {w for w in source_identifiers if len(w) > 2}

        if not source_identifiers:
            return 0.5  # Can't measure

        # How many source identifiers appear in the generated test?
        matched = sum(1 for ident in source_identifiers if ident in generated)
        score   = matched / len(source_identifiers)

        # Bonus: function name appears in import or test name
        if fn_name and (f"import {fn_name}" in generated or f"_{fn_name}_" in generated or f"_{fn_name}" in generated):
            score = min(1.0, score + 0.1)

        return round(min(1.0, score), 4)

    def _heuristic_answer_relevance(self, generated: str, input_type: str) -> float:
        """
        Answer Relevance: the output actually contains tests (not just comments/empty).

        For code: count test functions that have at least one assert
        For requirements: count Scenario blocks with Given/When/Then
        For api: count test functions that use requests and assert status
        """
        if input_type == "code":
            # Count test functions
            test_fns = re.findall(r"def test_\w+\(", generated)
            if not test_fns:
                return 0.0
            # Count test functions that have asserts
            has_assert   = len(re.findall(r"\bassert\b", generated))
            has_raises   = len(re.findall(r"pytest\.raises", generated))
            total_checks = has_assert + has_raises
            # Score: ratio of checks to test functions (capped)
            ratio = min(1.0, total_checks / max(len(test_fns), 1))
            return round(ratio, 4)

        elif input_type == "requirements":
            has_feature  = "Feature:" in generated
            has_scenario = "Scenario" in generated
            has_given    = "Given" in generated
            has_when     = "When" in generated
            has_then     = "Then" in generated
            gherkin_components = [has_feature, has_scenario, has_given, has_when, has_then]
            return round(sum(gherkin_components) / len(gherkin_components), 4)

        elif input_type == "api":
            test_fns     = re.findall(r"def test_\w+\(", generated)
            has_requests = "requests." in generated or "import requests" in generated
            has_assert   = "assert" in generated
            if not test_fns:
                return 0.0
            base  = 0.5 if has_requests else 0.2
            bonus = 0.3 if has_assert else 0.0
            return round(min(1.0, base + bonus + (0.2 if len(test_fns) >= 3 else 0.0)), 4)

        return 0.5

    def _heuristic_ctx_precision(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        expected_type: str,
    ) -> float:
        """
        Contextual Precision: what fraction of retrieved chunks are the correct type?
        A chunk is "relevant" if its type matches the query type.
        """
        if not retrieved_chunks:
            return 0.0
        relevant = sum(
            1 for c in retrieved_chunks
            if c.get("metadata", {}).get("type") == expected_type
        )
        return round(relevant / len(retrieved_chunks), 4)

    def _heuristic_ctx_recall(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        generated_test: str,
    ) -> float:
        """
        Contextual Recall: how many retrieved chunks contributed something
        to the generated test (measured by identifier overlap).
        """
        if not retrieved_chunks:
            return 0.0

        used_chunks = 0
        for chunk in retrieved_chunks:
            chunk_text   = chunk.get("content", "")
            chunk_idents = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", chunk_text))
            chunk_idents -= {
                "def", "return", "self", "import", "class", "True", "False"
            }
            # Check if any identifier from this chunk appears in the test
            if any(ident in generated_test for ident in chunk_idents):
                used_chunks += 1

        return round(used_chunks / len(retrieved_chunks), 4)

    def _heuristic_hallucination(
        self,
        source: str,
        context_list: List[str],
        generated: str,
    ) -> float:
        """
        Hallucination: imports or function calls in the test that don't
        appear anywhere in source or context. Lower is better.
        """
        # Extract imports from generated test
        import_lines = re.findall(
            r"^(?:from|import)\s+([\w.]+)",
            generated,
            re.MULTILINE,
        )

        if not import_lines:
            return 0.0  # No imports = can't measure hallucination

        all_context = source + " " + " ".join(context_list)
        hallucinated = 0

        # Standard test libraries — always allowed
        allowed = {
            "pytest", "unittest", "requests", "json", "os", "sys",
            "re", "math", "typing", "datetime", "collections", "itertools",
            "functools", "pathlib", "io", "copy", "time", "random",
        }

        for imp in import_lines:
            root_module = imp.split(".")[0]
            if root_module in allowed:
                continue
            # Check if this module name appears anywhere in source or context
            if root_module not in all_context:
                hallucinated += 1

        score = hallucinated / max(len(import_lines), 1)
        return round(min(1.0, score), 4)

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def _compute_overall(self, result: LLMEvalResult) -> float:
        """Compute weighted overall score. Hallucination is inverted (lower=better)."""
        w = self.METRIC_WEIGHTS
        score = (
            w["faithfulness"]          * result.faithfulness
            + w["answer_relevance"]    * result.answer_relevance
            + w["contextual_precision"]* result.contextual_precision
            + w["contextual_recall"]   * result.contextual_recall
            + w["hallucination"]       * (1.0 - result.hallucination_score)
        )
        return round(score, 4)


# ── RAG vs No-RAG comparison ───────────────────────────────────────────────────

def compare_rag_vs_norag(
    rag_cases   : List[Dict[str, Any]],
    norag_cases : List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare LLM evaluation scores for RAG vs No-RAG runs.

    Args:
        rag_cases   : List of test cases from the RAG pipeline run.
        norag_cases : List of test cases from the no-RAG (zero-shot) run.

    Returns:
        Dict with side-by-side comparison of all metrics.
    """
    evaluator = LLMEvaluator()

    print("\n[LLMEval] Evaluating RAG run...")
    rag_results   = evaluator.evaluate_batch(rag_cases)
    rag_agg       = evaluator.aggregate(rag_results)

    print("\n[LLMEval] Evaluating No-RAG run...")
    norag_results = evaluator.evaluate_batch(norag_cases)
    norag_agg     = evaluator.aggregate(norag_results)

    print("\n" + "=" * 60)
    print("  LLM EVALUATION COMPARISON — RAG vs No-RAG")
    print("=" * 60)
    metrics = [
        ("Faithfulness",        "avg_faithfulness"),
        ("Answer Relevance",    "avg_answer_relevance"),
        ("Ctx Precision",       "avg_ctx_precision"),
        ("Ctx Recall",          "avg_ctx_recall"),
        ("Hallucination (↓)",   "avg_hallucination"),
        ("OVERALL SCORE",       "avg_overall"),
    ]
    print(f"  {'Metric':<25} {'RAG':>10} {'No-RAG':>10}  {'Delta':>8}")
    print("  " + "-" * 53)
    for label, key in metrics:
        r = rag_agg.get(key, 0.0)
        n = norag_agg.get(key, 0.0)
        d = r - n
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
        print(f"  {label:<25} {r:>10.3f} {n:>10.3f}  {arrow}{abs(d):>6.3f}")
    print("=" * 60)

    return {
        "rag"   : rag_agg,
        "norag" : norag_agg,
        "delta" : {k: round(rag_agg.get(k, 0) - norag_agg.get(k, 0), 4) for k in rag_agg},
    }


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke test with a synthetic example
    evaluator = LLMEvaluator(use_deepeval=False)

    fake_source = {
        "content": "def add(a: int, b: int) -> int:\n    return a + b",
        "metadata": {"type": "code", "function_name": "add", "file_name": "math_utils.py"},
    }
    fake_retrieved = [
        {
            "content" : "def multiply(a, b): return a * b",
            "metadata": {"type": "code", "function_name": "multiply"},
        }
    ]
    fake_generated = """
import pytest
from math_utils import add

# Add two positive integers
def test_add_positive_numbers():
    assert add(2, 3) == 5

# Add negative numbers
def test_add_negative():
    assert add(-1, -2) == -3

# Add zero
def test_add_with_zero():
    assert add(0, 5) == 5

# Type check
def test_add_wrong_type():
    with pytest.raises(TypeError):
        add("a", 1)
"""

    result = evaluator.evaluate_single(fake_source, fake_retrieved, fake_generated, "test_add.py")
    print(result.summary())
