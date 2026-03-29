"""
test_formatter.py
-----------------
Parses LLM output into a structured format, adds traceability footers,
and saves generated tests to the correct file type.

Output formats:
  - Code mode       → .py  (annotated pytest file)
  - Requirements    → .feature (Gherkin BDD file)
  - API spec        → .py  (annotated pytest + requests file)

Usage:
    from generation.test_formatter import format_and_save
    output_path = format_and_save(raw_llm_output, primary_chunk, retrieved_chunks, output_dir)
"""

from __future__ import annotations

import ast
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


def format_and_save(
    raw_output: str,
    primary_chunk: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    output_dir: str = "outputs",
) -> str:
    """
    Format the LLM's raw output and save it to a file.

    Args:
        raw_output       : Raw text returned by llm_client.generate().
        primary_chunk    : The chunk tests were generated FOR.
        retrieved_chunks : The similar chunks used as context (for traceability).
        output_dir       : Directory to save the output file (created if missing).

    Returns:
        Absolute path of the saved output file.
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata   = primary_chunk.get("metadata", {})
    input_type = str(metadata.get("type", "unknown")).lower()

    if input_type == "code":
        return _save_code_tests(raw_output, metadata, retrieved_chunks, output_dir)
    elif input_type == "requirements":
        return _save_gherkin_tests(raw_output, metadata, retrieved_chunks, output_dir)
    elif input_type == "api":
        return _save_api_tests(raw_output, metadata, retrieved_chunks, output_dir)
    else:
        # Unknown type — save as plain text
        return _save_raw(raw_output, metadata, output_dir)


def validate_python_syntax(code: str) -> Dict[str, Any]:
    """
    Check if a Python code string is syntactically valid.
    Used by Person 3's evaluator.

    Returns:
        {
            "valid"   : bool,
            "error"   : str or None,   # error message if invalid
            "lines"   : int,           # number of lines
        }
    """
    try:
        ast.parse(code)
        return {"valid": True, "error": None, "lines": len(code.splitlines())}
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"SyntaxError at line {e.lineno}: {e.msg}",
            "lines": len(code.splitlines()),
        }


# ── Code mode (.py) ────────────────────────────────────────────────────────────

def _save_code_tests(
    raw_output: str,
    metadata: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    function_name = metadata.get("function_name", "unknown")
    file_name     = metadata.get("file_name", "unknown")
    signature     = metadata.get("signature", "")

    # Clean up the raw output
    code = _clean_code_output(raw_output)

    # Build traceability footer as a comment block
    traceability = _build_python_traceability(
        source_label  = f"{file_name} :: {function_name}()",
        signature     = signature,
        metadata      = metadata,
        retrieved     = retrieved_chunks,
        input_type    = "code",
    )

    final_content = code + "\n\n" + traceability

    # Validate syntax and warn if broken (don't block saving)
    validation = validate_python_syntax(final_content)
    if not validation["valid"]:
        print(f"[Formatter] ⚠️  Syntax warning: {validation['error']}")
    else:
        print(f"[Formatter] ✅ Syntax OK ({validation['lines']} lines)")

    # Save file
    out_name = f"test_{_slugify(function_name)}.py"
    out_path = os.path.join(output_dir, out_name)
    _write_file(out_path, final_content)

    return out_path


# ── Requirements mode (.feature) ──────────────────────────────────────────────

def _save_gherkin_tests(
    raw_output: str,
    metadata: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    section   = metadata.get("section", "unknown_section")
    file_name = metadata.get("file_name", "unknown")
    page      = metadata.get("page", "?")
    chunk_idx = metadata.get("chunk_index", "?")

    content = raw_output.strip()

    # Build traceability footer as a Gherkin comment block
    traceability = _build_gherkin_traceability(
        source_label = f"{file_name} :: {section} (p.{page}, chunk {chunk_idx})",
        metadata     = metadata,
        retrieved    = retrieved_chunks,
    )

    final_content = content + "\n\n" + traceability

    out_name = f"test_{_slugify(section)}.feature"
    out_path = os.path.join(output_dir, out_name)
    _write_file(out_path, final_content)

    return out_path


# ── API mode (.py) ─────────────────────────────────────────────────────────────

def _save_api_tests(
    raw_output: str,
    metadata: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    endpoint     = metadata.get("endpoint", "unknown")
    method       = metadata.get("method", "GET")
    operation_id = metadata.get("operation_id", "")
    file_name    = metadata.get("file_name", "unknown")
    chunk_idx    = metadata.get("chunk_index", "?")

    code = _clean_code_output(raw_output)

    traceability = _build_python_traceability(
        source_label = f"{file_name} :: {method} {endpoint}",
        signature    = f"{method} {endpoint}",
        metadata     = metadata,
        retrieved    = retrieved_chunks,
        input_type   = "api",
    )

    final_content = code + "\n\n" + traceability

    validation = validate_python_syntax(final_content)
    if not validation["valid"]:
        print(f"[Formatter] ⚠️  Syntax warning: {validation['error']}")
    else:
        print(f"[Formatter] ✅ Syntax OK ({validation['lines']} lines)")

    slug     = _slugify(f"{method}_{endpoint}")
    out_name = f"test_api_{slug}.py"
    out_path = os.path.join(output_dir, out_name)
    _write_file(out_path, final_content)

    return out_path


# ── Raw fallback ───────────────────────────────────────────────────────────────

def _save_raw(
    raw_output: str,
    metadata: Dict[str, Any],
    output_dir: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(output_dir, f"generated_tests_{timestamp}.txt")
    _write_file(out_path, raw_output)
    return out_path


# ── Traceability builders ──────────────────────────────────────────────────────

def _build_python_traceability(
    source_label: str,
    signature: str,
    metadata: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    input_type: str,
) -> str:
    """Build a Python comment block listing source and retrieved context."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines     = [
        "# " + "=" * 77,
        "# TRACEABILITY REPORT",
        "# " + "=" * 77,
        f"# Generated at   : {timestamp}",
        f"# Input type      : {input_type}",
        f"# Primary source  : {source_label}",
        f"# Signature       : {signature}",
        "#",
        "# Retrieved context chunks used:",
    ]

    if retrieved:
        for r in retrieved:
            rank   = r.get("rank", "?")
            score  = r.get("score", 0.0)
            src    = r.get("source", "unknown")
            cid    = r.get("id", "unknown")
            lines.append(f"#   [{rank}] (score={score:.4f}) {src}")
            lines.append(f"#       chunk_id: {cid}")
    else:
        lines.append("#   (no retrieved context — zero-shot generation)")

    lines.append("# " + "=" * 77)
    return "\n".join(lines)


def _build_gherkin_traceability(
    source_label: str,
    metadata: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
) -> str:
    """Build a Gherkin comment block for .feature files."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines     = [
        "# " + "=" * 77,
        "# TRACEABILITY REPORT",
        "# " + "=" * 77,
        f"# Generated at  : {timestamp}",
        f"# Primary source: {source_label}",
        "#",
        "# Retrieved context chunks used:",
    ]

    if retrieved:
        for r in retrieved:
            rank  = r.get("rank", "?")
            score = r.get("score", 0.0)
            src   = r.get("source", "unknown")
            cid   = r.get("id", "unknown")
            lines.append(f"#   [{rank}] (score={score:.4f}) {src}")
            lines.append(f"#       chunk_id: {cid}")
    else:
        lines.append("#   (no retrieved context — zero-shot generation)")

    lines.append("# " + "=" * 77)
    return "\n".join(lines)


# ── Utilities ──────────────────────────────────────────────────────────────────

def _clean_code_output(text: str) -> str:
    """
    Remove stray markdown fences that the LLM might add even when instructed not to.
    Also strip leading/trailing whitespace.
    """
    text = text.strip()
    # Remove ```python, ```gherkin, or plain ``` fences
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)
    return text.strip()


def _slugify(text: str) -> str:
    """Convert a string to a safe filename slug."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:60]  # Limit length


def _write_file(path: str, content: str) -> None:
    """Write content to a file and print confirmation."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Formatter] 💾 Saved → {os.path.abspath(path)}")


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_code = """
# =============================================================================
# TRACEABILITY
# Source function : add
# Source file     : math_utils.py
# =============================================================================

import pytest
from math_utils import add

# Tests for the add() function

def test_add_positive_numbers():
    # Happy path: adding two positive integers
    assert add(2, 3) == 5

def test_add_negative_numbers():
    # Edge case: both inputs are negative
    assert add(-1, -1) == -2

def test_add_zero():
    # Edge case: adding zero
    assert add(0, 5) == 5
"""

    fake_chunk = {
        "content": "def add(a, b): return a + b",
        "metadata": {
            "type": "code",
            "function_name": "add",
            "signature": "def add(a, b)",
            "file_name": "math_utils.py",
            "language": "python",
            "return_type": "int",
            "docstring": "Return sum.",
        },
    }

    path = format_and_save(sample_code, fake_chunk, [], output_dir="/tmp/testgen_outputs")
    print(f"\nSaved to: {path}")

    validation = validate_python_syntax(sample_code)
    print(f"Syntax valid: {validation['valid']}")
