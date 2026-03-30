"""
test_evaluator.py
-----------------
Unit tests for evaluation/evaluator.py.
Run with: pytest tests/test_evaluator.py -v
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import (
    _check_parse_rate,
    _check_exec_rate,
    _strip_traceability_comments,
    evaluate_file,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_test_file(tmp_path):
    """A syntactically valid, executable pytest file."""
    content = '''\
import pytest

def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 5 - 3 == 2

def test_string():
    assert "hello".upper() == "HELLO"
'''
    path = tmp_path / "test_valid.py"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def invalid_syntax_file(tmp_path):
    """A file with a Python syntax error."""
    content = '''\
def test_broken(:
    assert True
'''
    path = tmp_path / "test_broken.py"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def file_with_traceability(tmp_path):
    """A valid test file with a traceability comment block appended."""
    content = '''\
import pytest

def test_add():
    assert 1 + 1 == 2

# ===========================================================================
# TRACEABILITY REPORT
# ===========================================================================
# Generated at   : 2024-01-01 12:00:00
# Input type      : code
# Primary source  : math_utils.py :: add()
# ===========================================================================
'''
    path = tmp_path / "test_with_trace.py"
    path.write_text(content, encoding="utf-8")
    return str(path)


# ── Parse Rate tests ───────────────────────────────────────────────────────────

class TestCheckParseRate:

    def test_valid_file_returns_rate_one(self, valid_test_file):
        result = _check_parse_rate(valid_test_file)
        assert result["rate"] == 1.0
        assert result["error"] is None

    def test_invalid_file_returns_rate_zero(self, invalid_syntax_file):
        result = _check_parse_rate(invalid_syntax_file)
        assert result["rate"] == 0.0
        assert result["error"] is not None
        assert "SyntaxError" in result["error"]

    def test_file_with_traceability_passes(self, file_with_traceability):
        result = _check_parse_rate(file_with_traceability)
        assert result["rate"] == 1.0

    def test_nonexistent_file_returns_error(self, tmp_path):
        result = _check_parse_rate(str(tmp_path / "does_not_exist.py"))
        assert result["rate"] == 0.0
        assert result["error"] is not None


# ── Exec Rate tests ────────────────────────────────────────────────────────────

class TestCheckExecRate:

    def test_valid_file_is_collectable(self, valid_test_file):
        result = _check_exec_rate(valid_test_file)
        # pytest can collect it → exec_rate == 1.0
        assert result["rate"] == 1.0

    def test_invalid_syntax_cannot_be_collected(self, invalid_syntax_file):
        result = _check_exec_rate(invalid_syntax_file)
        assert result["rate"] == 0.0


# ── Traceability stripping ─────────────────────────────────────────────────────

class TestStripTraceabilityComments:

    def test_strips_traceability_block(self):
        source = '''\
def test_foo():
    assert True

# =============================================
# TRACEABILITY REPORT
# =============================================
# Generated at: 2024-01-01
# =============================================
'''
        stripped = _strip_traceability_comments(source)
        assert "TRACEABILITY" not in stripped
        assert "def test_foo" in stripped

    def test_no_traceability_unchanged(self):
        source = "def test_bar():\n    assert 1 == 1\n"
        stripped = _strip_traceability_comments(source)
        assert "def test_bar" in stripped

    def test_stripped_code_is_valid_python(self):
        source = '''\
import pytest

def test_add():
    assert 1 + 1 == 2

# =============================================
# TRACEABILITY REPORT
# =============================================
# Primary source: math.py :: add()
# =============================================
'''
        stripped = _strip_traceability_comments(source)
        # Should not raise
        ast.parse(stripped)


# ── evaluate_file integration ──────────────────────────────────────────────────

class TestEvaluateFile:

    def test_valid_file_full_evaluation(self, valid_test_file):
        result = evaluate_file(valid_test_file)

        assert result["test_file"]   == valid_test_file
        assert result["parse_rate"]  == 1.0
        assert result["exec_rate"]   == 1.0
        assert result["status"]      == "ok"
        assert result["parse_error"] is None

    def test_invalid_file_stops_at_parse(self, invalid_syntax_file):
        result = evaluate_file(invalid_syntax_file)

        assert result["parse_rate"] == 0.0
        assert result["exec_rate"]  == 0.0   # not attempted
        assert result["status"]     == "parse_error"
        assert result["parse_error"] is not None

    def test_result_has_all_required_keys(self, valid_test_file):
        result = evaluate_file(valid_test_file)
        required_keys = {
            "test_file", "parse_rate", "exec_rate",
            "coverage", "parse_error", "exec_error", "status"
        }
        assert required_keys.issubset(result.keys())

    def test_coverage_none_without_source_file(self, valid_test_file):
        result = evaluate_file(valid_test_file, source_file=None)
        assert result["coverage"] is None

    def test_nonexistent_test_file(self, tmp_path):
        result = evaluate_file(str(tmp_path / "ghost.py"))
        assert result["parse_rate"] == 0.0
        assert result["status"]     == "parse_error"