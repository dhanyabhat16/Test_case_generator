"""
tests/test_ingestion.py
-----------------------
Unit tests for all three parsers and the embedder.
Run with: pytest tests/test_ingestion.py -v
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.code_parser import parse_code_file
from ingestion.api_parser import parse_api_file

SAMPLE_CODE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_code.py")
SAMPLE_API = os.path.join(os.path.dirname(__file__), "..", "data", "sample_api.yaml")


# ─────────────────────────────────────────────
# Code parser tests
# ─────────────────────────────────────────────

class TestCodeParser:

    def test_returns_list(self):
        chunks = parse_code_file(SAMPLE_CODE)
        assert isinstance(chunks, list)

    def test_non_empty(self):
        chunks = parse_code_file(SAMPLE_CODE)
        assert len(chunks) > 0, "Should parse at least one function"

    def test_chunk_has_content_and_metadata(self):
        chunks = parse_code_file(SAMPLE_CODE)
        for chunk in chunks:
            assert "content" in chunk
            assert "metadata" in chunk

    def test_chunk_content_is_string(self):
        chunks = parse_code_file(SAMPLE_CODE)
        for chunk in chunks:
            assert isinstance(chunk["content"], str)
            assert len(chunk["content"]) > 0

    def test_metadata_type_is_code(self):
        chunks = parse_code_file(SAMPLE_CODE)
        for chunk in chunks:
            assert chunk["metadata"]["type"] == "code"

    def test_metadata_language_is_python(self):
        chunks = parse_code_file(SAMPLE_CODE)
        for chunk in chunks:
            assert chunk["metadata"]["language"] == "python"

    def test_metadata_has_function_name(self):
        chunks = parse_code_file(SAMPLE_CODE)
        for chunk in chunks:
            assert "function_name" in chunk["metadata"]
            assert isinstance(chunk["metadata"]["function_name"], str)

    def test_metadata_has_file_name(self):
        chunks = parse_code_file(SAMPLE_CODE)
        for chunk in chunks:
            assert chunk["metadata"]["file_name"] == "sample_code.py"

    def test_known_functions_parsed(self):
        chunks = parse_code_file(SAMPLE_CODE)
        function_names = [c["metadata"]["function_name"] for c in chunks]
        assert "add" in function_names
        assert "divide" in function_names
        assert "fibonacci" in function_names
        assert "validate_email" in function_names

    def test_docstring_extracted(self):
        chunks = parse_code_file(SAMPLE_CODE)
        divide_chunk = next(
            (c for c in chunks if c["metadata"]["function_name"] == "divide"), None
        )
        assert divide_chunk is not None
        assert "Divide" in divide_chunk["metadata"]["docstring"]

    def test_line_numbers_present(self):
        chunks = parse_code_file(SAMPLE_CODE)
        for chunk in chunks:
            assert "start_line" in chunk["metadata"]
            assert "end_line" in chunk["metadata"]
            assert chunk["metadata"]["start_line"] <= chunk["metadata"]["end_line"]

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_code_file("/nonexistent/path/file.py")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError):
            parse_code_file("some_file.rb")


# ─────────────────────────────────────────────
# API parser tests
# ─────────────────────────────────────────────

class TestApiParser:

    def test_returns_list(self):
        chunks = parse_api_file(SAMPLE_API)
        assert isinstance(chunks, list)

    def test_non_empty(self):
        chunks = parse_api_file(SAMPLE_API)
        assert len(chunks) > 0

    def test_chunk_has_content_and_metadata(self):
        chunks = parse_api_file(SAMPLE_API)
        for chunk in chunks:
            assert "content" in chunk
            assert "metadata" in chunk

    def test_metadata_type_is_api(self):
        chunks = parse_api_file(SAMPLE_API)
        for chunk in chunks:
            assert chunk["metadata"]["type"] == "api"

    def test_metadata_has_required_fields(self):
        chunks = parse_api_file(SAMPLE_API)
        for chunk in chunks:
            meta = chunk["metadata"]
            assert "endpoint" in meta
            assert "method" in meta
            assert "file_name" in meta
            assert "api_title" in meta

    def test_methods_are_uppercase(self):
        chunks = parse_api_file(SAMPLE_API)
        for chunk in chunks:
            assert chunk["metadata"]["method"] == chunk["metadata"]["method"].upper()

    def test_known_endpoints_parsed(self):
        chunks = parse_api_file(SAMPLE_API)
        endpoints = [
            (c["metadata"]["method"], c["metadata"]["endpoint"])
            for c in chunks
        ]
        assert ("GET", "/users") in endpoints
        assert ("POST", "/users") in endpoints
        assert ("POST", "/auth/login") in endpoints
        assert ("DELETE", "/users/{userId}") in endpoints

    def test_content_includes_method_and_path(self):
        chunks = parse_api_file(SAMPLE_API)
        for chunk in chunks:
            meta = chunk["metadata"]
            assert meta["method"] in chunk["content"]
            assert meta["endpoint"] in chunk["content"]

    def test_api_title_extracted(self):
        chunks = parse_api_file(SAMPLE_API)
        for chunk in chunks:
            assert "User Management API" in chunk["metadata"]["api_title"]

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_api_file("/nonexistent/spec.yaml")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError):
            parse_api_file("spec.toml")


# ─────────────────────────────────────────────
# Embedder tests (mocked — no real ChromaDB call)
# ─────────────────────────────────────────────

class TestEmbedderHelpers:
    """Test internal helper functions without hitting ChromaDB."""

    def test_flatten_metadata_strings_pass_through(self):
        from ingestion.embedder import _flatten_metadata
        meta = {"type": "code", "function_name": "add", "start_line": 5}
        flat = _flatten_metadata(meta)
        assert flat["type"] == "code"
        assert flat["function_name"] == "add"
        assert flat["start_line"] == 5

    def test_flatten_metadata_lists_become_strings(self):
        from ingestion.embedder import _flatten_metadata
        import json
        meta = {"decorators": ["@staticmethod", "@classmethod"]}
        flat = _flatten_metadata(meta)
        assert isinstance(flat["decorators"], str)
        assert json.loads(flat["decorators"]) == ["@staticmethod", "@classmethod"]

    def test_flatten_metadata_none_becomes_empty_string(self):
        from ingestion.embedder import _flatten_metadata
        meta = {"docstring": None}
        flat = _flatten_metadata(meta)
        assert flat["docstring"] == ""

    def test_make_chunk_id_code(self):
        from ingestion.embedder import _make_chunk_id
        meta = {"type": "code", "function_name": "add", "file_name": "math.py"}
        cid = _make_chunk_id(meta)
        assert cid.startswith("code_math_py_add_")
        assert len(cid) > 10

    def test_make_chunk_id_api(self):
        from ingestion.embedder import _make_chunk_id
        meta = {"type": "api", "method": "POST", "endpoint": "/auth/login", "file_name": "api.yaml"}
        cid = _make_chunk_id(meta)
        assert cid.startswith("api_api_yaml_POST_")

    def test_make_chunk_id_requirements(self):
        from ingestion.embedder import _make_chunk_id
        meta = {"type": "requirements", "chunk_index": 3, "file_name": "srs.pdf"}
        cid = _make_chunk_id(meta)
        assert cid.startswith("requirements_srs_pdf_3_")