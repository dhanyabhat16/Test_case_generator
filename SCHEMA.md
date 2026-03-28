# SCHEMA.md — Chunk Format Contract

> **Written by: Person 1**
> This file defines the exact structure of every chunk produced by the ingestion pipeline.
> Person 2 (retrieval/generation) and Person 3 (evaluation/UI) must read this before writing any code.

---

## Overview

Every chunk is a Python `dict` with exactly two top-level keys:

```python
{
    "content": str,       # The text that gets embedded into the vector store
    "metadata": dict      # Structured info about the chunk (used for filtering + traceability)
}
```

The `content` field is what ChromaDB embeds and searches over.
The `metadata` field is used by Person 2's retriever to filter by type and by Person 3's UI for traceability display.

---

## Mode 1: Source Code Chunk

**Produced by:** `ingestion/code_parser.py`
**Stored with:** `metadata.type = "code"`

```python
{
    "content": "def add(a: float, b: float) -> float:\n    \"\"\"Return the sum.\"\"\"\n    return a + b",
    "metadata": {
        "type": "code",                     # ALWAYS "code" for this mode
        "language": "python",               # "python" or "java"
        "function_name": "add",             # name of the function
        "signature": "def add(a, b)",       # cleaned signature string
        "docstring": "Return the sum.",     # extracted docstring (empty string if none)
        "return_type": "float",             # return type annotation ("unknown" if missing)
        "decorators": [],                   # list of decorator strings e.g. ["@staticmethod"]
        "start_line": 5,                    # line number in source file
        "end_line": 8,
        "file_name": "math_utils.py",       # basename of the source file
        # Java-only fields (absent for Python):
        "class_name": "MathUtils",          # Java class the method belongs to
        "modifiers": "public static",       # Java modifiers
    }
}
```

---

## Mode 2: Requirements Document Chunk

**Produced by:** `ingestion/doc_parser.py`
**Stored with:** `metadata.type = "requirements"`

```python
{
    "content": "The system shall validate user credentials before granting access to any protected resource. Failed login attempts must be logged with timestamp and IP address.",
    "metadata": {
        "type": "requirements",             # ALWAYS "requirements" for this mode
        "section": "3.2 Authentication",   # heading/section the chunk belongs to
        "page": 4,                          # page number (PDF) or estimated page (DOCX)
        "chunk_index": 7,                   # sequential index within this file
        "file_name": "srs.pdf",             # basename of the source file
        "source_format": "pdf",             # "pdf" or "docx"
    }
}
```

---

## Mode 3: API Specification Chunk

**Produced by:** `ingestion/api_parser.py`
**Stored with:** `metadata.type = "api"`

```python
{
    "content": "Endpoint: POST https://api.example.com/v1/auth/login\nSummary: Authenticate user\nDescription: Validates credentials and returns a JWT token.\nSecurity: None\n\nParameters:\n  (none)\n\nRequest Body (application/json):\n  Required: True\n  Fields: email, password\n\nResponses:\n  - 200: Login successful\n  - 401: Invalid credentials\n  - 429: Too many login attempts",
    "metadata": {
        "type": "api",                              # ALWAYS "api" for this mode
        "endpoint": "/auth/login",                  # path from OpenAPI spec
        "method": "POST",                           # HTTP method (uppercase)
        "operation_id": "loginUser",                # operationId from spec
        "summary": "Authenticate user",             # summary string
        "tags": ["Auth"],                           # JSON string of tags list
        "base_url": "https://api.example.com/v1",   # server base URL
        "api_title": "User Management API",         # info.title from spec
        "api_version": "1.0.0",                     # info.version from spec
        "chunk_index": 5,                           # sequential index
        "file_name": "sample_api.yaml",             # basename of the source file
        # Structured fields (stored as JSON strings in ChromaDB):
        "parameters": "[]",                         # JSON: list of param dicts
        "request_body": "{...}",                    # JSON: request body dict
        "responses": "[{...}]",                     # JSON: list of response dicts
    }
}
```

---

## ChromaDB Storage Details

| Property | Value |
|---|---|
| Collection name | `testgen_store` |
| Similarity metric | Cosine |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Embedding dim | 384 |
| Persistence path | `./chroma_db/` (gitignored) |

### Filtering by type (how Person 2 queries)

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("testgen_store")

# Filter to only code chunks
results = collection.query(
    query_texts=["function that validates email"],
    n_results=5,
    where={"type": "code"}              # <-- filter by metadata
)

# Filter to only requirement chunks
results = collection.query(
    query_texts=["authentication requirement"],
    n_results=5,
    where={"type": "requirements"}
)

# Filter to only API chunks
results = collection.query(
    query_texts=["login endpoint"],
    n_results=5,
    where={"type": "api"}
)
```

---

## Notes for Person 2

1. **ChromaDB stores list/dict metadata as JSON strings.** When you read back `parameters`, `responses`, or `decorators` from metadata, parse them with `json.loads()`.
2. **The `content` field is always plain text** — no markdown, no special formatting.
3. **`chunk_index` is per-file**, not globally unique. Use the full chunk ID (from ChromaDB's `ids` field) for global uniqueness.
4. **Traceability:** The ChromaDB result includes `ids` (our generated IDs). Pass these through to Person 3 for the UI traceability panel.

## Notes for Person 3

The traceability footer added by `test_formatter.py` will reference:
- `metadata.file_name` — which file the test was generated from
- `metadata.function_name` / `metadata.endpoint` / `metadata.section` — which specific element
- The ChromaDB chunk ID — for exact lookup if needed