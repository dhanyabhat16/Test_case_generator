"""
prompt_engine.py
----------------
Builds mode-aware prompts for the LLM based on the input type.
Three modes: code → PyTest, requirements → BDD Gherkin, api → integration tests.

Usage:
    from retrieval.prompt_engine import build_prompt
    prompt = build_prompt(primary_chunk, retrieved_chunks)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def build_prompt(
    primary_chunk: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Build the full LLM prompt for a given chunk and its retrieved context.

    Args:
        primary_chunk    : The chunk we are generating tests FOR.
                           Must have 'content' and 'metadata' keys.
        retrieved_chunks : Similar chunks retrieved from the FAISS store
                           (output of retriever.retrieve()).

    Returns:
        A complete prompt string ready to send to the LLM.

    Raises:
        ValueError: If the metadata 'type' is not recognised.
    """
    metadata   = primary_chunk.get("metadata", {})
    input_type = str(metadata.get("type", "unknown")).lower()

    if input_type == "code":
        return _build_code_prompt(primary_chunk, retrieved_chunks)
    elif input_type == "requirements":
        return _build_requirements_prompt(primary_chunk, retrieved_chunks)
    elif input_type == "api":
        return _build_api_prompt(primary_chunk, retrieved_chunks)
    else:
        raise ValueError(
            f"Unknown input type: '{input_type}'. "
            "Expected 'code', 'requirements', or 'api'."
        )


# ── Mode 1: Code → PyTest unit tests ──────────────────────────────────────────

def _build_code_prompt(
    primary_chunk: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Build a prompt to generate PyTest unit tests for a Python/Java function.
    """
    meta          = primary_chunk.get("metadata", {})
    function_name = meta.get("function_name", "unknown_function")
    file_name     = meta.get("file_name", "unknown_file")
    language      = meta.get("language", "python")
    docstring     = meta.get("docstring", "")
    return_type   = meta.get("return_type", "unknown")
    signature     = meta.get("signature", "")
    class_name    = meta.get("class_name", None)  # None for module-level functions

    primary_code = primary_chunk.get("content", "")

    # Build the correct import instruction based on whether this is a
    # class method or a standalone function.
    module_name = file_name.replace(".py", "").replace(".java", "")
    if class_name:
        import_instruction = (
            f"Import using: `from {module_name} import {class_name}`\n"
            f"   The function `{function_name}` is a method of class `{class_name}`.\n"
            f"   Instantiate the class with its required arguments before calling the method."
        )
    else:
        import_instruction = (
            f"Import using: `from {module_name} import {function_name}`\n"
            f"   The function `{function_name}` is a standalone module-level function."
        )

    # Format similar functions as context
    similar_context = _format_similar_chunks(retrieved_chunks, mode="code")

    prompt = f"""You are an expert software testing engineer. Your task is to write comprehensive PyTest unit tests.

## Function to Test

**File:** {file_name}
**Language:** {language}
**Signature:** {signature}
**Return type:** {return_type}
**Docstring:** {docstring if docstring else "None provided"}

**Full implementation:**
```{language}
{primary_code}
```

## Similar Functions for Context
{similar_context}

## Your Task

Generate comprehensive PyTest unit tests for the function `{function_name}` above.

**Requirements:**
1. Write a complete, runnable Python test file
2. {import_instruction}
3. Cover ALL of the following test categories — use comments to label each section:
   - **Happy path**: Normal inputs that should work correctly
   - **Edge cases**: Empty inputs, zero, None, boundary values, single items
   - **Error handling**: Invalid types, out-of-range values, expected exceptions
4. Each test function must follow the naming convention: `test_<function_name>_<scenario>`
5. Use `pytest.raises()` for testing exceptions
6. Add a one-line comment above each test explaining what it checks
7. Do NOT use any mocking unless absolutely necessary
8. Add a traceability comment block at the top of the file in this exact format:

```python
# =============================================================================
# TRACEABILITY
# Source function : {function_name}
# Source file     : {file_name}
# Generated for   : {signature}
# =============================================================================
```

Return ONLY the Python test file contents. No explanations outside the code."""

    return prompt


# ── Mode 2: Requirements → BDD Gherkin acceptance tests ───────────────────────

def _build_requirements_prompt(
    primary_chunk: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Build a prompt to generate BDD-style Gherkin acceptance tests from a requirement.
    """
    meta       = primary_chunk.get("metadata", {})
    section    = meta.get("section", "Unknown Section")
    file_name  = meta.get("file_name", "unknown_file")
    page       = meta.get("page", "?")
    chunk_idx  = meta.get("chunk_index", "?")

    requirement_text = primary_chunk.get("content", "")

    # Format related requirements as context
    similar_context = _format_similar_chunks(retrieved_chunks, mode="requirements")

    prompt = f"""You are an expert QA engineer specialising in Behaviour-Driven Development (BDD).

## Requirement to Test

**Source file:** {file_name}
**Section:** {section}
**Page:** {page}  |  **Chunk index:** {chunk_idx}

**Requirement text:**
{requirement_text}

## Related Requirements (for context)
{similar_context}

## Your Task

Generate BDD-style acceptance test cases in **Gherkin format** for the requirement above.

**Requirements:**
1. Use proper Gherkin syntax: Feature, Scenario (or Scenario Outline), Given, When, Then, And, But
2. Write a **Feature block** that describes the overall capability being tested
3. Cover ALL of the following scenario types — use comments to label each group:
   - **Happy path scenarios**: Normal successful flows
   - **Negative scenarios**: Invalid inputs, failed conditions, error states
   - **Boundary scenarios**: Limit values, empty states, maximum capacity
4. Each scenario must have a clear, descriptive title
5. Use `Scenario Outline` with `Examples` table where multiple similar cases exist
6. Add a traceability comment at the top in this exact format:

```
# =============================================================================
# TRACEABILITY
# Source section  : {section}
# Source file     : {file_name}
# Page            : {page}
# Chunk index     : {chunk_idx}
# =============================================================================
```

Return ONLY the .feature file contents. No explanations outside the Gherkin syntax."""

    return prompt


# ── Mode 3: API spec → integration tests ──────────────────────────────────────

def _build_api_prompt(
    primary_chunk: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Build a prompt to generate integration test cases for an API endpoint.
    """
    meta         = primary_chunk.get("metadata", {})
    endpoint     = meta.get("endpoint", "/unknown")
    method       = meta.get("method", "GET")
    summary      = meta.get("summary", "")
    operation_id = meta.get("operation_id", "")
    file_name    = meta.get("file_name", "unknown_file")
    base_url     = meta.get("base_url", "http://localhost")
    api_title    = meta.get("api_title", "Unknown API")
    chunk_idx    = meta.get("chunk_index", "?")

    # Parse structured fields back from JSON strings
    parameters   = _safe_parse(meta.get("parameters", "[]"))
    request_body = _safe_parse(meta.get("request_body", "{}"))
    responses    = _safe_parse(meta.get("responses", "[]"))

    endpoint_content = primary_chunk.get("content", "")

    # Format related endpoints as context
    similar_context = _format_similar_chunks(retrieved_chunks, mode="api")

    # Build human-readable parameter and response summaries
    param_summary    = _format_parameters(parameters)
    response_summary = _format_responses(responses)
    body_summary     = _format_request_body(request_body)

    prompt = f"""You are an expert API testing engineer. Your task is to write comprehensive integration test cases.

## API Endpoint to Test

**API:** {api_title}
**File:** {file_name}
**Endpoint:** {method} {base_url}{endpoint}
**Operation ID:** {operation_id}
**Summary:** {summary if summary else "Not provided"}

**Full endpoint specification:**
{endpoint_content}

**Parameters:**
{param_summary}

**Request Body:**
{body_summary}

**Responses:**
{response_summary}

## Related Endpoints (for context)
{similar_context}

## Your Task

Generate comprehensive integration test cases for the `{method} {endpoint}` endpoint above.

**Requirements:**
1. Write a complete, runnable Python test file using `pytest` and `requests`
2. Use a `BASE_URL` constant at the top of the file (set to `"{base_url}"`)
3. Cover ALL of the following test categories — use comments to label each section:
   - **Valid requests**: Correct inputs, expected successful responses
   - **Invalid inputs**: Missing required fields, wrong data types, malformed JSON
   - **Boundary conditions**: Empty strings, max-length values, null fields
   - **Authentication/Authorization**: Missing tokens, invalid tokens (if auth is required)
   - **Response validation**: Correct status codes, response schema validation
4. Each test function must follow: `test_<method>_<endpoint_slug>_<scenario>`
5. Assert both the HTTP status code AND key response body fields
6. Add a traceability comment block at the top in this exact format:

```python
# =============================================================================
# TRACEABILITY
# Source endpoint : {method} {endpoint}
# Source file     : {file_name}
# Operation ID    : {operation_id}
# Chunk index     : {chunk_idx}
# =============================================================================
```

Return ONLY the Python test file contents. No explanations outside the code."""

    return prompt


# ── Shared formatters ──────────────────────────────────────────────────────────

def _format_similar_chunks(
    chunks: List[Dict[str, Any]],
    mode: str,
) -> str:
    """Format retrieved chunks into a readable context block for the prompt."""
    if not chunks:
        return "No similar items found in the knowledge base."

    lines = []
    for chunk in chunks:
        rank    = chunk.get("rank", "?")
        score   = chunk.get("score", 0.0)
        source  = chunk.get("source", "unknown")
        content = chunk.get("content", "")

        preview = content[:600] + ("..." if len(content) > 600 else "")

        lines.append(f"### [Context {rank}] (similarity: {score:.3f})")
        lines.append(f"**Source:** {source}")
        lines.append(f"```")
        lines.append(preview)
        lines.append(f"```")
        lines.append("")

    return "\n".join(lines)


def _format_parameters(parameters: Any) -> str:
    """Format API parameters list into a readable string."""
    if not parameters or not isinstance(parameters, list):
        return "  None"
    lines = []
    for p in parameters:
        if not isinstance(p, dict):
            continue
        req = "required" if p.get("required") else "optional"
        lines.append(
            f"  - {p.get('name', '?')} "
            f"({p.get('in', '?')}, {req}): "
            f"{p.get('description', '')} "
            f"[type: {p.get('schema', {}).get('type', 'unknown') if isinstance(p.get('schema'), dict) else 'unknown'}]"
        )
    return "\n".join(lines) if lines else "  None"


def _format_responses(responses: Any) -> str:
    """Format API responses list into a readable string."""
    if not responses or not isinstance(responses, list):
        return "  None"
    lines = []
    for r in responses:
        if not isinstance(r, dict):
            continue
        lines.append(
            f"  - {r.get('status_code', '?')}: {r.get('description', '')}"
        )
    return "\n".join(lines) if lines else "  None"


def _format_request_body(body: Any) -> str:
    """Format API request body dict into a readable string."""
    if not body or not isinstance(body, dict):
        return "  None"
    media_types = body.get("media_types", [])
    required    = body.get("required", False)
    schema      = body.get("schema", {})
    fields      = schema.get("properties", []) if isinstance(schema, dict) else []
    lines = [
        f"  Media type : {', '.join(media_types) if media_types else 'unknown'}",
        f"  Required   : {required}",
        f"  Fields     : {', '.join(fields) if fields else 'unknown'}",
    ]
    return "\n".join(lines)


def _safe_parse(value: Any) -> Any:
    """
    Parse a value that may be a JSON string.
    Returns the original value if it's already a list/dict or parsing fails.
    """
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                pass
    return value


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fake_chunk = {
        "content": "def add(a: int, b: int) -> int:\n    return a + b",
        "metadata": {
            "type"          : "code",
            "function_name" : "add",
            "signature"     : "def add(a, b)",
            "return_type"   : "int",
            "docstring"     : "Return the sum of a and b.",
            "language"      : "python",
            "file_name"     : "math_utils.py",
            "class_name"    : None,
        },
    }
    prompt = build_prompt(fake_chunk, [])
    print("=== Generated Prompt (code mode) ===")
    print(prompt[:800])
    print("...")