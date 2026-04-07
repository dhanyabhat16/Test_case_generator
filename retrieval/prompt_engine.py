"""
prompt_engine.py  (UPGRADED)
----------------------------
Builds mode-aware prompts using explicit, named prompting techniques.

Prompting techniques implemented (Syllabus Module 2):
  - Zero-Shot    : No examples — just instructions
  - Few-Shot     : 1-3 worked examples in the prompt before the target
  - Chain-of-Thought (CoT) : Explicit reasoning steps before final output
  - Prompt Modes : Hard prompts (fixed templates) with structured output

Usage:
    from retrieval.prompt_engine import build_prompt
    prompt = build_prompt(primary_chunk, retrieved_chunks, technique="few_shot")

    # Or use the technique selector explicitly:
    from retrieval.prompt_engine import PromptTechnique
    prompt = build_prompt(chunk, retrieved, technique=PromptTechnique.COT)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List


# ── Prompting technique selector ──────────────────────────────────────────────

class PromptTechnique(str, Enum):
    """
    Named prompting techniques — explicitly labelled for syllabus alignment.

    Zero-Shot   : No examples given. Model must rely purely on instructions.
    One-Shot    : One worked example provided.
    Few-Shot    : 2-3 worked examples provided.
    COT         : Chain-of-Thought — ask model to reason step by step.
    """
    ZERO_SHOT = "zero_shot"
    ONE_SHOT  = "one_shot"
    FEW_SHOT  = "few_shot"
    COT       = "chain_of_thought"


# ── Public API ─────────────────────────────────────────────────────────────────

def build_prompt(
    primary_chunk: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    technique: str = PromptTechnique.FEW_SHOT,
) -> str:
    """
    Build the full LLM prompt for a given chunk and its retrieved context.

    Args:
        primary_chunk    : The chunk we are generating tests FOR.
        retrieved_chunks : Similar chunks from hybrid retrieval.
        technique        : Prompting technique to use (default: few_shot).

    Returns:
        A complete prompt string ready to send to the LLM.
    """
    metadata   = primary_chunk.get("metadata", {})
    input_type = str(metadata.get("type", "unknown")).lower()

    if input_type == "code":
        return _build_code_prompt(primary_chunk, retrieved_chunks, technique)
    elif input_type == "requirements":
        return _build_requirements_prompt(primary_chunk, retrieved_chunks, technique)
    elif input_type == "api":
        return _build_api_prompt(primary_chunk, retrieved_chunks, technique)
    else:
        raise ValueError(
            f"Unknown input type: '{input_type}'. "
            "Expected 'code', 'requirements', or 'api'."
        )


# ── Mode 1: Code → PyTest unit tests ──────────────────────────────────────────

def _build_code_prompt(
    primary_chunk: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    technique: str,
) -> str:
    meta          = primary_chunk.get("metadata", {})
    function_name = meta.get("function_name", "unknown_function")
    file_name     = meta.get("file_name", "unknown_file")
    language      = meta.get("language", "python")
    docstring     = meta.get("docstring", "")
    return_type   = meta.get("return_type", "unknown")
    signature     = meta.get("signature", "")
    class_name    = meta.get("class_name", None)
    primary_code  = primary_chunk.get("content", "")

    module_name = file_name.replace(".py", "").replace(".java", "")
    if class_name:
        import_instruction = (
            f"Import using: `from {module_name} import {class_name}`\n"
            f"   Instantiate the class before calling `{function_name}`."
        )
    else:
        import_instruction = f"Import using: `from {module_name} import {function_name}`"

    similar_context = _format_similar_chunks(retrieved_chunks, mode="code")
    technique_block = _get_technique_block(technique, mode="code")

    traceability = f"""# =============================================================================
# TRACEABILITY
# Source function : {function_name}
# Source file     : {file_name}
# Generated for   : {signature}
# Prompt technique: {technique}
# ============================================================================="""

    prompt = f"""You are an expert software testing engineer. Your task is to write comprehensive PyTest unit tests.

{technique_block}

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

## Retrieved Context (similar functions from knowledge base)
{similar_context}

## Instructions

Generate comprehensive PyTest unit tests for `{function_name}`.

**Requirements:**
1. Write a complete, runnable Python test file
2. {import_instruction}
3. Cover ALL categories — label each section with comments:
   - # Happy path: Normal inputs that work correctly
   - # Edge cases: Empty inputs, zero, None, boundary values
   - # Error handling: Invalid types, expected exceptions
4. Naming: `test_<function_name>_<scenario>`
5. Use `pytest.raises()` for exceptions
6. Add a one-line comment above each test

Start the file with this traceability block:
```python
{traceability}
```

Return ONLY the Python test file. No explanations outside the code."""

    return prompt


# ── Mode 2: Requirements → BDD Gherkin ────────────────────────────────────────

def _build_requirements_prompt(
    primary_chunk: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    technique: str,
) -> str:
    meta             = primary_chunk.get("metadata", {})
    section          = meta.get("section", "Unknown Section")
    file_name        = meta.get("file_name", "unknown_file")
    page             = meta.get("page", "?")
    chunk_idx        = meta.get("chunk_index", "?")
    requirement_text = primary_chunk.get("content", "")

    similar_context  = _format_similar_chunks(retrieved_chunks, mode="requirements")
    technique_block  = _get_technique_block(technique, mode="requirements")

    prompt = f"""You are an expert QA engineer specialising in Behaviour-Driven Development (BDD).

{technique_block}

## Requirement to Test

**Source file:** {file_name}
**Section:** {section}
**Page:** {page}  |  **Chunk index:** {chunk_idx}

**Requirement text:**
{requirement_text}

## Related Requirements (retrieved context)
{similar_context}

## Instructions

Generate BDD-style acceptance tests in **Gherkin format**.

**Requirements:**
1. Use proper Gherkin: Feature, Scenario, Given, When, Then, And, But
2. Write a Feature block describing the overall capability
3. Cover all scenario types:
   - # Happy path: Normal successful flows
   - # Negative: Invalid inputs, error states
   - # Boundary: Limit values, empty states
4. Use `Scenario Outline` with `Examples` for multiple similar cases
5. Add traceability header:

```
# =============================================================================
# TRACEABILITY
# Source section  : {section}
# Source file     : {file_name}
# Prompt technique: {technique}
# =============================================================================
```

Return ONLY the .feature file. No explanations outside Gherkin syntax."""

    return prompt


# ── Mode 3: API spec → integration tests ──────────────────────────────────────

def _build_api_prompt(
    primary_chunk: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    technique: str,
) -> str:
    meta             = primary_chunk.get("metadata", {})
    endpoint         = meta.get("endpoint", "/unknown")
    method           = meta.get("method", "GET")
    summary          = meta.get("summary", "")
    operation_id     = meta.get("operation_id", "")
    file_name        = meta.get("file_name", "unknown_file")
    base_url         = meta.get("base_url", "http://localhost")
    api_title        = meta.get("api_title", "Unknown API")
    chunk_idx        = meta.get("chunk_index", "?")
    parameters       = _safe_parse(meta.get("parameters", "[]"))
    request_body     = _safe_parse(meta.get("request_body", "{}"))
    responses        = _safe_parse(meta.get("responses", "[]"))
    endpoint_content = primary_chunk.get("content", "")

    similar_context  = _format_similar_chunks(retrieved_chunks, mode="api")
    technique_block  = _get_technique_block(technique, mode="api")
    param_summary    = _format_parameters(parameters)
    response_summary = _format_responses(responses)
    body_summary     = _format_request_body(request_body)

    prompt = f"""You are an expert API testing engineer.

{technique_block}

## API Endpoint to Test

**API:** {api_title}
**Endpoint:** {method} {base_url}{endpoint}
**Operation ID:** {operation_id}
**Summary:** {summary if summary else "Not provided"}

**Specification:**
{endpoint_content}

**Parameters:**
{param_summary}

**Request Body:**
{body_summary}

**Responses:**
{response_summary}

## Related Endpoints (retrieved context)
{similar_context}

## Instructions

Generate comprehensive integration tests for `{method} {endpoint}`.

**Requirements:**
1. Complete runnable Python test file using pytest + requests
2. `BASE_URL = "{base_url}"` constant at the top
3. Cover all categories:
   - # Valid requests: Correct inputs, expected success
   - # Invalid inputs: Missing fields, wrong types, malformed JSON
   - # Boundary: Empty strings, max-length, null fields
   - # Auth: Missing/invalid tokens if auth required
   - # Response validation: Status codes + schema
4. Naming: `test_<method>_<endpoint_slug>_<scenario>`
5. Assert both status code AND response body fields
6. Add traceability header:

```python
# =============================================================================
# TRACEABILITY
# Source endpoint : {method} {endpoint}
# Source file     : {file_name}
# Prompt technique: {technique}
# =============================================================================
```

Return ONLY the Python test file."""

    return prompt


# ── Technique blocks ───────────────────────────────────────────────────────────

def _get_technique_block(technique: str, mode: str) -> str:
    """
    Return the technique-specific section of the prompt.
    This makes the prompting method EXPLICIT and named — required for syllabus.

    Techniques:
      Zero-Shot  : Pure instructions, no examples
      One-Shot   : One worked example
      Few-Shot   : Multiple worked examples
      CoT        : Chain-of-Thought reasoning steps
    """
    if technique == PromptTechnique.ZERO_SHOT:
        return _zero_shot_block()
    elif technique == PromptTechnique.ONE_SHOT:
        return _one_shot_block(mode)
    elif technique == PromptTechnique.FEW_SHOT:
        return _few_shot_block(mode)
    elif technique == PromptTechnique.COT:
        return _cot_block(mode)
    else:
        return _few_shot_block(mode)  # Default to few-shot


def _zero_shot_block() -> str:
    """Zero-shot: no examples, just instructions."""
    return """## Prompting Technique: Zero-Shot
Generate the test cases directly based on the function/requirement below.
No examples are provided — use your expertise to determine appropriate test scenarios."""


def _one_shot_block(mode: str) -> str:
    """One-shot: one worked example."""
    if mode == "code":
        return """## Prompting Technique: One-Shot

Here is one example of well-written pytest tests to guide your style:

**Example input function:**
```python
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

**Example output tests:**
```python
import pytest
from math_utils import divide

# Happy path
def test_divide_positive_numbers():
    assert divide(10, 2) == 5.0

# Edge case
def test_divide_returns_float():
    assert isinstance(divide(1, 3), float)

# Error handling
def test_divide_by_zero_raises():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(5, 0)
```

Now generate tests for the function below using this same structure:"""

    elif mode == "requirements":
        return """## Prompting Technique: One-Shot

**Example requirement:**
"Users must be able to log in with a valid email and password."

**Example Gherkin output:**
```gherkin
Feature: User Login

  Scenario: Successful login with valid credentials
    Given a user exists with email "user@example.com" and password "pass123"
    When the user submits the login form
    Then the user should be redirected to the dashboard

  Scenario: Login fails with wrong password
    Given a user exists with email "user@example.com"
    When the user submits with password "wrongpass"
    Then the user should see "Invalid credentials"
```

Now generate Gherkin tests for the requirement below:"""

    else:
        return _zero_shot_block()


def _few_shot_block(mode: str) -> str:
    """Few-shot: 2-3 worked examples."""
    if mode == "code":
        return """## Prompting Technique: Few-Shot

Study these examples. Notice: each test has a comment, uses a descriptive name, and covers a distinct scenario.

**Example 1 — Simple function:**
Input: `def add(a, b): return a + b`
Tests:
```python
# Adds two positive integers correctly
def test_add_positive_numbers():
    assert add(2, 3) == 5

# Handles negative numbers
def test_add_negative_numbers():
    assert add(-1, -2) == -3

# Adding zero returns the other value unchanged
def test_add_with_zero():
    assert add(5, 0) == 5
```

**Example 2 — Validation function:**
Input: `def is_palindrome(s): return s == s[::-1]`
Tests:
```python
# Returns True for a simple palindrome
def test_is_palindrome_true_case():
    assert is_palindrome("racecar") is True

# Returns False for non-palindrome
def test_is_palindrome_false_case():
    assert is_palindrome("hello") is False

# Empty string is a palindrome (edge case)
def test_is_palindrome_empty_string():
    assert is_palindrome("") is True
```

Now generate tests for the target function below, following the same pattern:"""

    elif mode == "requirements":
        return """## Prompting Technique: Few-Shot

**Example 1 — Login feature:**
Requirement: "Users can log in with email and password."
Output includes: happy path (correct credentials), negative (wrong password), boundary (empty fields).

**Example 2 — Search feature:**
Requirement: "Users can search products by name."
Output includes: happy path (found results), negative (no results), boundary (single char query).

Apply the same three-scenario structure (happy, negative, boundary) to the requirement below:"""

    else:
        return """## Prompting Technique: Few-Shot

**Example — GET /users/{id}:**
Tests cover: valid ID (200), invalid ID (404), missing auth (401), malformed ID (400).

Apply the same four-category structure to the endpoint below:"""


def _cot_block(mode: str) -> str:
    """
    Chain-of-Thought: ask model to reason before generating.
    Syllabus: Chain of Thought Prompting — Module 2
    """
    if mode == "code":
        return """## Prompting Technique: Chain-of-Thought (CoT)

Before writing any test code, reason through these steps:

**Step 1 — Understand the function:**
  What does it do? What are its inputs and outputs?

**Step 2 — Identify scenarios:**
  a) What inputs lead to normal/expected outputs? (happy path)
  b) What boundary values exist? (empty, zero, max, min)
  c) What should raise exceptions? (invalid types, out-of-range)

**Step 3 — Consider the retrieved context:**
  What patterns do similar functions use? Any shared utilities to import?

**Step 4 — Write the tests:**
  One test per scenario. Name clearly. Comment each test.

Now reason through these steps for the function below, then output the final test code:"""

    elif mode == "requirements":
        return """## Prompting Technique: Chain-of-Thought (CoT)

Before writing Gherkin, reason through:

**Step 1 — Parse the requirement:** What is the user trying to do?
**Step 2 — Find the happy path:** What does success look like?
**Step 3 — Find failure conditions:** What could go wrong?
**Step 4 — Find boundaries:** What are the edge cases?

Reason through these for the requirement below, then write the Gherkin:"""

    else:
        return """## Prompting Technique: Chain-of-Thought (CoT)

Reason through: (1) what this endpoint does, (2) valid vs invalid inputs,
(3) expected status codes for each case, then write the tests:"""


# ── Shared formatters (unchanged from original) ────────────────────────────────

def _format_similar_chunks(chunks: List[Dict[str, Any]], mode: str) -> str:
    if not chunks:
        return "No similar items found in the knowledge base."
    lines = []
    for chunk in chunks:
        rank    = chunk.get("rank", "?")
        score   = chunk.get("score", 0.0)
        source  = chunk.get("source", "unknown")
        method  = chunk.get("retrieval_method", "dense")
        content = chunk.get("content", "")
        preview = content[:600] + ("..." if len(content) > 600 else "")
        lines.append(f"### [Context {rank}] (score: {score:.3f}, method: {method})")
        lines.append(f"**Source:** {source}")
        lines.append("```")
        lines.append(preview)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _format_parameters(parameters: Any) -> str:
    if not parameters or not isinstance(parameters, list):
        return "  None"
    lines = []
    for p in parameters:
        if not isinstance(p, dict):
            continue
        req = "required" if p.get("required") else "optional"
        lines.append(
            f"  - {p.get('name','?')} ({p.get('in','?')}, {req}): "
            f"{p.get('description','')} "
            f"[type: {p.get('schema',{}).get('type','unknown') if isinstance(p.get('schema'),dict) else 'unknown'}]"
        )
    return "\n".join(lines) if lines else "  None"


def _format_responses(responses: Any) -> str:
    if not responses or not isinstance(responses, list):
        return "  None"
    lines = []
    for r in responses:
        if not isinstance(r, dict):
            continue
        lines.append(f"  - {r.get('status_code','?')}: {r.get('description','')}")
    return "\n".join(lines) if lines else "  None"


def _format_request_body(body: Any) -> str:
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
