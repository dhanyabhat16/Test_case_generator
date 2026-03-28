"""
code_parser.py
--------------
Parses Python (.py) and Java (.java) source files into function-level chunks.
Each chunk follows the schema defined in SCHEMA.md.

Usage:
    from ingestion.code_parser import parse_code_file
    chunks = parse_code_file("path/to/file.py")
"""

import ast
import re
import os
from typing import List, Dict, Any


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def parse_code_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a Python or Java source file and return a list of chunk dicts.

    Args:
        file_path: Absolute or relative path to the source file.

    Returns:
        List of chunk dicts conforming to SCHEMA.md.

    Raises:
        ValueError: If the file extension is not .py or .java.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    if ext == ".py":
        return _parse_python(file_path, file_name)
    elif ext == ".java":
        return _parse_java(file_path, file_name)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .py and .java are supported.")


# ─────────────────────────────────────────────
# Python parser (AST-based)
# ─────────────────────────────────────────────

def _parse_python(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    """Parse Python file using the built-in ast module."""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Python syntax error in {file_path}: {e}")

    lines = source.splitlines()
    chunks = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private/dunder methods if nested too deep (keep top-level + class methods)
            chunk = _extract_python_function(node, lines, file_name)
            if chunk:
                chunks.append(chunk)

    return chunks


def _extract_python_function(
    node: ast.FunctionDef,
    lines: List[str],
    file_name: str
) -> Dict[str, Any]:
    """Extract a single Python function into a chunk dict."""
    # Function signature
    args = []
    for arg in node.args.args:
        args.append(arg.arg)
    signature = f"def {node.name}({', '.join(args)})"

    # Docstring
    docstring = ast.get_docstring(node) or ""

    # Full function body (raw source lines)
    start_line = node.lineno - 1        # ast is 1-indexed
    end_line = node.end_lineno          # inclusive
    body = "\n".join(lines[start_line:end_line])

    # Decorators
    decorators = [ast.unparse(d) for d in node.decorator_list]

    # Return annotation
    return_type = ast.unparse(node.returns) if node.returns else "unknown"

    content = body  # full function source is the content for embedding

    return {
        "content": content,
        "metadata": {
            "type": "code",
            "language": "python",
            "function_name": node.name,
            "signature": signature,
            "docstring": docstring,
            "return_type": return_type,
            "decorators": decorators,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "file_name": file_name,
        }
    }


# ─────────────────────────────────────────────
# Java parser (regex-based)
# ─────────────────────────────────────────────

# Matches standard Java method declarations (public/private/protected/static etc.)
_JAVA_METHOD_PATTERN = re.compile(
    r'(?P<javadoc>/\*\*.*?\*/\s*)?'          # optional Javadoc comment
    r'(?P<modifiers>(?:(?:public|private|protected|static|final|abstract|synchronized|native|strictfp)\s+)*)'
    r'(?P<return_type>[\w<>\[\].,\s]+?)\s+'  # return type
    r'(?P<name>\w+)\s*'                       # method name
    r'\((?P<params>[^)]*)\)\s*'              # parameters
    r'(?:throws\s+[\w,\s]+)?\s*'            # optional throws
    r'\{',                                    # opening brace
    re.DOTALL
)

_JAVA_CLASS_PATTERN = re.compile(
    r'(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)'
)


def _parse_java(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    """Parse Java file using regex-based method extraction."""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Detect class name
    class_match = _JAVA_CLASS_PATTERN.search(source)
    class_name = class_match.group(1) if class_match else "UnknownClass"

    chunks = []
    lines = source.splitlines()

    for match in _JAVA_METHOD_PATTERN.finditer(source):
        method_name = match.group("name")
        params_raw = match.group("params").strip()
        return_type = match.group("return_type").strip()
        modifiers = match.group("modifiers").strip()
        javadoc = (match.group("javadoc") or "").strip()

        # Extract method body by counting braces from the opening {
        body_start = match.end() - 1  # position of opening {
        body = _extract_java_body(source, body_start)

        # Compute line numbers
        start_line = source[:match.start()].count("\n") + 1
        end_line = start_line + body.count("\n")

        # Build signature
        signature = f"{modifiers} {return_type} {method_name}({params_raw})".strip()

        # Clean docstring from javadoc
        docstring = _clean_javadoc(javadoc)

        content = signature + "\n" + body

        chunks.append({
            "content": content,
            "metadata": {
                "type": "code",
                "language": "java",
                "function_name": method_name,
                "class_name": class_name,
                "signature": signature,
                "docstring": docstring,
                "return_type": return_type,
                "modifiers": modifiers,
                "start_line": start_line,
                "end_line": end_line,
                "file_name": file_name,
            }
        })

    return chunks


def _extract_java_body(source: str, opening_brace_pos: int) -> str:
    """Extract the full method body by counting balanced braces."""
    depth = 0
    i = opening_brace_pos
    start = i

    while i < len(source):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
        i += 1

    return source[start:]  # fallback: return rest of file


def _clean_javadoc(javadoc: str) -> str:
    """Strip Javadoc comment markers to get plain text."""
    if not javadoc:
        return ""
    # Remove /** */ and leading * on each line
    cleaned = re.sub(r"/\*\*|\*/", "", javadoc)
    cleaned = re.sub(r"^\s*\*", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


# ─────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python code_parser.py <file_path>")
        sys.exit(1)

    path = sys.argv[1]
    result = parse_code_file(path)
    print(f"✅ Parsed {len(result)} function(s) from {path}\n")
    for i, chunk in enumerate(result):
        print(f"--- Chunk {i + 1}: {chunk['metadata']['function_name']} ---")
        print(json.dumps(chunk['metadata'], indent=2))
        print()