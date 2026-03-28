"""
api_parser.py
-------------
Parses OpenAPI / Swagger specification files (.yaml, .json) into
endpoint-level chunks. Each chunk follows the schema defined in SCHEMA.md.

Supports OpenAPI 2.0 (Swagger) and OpenAPI 3.x.

Usage:
    from ingestion.api_parser import parse_api_file
    chunks = parse_api_file("path/to/openapi.yaml")
"""

import os
import json
from typing import List, Dict, Any


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def parse_api_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse an OpenAPI YAML or JSON specification into endpoint-level chunks.

    Args:
        file_path: Absolute or relative path to the spec file.

    Returns:
        List of chunk dicts conforming to SCHEMA.md.

    Raises:
        ValueError: If the file extension is not .yaml, .yml, or .json.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    spec = _load_spec(file_path, ext)
    return _parse_openapi(spec, file_name)


# ─────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────

def _load_spec(file_path: str, ext: str) -> Dict:
    """Load YAML or JSON spec into a dict."""
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required. Run: pip install pyyaml")
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .yaml, .yml, .json are supported.")


# ─────────────────────────────────────────────
# OpenAPI parser
# ─────────────────────────────────────────────

# HTTP methods recognized in OpenAPI specs
_HTTP_METHODS = ["get", "post", "put", "patch", "delete", "options", "head", "trace"]


def _parse_openapi(spec: Dict, file_name: str) -> List[Dict[str, Any]]:
    """Parse OpenAPI 2.x / 3.x spec into endpoint chunks."""
    chunks = []
    chunk_index = 0

    # API-level metadata
    info = spec.get("info", {})
    api_title = info.get("title", "Unknown API")
    api_version = info.get("version", "unknown")
    api_description = info.get("description", "")

    # Resolve global server base URL (OpenAPI 3.x)
    base_url = _extract_base_url(spec)

    # Shared components / definitions for resolving $ref (basic)
    components = _extract_components(spec)

    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        # Path-level parameters (shared across methods)
        path_params = path_item.get("parameters", [])

        for method in _HTTP_METHODS:
            operation = path_item.get(method)
            if not operation or not isinstance(operation, dict):
                continue

            chunk = _build_endpoint_chunk(
                path=path,
                method=method.upper(),
                operation=operation,
                path_params=path_params,
                components=components,
                api_title=api_title,
                api_version=api_version,
                base_url=base_url,
                chunk_index=chunk_index,
                file_name=file_name,
            )
            chunks.append(chunk)
            chunk_index += 1

    return chunks


def _build_endpoint_chunk(
    path: str,
    method: str,
    operation: Dict,
    path_params: List,
    components: Dict,
    api_title: str,
    api_version: str,
    base_url: str,
    chunk_index: int,
    file_name: str,
) -> Dict[str, Any]:
    """Build a single endpoint chunk from an OpenAPI operation object."""
    summary = operation.get("summary", "")
    description = operation.get("description", "")
    operation_id = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
    tags = operation.get("tags", [])

    # Merge path-level and operation-level parameters
    all_params = path_params + operation.get("parameters", [])
    parameters = _extract_parameters(all_params, components)

    # Request body (OpenAPI 3.x)
    request_body = _extract_request_body(operation.get("requestBody", {}), components)

    # Responses
    responses = _extract_responses(operation.get("responses", {}), components)

    # Security
    security = operation.get("security", [])
    security_str = _format_security(security)

    # Build human-readable content for embedding
    content = _format_endpoint_content(
        method=method,
        path=path,
        base_url=base_url,
        summary=summary,
        description=description,
        parameters=parameters,
        request_body=request_body,
        responses=responses,
        security_str=security_str,
    )

    return {
        "content": content,
        "metadata": {
            "type": "api",
            "endpoint": path,
            "method": method,
            "operation_id": operation_id,
            "summary": summary,
            "tags": tags,
            "base_url": base_url,
            "api_title": api_title,
            "api_version": api_version,
            "chunk_index": chunk_index,
            "file_name": file_name,
            # Structured data for prompt construction
            "parameters": parameters,
            "request_body": request_body,
            "responses": responses,
        }
    }


# ─────────────────────────────────────────────
# Parameter / response extractors
# ─────────────────────────────────────────────

def _extract_parameters(params: List, components: Dict) -> List[Dict]:
    """Extract and normalize parameter list, resolving $ref if needed."""
    result = []
    for param in params:
        param = _resolve_ref(param, components)
        if not isinstance(param, dict):
            continue
        result.append({
            "name": param.get("name", "unknown"),
            "in": param.get("in", "unknown"),      # path, query, header, cookie
            "required": param.get("required", False),
            "description": param.get("description", ""),
            "schema": _simplify_schema(param.get("schema", {})),
        })
    return result


def _extract_request_body(body: Dict, components: Dict) -> Dict:
    """Extract request body content and schema."""
    if not body:
        return {}

    body = _resolve_ref(body, components)
    content = body.get("content", {})
    required = body.get("required", False)
    description = body.get("description", "")

    # Pick the first media type (usually application/json)
    media_types = list(content.keys())
    schema = {}
    if media_types:
        first_media = content[media_types[0]]
        schema = _simplify_schema(first_media.get("schema", {}))

    return {
        "required": required,
        "description": description,
        "media_types": media_types,
        "schema": schema,
    }


def _extract_responses(responses: Dict, components: Dict) -> List[Dict]:
    """Extract response status codes and descriptions."""
    result = []
    for status_code, response in responses.items():
        response = _resolve_ref(response, components)
        if not isinstance(response, dict):
            continue

        content = response.get("content", {})
        schema = {}
        media_types = list(content.keys())
        if media_types:
            first = content[media_types[0]]
            schema = _simplify_schema(first.get("schema", {}))

        result.append({
            "status_code": str(status_code),
            "description": response.get("description", ""),
            "schema": schema,
        })
    return result


def _simplify_schema(schema: Dict) -> Dict:
    """Return a simplified schema dict (type, properties, items, example)."""
    if not schema or not isinstance(schema, dict):
        return {}
    return {
        "type": schema.get("type", "object"),
        "properties": list(schema.get("properties", {}).keys()),
        "required_fields": schema.get("required", []),
        "example": schema.get("example"),
    }


def _resolve_ref(obj: Any, components: Dict) -> Any:
    """Resolve a $ref pointer to its actual definition in components."""
    if not isinstance(obj, dict) or "$ref" not in obj:
        return obj

    ref_path = obj["$ref"]  # e.g. "#/components/schemas/User"
    parts = ref_path.lstrip("#/").split("/")

    current = components
    for part in parts[1:]:  # skip "components" or "definitions" prefix
        if isinstance(current, dict):
            current = current.get(part, {})
        else:
            return obj  # can't resolve, return original

    return current if current else obj


def _extract_components(spec: Dict) -> Dict:
    """Extract component definitions (OpenAPI 3.x) or definitions (2.x)."""
    # OpenAPI 3.x
    if "components" in spec:
        return spec["components"]
    # OpenAPI 2.x (Swagger)
    if "definitions" in spec:
        return {"schemas": spec["definitions"]}
    return {}


def _extract_base_url(spec: Dict) -> str:
    """Extract the base URL from the spec."""
    # OpenAPI 3.x
    servers = spec.get("servers", [])
    if servers and isinstance(servers, list):
        return servers[0].get("url", "")
    # OpenAPI 2.x
    host = spec.get("host", "")
    base_path = spec.get("basePath", "")
    schemes = spec.get("schemes", ["https"])
    scheme = schemes[0] if schemes else "https"
    if host:
        return f"{scheme}://{host}{base_path}"
    return base_path


def _format_security(security: List) -> str:
    """Format security requirements into a readable string."""
    if not security:
        return "None"
    parts = []
    for item in security:
        if isinstance(item, dict):
            parts.extend(item.keys())
    return ", ".join(parts) if parts else "None"


# ─────────────────────────────────────────────
# Content formatter (for embedding)
# ─────────────────────────────────────────────

def _format_endpoint_content(
    method: str,
    path: str,
    base_url: str,
    summary: str,
    description: str,
    parameters: List[Dict],
    request_body: Dict,
    responses: List[Dict],
    security_str: str,
) -> str:
    """Format endpoint details into a human-readable string for embedding."""
    lines = [
        f"Endpoint: {method} {base_url}{path}",
        f"Summary: {summary}" if summary else "",
        f"Description: {description}" if description else "",
        f"Security: {security_str}",
        "",
    ]

    if parameters:
        lines.append("Parameters:")
        for p in parameters:
            req = "required" if p["required"] else "optional"
            lines.append(
                f"  - {p['name']} ({p['in']}, {req}): {p['description']} "
                f"[type: {p['schema'].get('type', 'unknown')}]"
            )

    if request_body:
        lines.append("")
        lines.append(f"Request Body ({', '.join(request_body.get('media_types', ['unknown']))}):")
        lines.append(f"  Required: {request_body.get('required', False)}")
        schema = request_body.get("schema", {})
        if schema.get("properties"):
            lines.append(f"  Fields: {', '.join(schema['properties'])}")
        if schema.get("example"):
            lines.append(f"  Example: {schema['example']}")

    if responses:
        lines.append("")
        lines.append("Responses:")
        for r in responses:
            lines.append(f"  - {r['status_code']}: {r['description']}")

    return "\n".join(line for line in lines if line is not None)


# ─────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python api_parser.py <file_path>")
        sys.exit(1)

    path = sys.argv[1]
    result = parse_api_file(path)
    print(f"✅ Parsed {len(result)} endpoint(s) from {path}\n")
    for i, chunk in enumerate(result):
        m = chunk["metadata"]
        print(f"--- Endpoint {i + 1}: {m['method']} {m['endpoint']} ---")
        print(chunk["content"][:400])
        print()