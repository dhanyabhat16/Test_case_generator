"""
doc_parser.py
-------------
Parses requirements documents (.pdf, .docx) into paragraph-level chunks.
Each chunk follows the schema defined in SCHEMA.md.

Usage:
    from ingestion.doc_parser import parse_doc_file
    chunks = parse_doc_file("path/to/requirements.pdf")
"""

import os
import re
from typing import List, Dict, Any

# Max tokens (approx words) per chunk before splitting
MAX_CHUNK_WORDS = 400
MIN_CHUNK_WORDS = 20   # Discard chunks that are too short (headings, noise)


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def parse_doc_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a PDF or DOCX requirements document into paragraph-level chunks.

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        List of chunk dicts conforming to SCHEMA.md.

    Raises:
        ValueError: If the file extension is not .pdf or .docx.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    if ext == ".pdf":
        return _parse_pdf(file_path, file_name)
    elif ext == ".docx":
        return _parse_docx(file_path, file_name)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .docx are supported.")


# ─────────────────────────────────────────────
# PDF parser
# ─────────────────────────────────────────────

def _parse_pdf(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    """Parse PDF using pypdf, chunking by paragraph."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required. Run: pip install pypdf")

    reader = PdfReader(file_path)
    chunks = []
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        paragraphs = _split_into_paragraphs(raw_text)

        current_section = _detect_section(paragraphs[0]) if paragraphs else "Unknown"

        for para in paragraphs:
            # Update section heading if this paragraph looks like a heading
            detected = _detect_section(para)
            if detected != "Unknown":
                current_section = detected

            # Skip if too short (likely a heading or noise)
            if len(para.split()) < MIN_CHUNK_WORDS:
                continue

            # Split long paragraphs into sub-chunks
            sub_chunks = _split_long_text(para, MAX_CHUNK_WORDS)

            for sub in sub_chunks:
                chunks.append({
                    "content": sub.strip(),
                    "metadata": {
                        "type": "requirements",
                        "section": current_section,
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "file_name": file_name,
                        "source_format": "pdf",
                    }
                })
                chunk_index += 1

    return chunks


# ─────────────────────────────────────────────
# DOCX parser
# ─────────────────────────────────────────────

def _parse_docx(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    """Parse DOCX using python-docx, chunking by paragraph with heading awareness."""
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx is required. Run: pip install python-docx")

    document = docx.Document(file_path)
    chunks = []
    chunk_index = 0
    current_section = "Unknown"
    current_page = 1  # DOCX doesn't have real page info; use best-effort estimate
    paragraph_buffer = []

    for para in document.paragraphs:
        text = para.text.strip()

        if not text:
            continue

        # Detect headings — flush buffer and update section
        if para.style.name.startswith("Heading"):
            # Flush accumulated paragraphs as one chunk
            if paragraph_buffer:
                chunk = _flush_buffer(paragraph_buffer, current_section, current_page, chunk_index, file_name)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                paragraph_buffer = []

            current_section = text
            # Rough page estimate: every ~40 paragraphs ≈ 1 page
            current_page = max(1, chunk_index // 10 + 1)
            continue

        paragraph_buffer.append(text)

        # Flush buffer if it exceeds max chunk size
        combined = " ".join(paragraph_buffer)
        if len(combined.split()) >= MAX_CHUNK_WORDS:
            sub_chunks = _split_long_text(combined, MAX_CHUNK_WORDS)
            for sub in sub_chunks:
                if len(sub.split()) >= MIN_CHUNK_WORDS:
                    chunks.append({
                        "content": sub.strip(),
                        "metadata": {
                            "type": "requirements",
                            "section": current_section,
                            "page": current_page,
                            "chunk_index": chunk_index,
                            "file_name": file_name,
                            "source_format": "docx",
                        }
                    })
                    chunk_index += 1
            paragraph_buffer = []

    # Flush any remaining paragraphs
    if paragraph_buffer:
        chunk = _flush_buffer(paragraph_buffer, current_section, current_page, chunk_index, file_name)
        if chunk:
            chunks.append(chunk)

    return chunks


def _flush_buffer(
    buffer: List[str],
    section: str,
    page: int,
    chunk_index: int,
    file_name: str
) -> Dict[str, Any] | None:
    """Combine buffered paragraphs into a single chunk dict."""
    combined = " ".join(buffer).strip()
    if len(combined.split()) < MIN_CHUNK_WORDS:
        return None
    return {
        "content": combined,
        "metadata": {
            "type": "requirements",
            "section": section,
            "page": page,
            "chunk_index": chunk_index,
            "file_name": file_name,
            "source_format": "docx",
        }
    }


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _split_into_paragraphs(text: str) -> List[str]:
    """Split raw text into paragraphs by double newlines or sentence clusters."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on blank lines
    raw_paras = re.split(r"\n{2,}", text)
    # Further clean individual paragraphs
    paras = []
    for para in raw_paras:
        cleaned = re.sub(r"\s+", " ", para).strip()
        if cleaned:
            paras.append(cleaned)
    return paras


def _detect_section(text: str) -> str:
    """
    Heuristically detect if a paragraph is a section heading.
    Returns the heading text or 'Unknown'.
    """
    text = text.strip()
    # Common heading patterns: all caps, numbered (1.2.3), short lines
    if re.match(r"^(\d+\.)+\s+\w", text) and len(text) < 100:
        return text
    if text.isupper() and len(text) < 80:
        return text
    if re.match(r"^(Chapter|Section|Part|Appendix)\s+", text, re.IGNORECASE):
        return text
    return "Unknown"


def _split_long_text(text: str, max_words: int) -> List[str]:
    """
    Split a long paragraph into smaller chunks of at most max_words words.
    Tries to split at sentence boundaries.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    # Split into sentences first
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_count = 0

    for sentence in sentences:
        s_words = sentence.split()
        if current_count + len(s_words) > max_words and current:
            chunks.append(" ".join(current))
            current = s_words
            current_count = len(s_words)
        else:
            current.extend(s_words)
            current_count += len(s_words)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ─────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python doc_parser.py <file_path>")
        sys.exit(1)

    path = sys.argv[1]
    result = parse_doc_file(path)
    print(f"✅ Parsed {len(result)} chunk(s) from {path}\n")
    for i, chunk in enumerate(result[:5]):  # preview first 5
        print(f"--- Chunk {i + 1} ---")
        print(f"Section : {chunk['metadata']['section']}")
        print(f"Page    : {chunk['metadata']['page']}")
        print(f"Content : {chunk['content'][:200]}...")
        print()