#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File I/O utilities for reading PDF/DOCX files and text normalization.
"""

import re
from pypdf import PdfReader
from docx import Document


# =============================
# Text normalization
# =============================

_WHITESPACE_RE = re.compile(r"[\t\f\r ]+")
_MULTI_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize whitespace and ensure clean UTF-8-like text.

    - Collapses multiple spaces/tabs.
    - Collapses 3+ newlines to exactly two newlines (paragraph break).
    - Strips leading/trailing whitespace.

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text string
    """
    if not text:
        return ""
    # Replace common non-breaking spaces, etc.
    normalized = text.replace("\u00a0", " ")
    # Collapse spaces/tabs
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    # Normalize long newline sequences
    normalized = _MULTI_NEWLINES_RE.sub("\n\n", normalized)
    # Trim
    normalized = normalized.strip()
    return normalized


# =============================
# File reading
# =============================

def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF using pypdf.

    Args:
        path: Path to PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: if file doesn't exist
        Exception: if PDF reading fails
    """
    reader = PdfReader(path)
    texts: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            texts.append(page_text)
    return "\n\n".join(texts)


def extract_text_from_docx(path: str) -> str:
    """Extract text from a DOCX using python-docx.

    Args:
        path: Path to DOCX file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: if file doesn't exist
        Exception: if DOCX reading fails
    """
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(paras)

