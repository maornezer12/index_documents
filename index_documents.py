#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_documents.py - Document indexing main script

Extracts text from PDF/DOCX, chunks it, generates embeddings, and stores in PostgreSQL.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

# Third-party dependencies
from dotenv import load_dotenv as dotenv_load
import nltk
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
import psycopg2

# Local utilities
from utils import io_utils
from utils import db_utils
from utils import cli_utils
from utils import embeddings_utils
from utils import errors_utils
from utils.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    MAX_CHARS_PER_EMBED,
    SUPPORTED_EXTENSIONS,
    TABLE_NAME,
)


def load_env():
    """Load environment variables from .env and validate required keys."""
    dotenv_load()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    postgres_url = os.getenv("POSTGRES_URL", "").strip()

    missing = []
    if not gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if not postgres_url:
        missing.append("POSTGRES_URL")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return {"gemini_api_key": gemini_api_key, "postgres_url": postgres_url}


def non_empty(chunks: Iterable[Tuple[int, str]], logger: logging.Logger) -> list[dict]:
    """Filter out empty chunks and enforce MAX_CHARS_PER_EMBED limit for all chunking strategies.
    
    This is the single enforcement point for chunk size limits. Any chunk exceeding
    MAX_CHARS_PER_EMBED will be truncated with a warning logged.
    
    Args:
        chunks: Iterable of (index, content) tuples from chunking functions
        logger: Logger instance for warnings
        
    Returns:
        List of dicts with "chunk_text" keys, all <= MAX_CHARS_PER_EMBED
    """
    results = []
    for index, content in chunks:
        trimmed = content.strip()
        if not trimmed:
            continue
        
        if len(trimmed) > MAX_CHARS_PER_EMBED:
            logger.warning(
                "Chunk exceeds %d characters, truncating from %d to %d",
                MAX_CHARS_PER_EMBED, len(trimmed), MAX_CHARS_PER_EMBED
            )
            trimmed = trimmed[:MAX_CHARS_PER_EMBED]
        
        results.append({"chunk_text": trimmed})
    return results


def ensure_nltk_punkt(logger: logging.Logger) -> None:
    """Ensure NLTK Punkt tokenizer is available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)


def chunk_fixed(text: str, logger: logging.Logger) -> list[dict]:
    """Fixed-size character window chunking with overlap."""
    chunk_size = DEFAULT_CHUNK_SIZE
    overlap = DEFAULT_OVERLAP
    
    chunks = []
    start = 0
    index = 0
    n = len(text)
    if n == 0:
        return []
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append((index, chunk))
        index += 1
        if end == n:
            break
        start = end - overlap
    return non_empty(chunks, logger)


def chunk_by_sentences(text: str, logger: logging.Logger) -> list[dict]:
    """Sentence-based chunking using NLTK Punkt."""
    ensure_nltk_punkt(logger)
    chunk_size = DEFAULT_CHUNK_SIZE

    sentences = [s.strip() for s in sent_tokenize(text) if s and s.strip()]
    if not sentences:
        return []
    chunks = []
    buf = []
    buf_len = 0
    index = 0
    for s in sentences:
        s_len = len(s)
        if buf_len == 0:
            buf.append(s)
            buf_len = s_len
            continue
        if buf_len + 1 + s_len <= chunk_size:
            buf.append(s)
            buf_len += 1 + s_len
        else:
            chunks.append((index, " ".join(buf)))
            index += 1
            buf = [s]
            buf_len = s_len
    if buf:
        chunks.append((index, " ".join(buf)))
    return non_empty(chunks, logger)


def chunk_by_paragraphs(text: str, logger: logging.Logger) -> list[dict]:
    """Paragraph-based chunking by double newlines."""
    chunk_size = DEFAULT_CHUNK_SIZE
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    chunks = []
    buf = []
    buf_len = 0
    index = 0
    for p in paragraphs:
        p_len = len(p)
        if buf_len == 0:
            buf.append(p)
            buf_len = p_len
            continue
        if buf_len + 2 + p_len <= chunk_size:
            buf.append(p)
            buf_len += 2 + p_len
        else:
            chunks.append((index, "\n\n".join(buf)))
            index += 1
            buf = [p]
            buf_len = p_len
    if buf:
        chunks.append((index, "\n\n".join(buf)))
    return non_empty(chunks, logger)


def print_error(cause: str, details: str = "", action: str = "") -> None:
    """Print a formatted error message to stderr.
    
    Args:
        cause: Main error message
        details: Optional error details
        action: Optional suggested action
    """
    print(f"\n❌ {cause}", file=sys.stderr)
    if details:
        print(f"   {details}", file=sys.stderr)
    if action:
        print(f"   → {action}", file=sys.stderr)


# =============================
# Main function
# =============================

def main() -> None:
    """Main entry point with human-friendly output."""
    args = cli_utils.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("index_documents")
    
    try:
        # Load environment
        try:
            env_config = load_env()
        except Exception as e:
            cause, details, action = errors_utils.format_friendly_error(e)
            print_error(cause, details, action)
            sys.exit(1)

        # Configure Gemini
        genai.configure(api_key=env_config["gemini_api_key"])
        logger.debug("Gemini API configured")

        # Validate file
        file_path = Path(args.file).expanduser().resolve()
        if not file_path.exists() or not file_path.is_file():
            cause, details, action = errors_utils.format_friendly_error(
                FileNotFoundError(f"File not found: {file_path}")
            )
            print_error(cause, details, action)
            sys.exit(1)
        
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.error("Unsupported file extension: %s. Supported: %s", ext, sorted(SUPPORTED_EXTENSIONS))
            print_error(
                f"Unsupported file extension: {ext}",
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                "Use a PDF or DOCX file"
            )
            sys.exit(1)

        # Validate strategy
        try:
            strategy = cli_utils.validate_strategy(args.strategy)
        except ValueError as e:
            logger.error(str(e))
            print_error(str(e), "", "Use one of: fixed, sentence, paragraph")
            sys.exit(1)

        # Extract text
        try:
            if ext == ".pdf":
                raw_text = io_utils.extract_text_from_pdf(str(file_path))
            else:
                raw_text = io_utils.extract_text_from_docx(str(file_path))
        except Exception as e:
            logger.error("Failed to extract text: %s", str(e))
            cause, details, action = errors_utils.format_friendly_error(e)
            print_error(cause, details, action)
            sys.exit(1)

        # Normalize
        text = io_utils.normalize_text(raw_text)
        if not text:
            logger.error("No text extracted from document.")
            print_error("No text extracted from document", "", "Ensure the file contains readable text")
            sys.exit(1)

        # Chunk
        try:
            if strategy == "fixed":
                chunks = chunk_fixed(text, logger)
            elif strategy == "sentence":
                chunks = chunk_by_sentences(text, logger)
            else:
                chunks = chunk_by_paragraphs(text, logger)
        except Exception as e:
            logger.error("Chunking failed: %s", str(e))
            cause, details, action = errors_utils.format_friendly_error(e)
            print_error(cause, details, action)
            sys.exit(1)

        if not chunks:
            logger.error("No chunks produced from text.")
            print_error("No chunks produced from text", "", "Try a different chunking strategy")
            sys.exit(1)

        chunk_texts = [c["chunk_text"] for c in chunks]

        # Embed
        try:
            embeddings_list = embeddings_utils.embed_texts(chunk_texts, logger)
        except Exception as e:
            logger.error("Embedding failed: %s", str(e))
            cause, details, action = errors_utils.format_friendly_error(e)
            print_error(cause, details, action)
            sys.exit(1)

        if len(embeddings_list) != len(chunk_texts):
            logger.error("Embedding count mismatch: got %d, expected %d", len(embeddings_list), len(chunk_texts))
            print_error(
                "Embedding count mismatch",
                f"Got {len(embeddings_list)}, expected {len(chunk_texts)}",
                "This is an internal error. Please report it."
            )
            sys.exit(1)

        # Persist to DB
        try:
            conn = psycopg2.connect(env_config["postgres_url"])
            logger.debug("Connected to PostgreSQL database")
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL: %s", str(e))
            cause, details, action = errors_utils.format_friendly_error(e)
            print_error(cause, details, action)
            sys.exit(1)

        try:
            db_utils.ensure_table_exists(conn, logger, TABLE_NAME)
            db_utils.delete_existing_chunks(conn, file_path.name, strategy, logger, TABLE_NAME)
            
            rows = []
            for c, emb in zip(chunks, embeddings_list):
                rows.append((file_path.name, strategy, c["chunk_text"], emb))
            
            inserted = db_utils.insert_chunks(conn, rows, logger, TABLE_NAME)
            logger.debug("Successfully inserted %d chunks into database", inserted)
        except Exception as e:
            logger.error("Failed to insert rows: %s", str(e))
            cause, details, action = errors_utils.format_friendly_error(e)
            print_error(cause, details, action)
            conn.close()
            sys.exit(1)
        finally:
            try:
                conn.close()
                logger.debug("Database connection closed")
            except Exception:
                pass
        
        # Print success message
        print(f"File indexed successfully: {file_path.name}")
        
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error")
        cause, details, action = errors_utils.format_friendly_error(e)
        print_error(cause, details, action)
        sys.exit(1)


if __name__ == "__main__":
    main()
