#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding generation - Gemini API.
"""

import logging
import time

import google.generativeai as genai

from utils.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_FALLBACK,
    EMBEDDING_MODEL_PRIMARY,
    MAX_CHARS_PER_EMBED,
)


# =============================
# Gemini embeddings
# =============================

def embed_single(content: str, model: str) -> list[float]:
    """Embed a single text using Gemini API.
    
    Args:
        content: Text to embed
        model: Model name to use
        
    Returns:
        List of floats representing embedding vector
    """
    # genai.embed_content returns a dict with { 'embedding': { 'values': [...] } } in recent versions
    resp = genai.embed_content(model=model, content=content)
    # Handle both possible shapes
    if isinstance(resp, dict):
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return list(emb["values"])  # type: ignore
        # Some SDK versions return list directly as 'embedding'
        if isinstance(emb, list):
            return [float(x) for x in emb]
    # Fallback attempt
    if hasattr(resp, "embedding"):
        emb_obj = getattr(resp, "embedding")
        if isinstance(emb_obj, dict) and "values" in emb_obj:
            return list(emb_obj["values"])  # type: ignore
    raise RuntimeError("Unexpected embedding response format from Gemini API")


def embed_texts(texts: list[str], logger: logging.Logger) -> list[list[float]]:
    """Embed a list of texts using Google Gemini embeddings with retries.

    - Exponential backoff with jitter for transient errors.
    - Inputs are expected to be pre-trimmed to MAX_CHARS_PER_EMBED by the chunking functions.
    - Preserves ordering.
    - Handles quota errors gracefully with clear messages.

    Args:
        texts: List of text strings to embed (must be <= MAX_CHARS_PER_EMBED each)
        logger: Logger instance

    Returns:
        List of embedding vectors (list of floats)

    Raises:
        RuntimeError: if embedding fails after retries, or if a chunk exceeds MAX_CHARS_PER_EMBED
    """
    if not texts:
        return []

    model = EMBEDDING_MODEL_PRIMARY
    results: list[list[float]] = []
    
    for text_idx, text in enumerate(texts):
        # Defensive check: chunkers must enforce size; don't silently slice here.
        if len(text) > MAX_CHARS_PER_EMBED:
            raise RuntimeError(
                f"Chunk exceeds MAX_CHARS_PER_EMBED ({len(text)} > {MAX_CHARS_PER_EMBED}). "
                "Chunking functions must enforce size limits (see non_empty)."
            )
        content = text
        attempt = 0
        last_err: Exception | None = None
        quota_error = False
        
        while attempt < 5:
            try:
                try:
                    vec = embed_single(content, model)
                except Exception:
                    # Fallback to older model if primary fails
                    vec = embed_single(content, EMBEDDING_MODEL_FALLBACK)
                results.append([float(x) for x in vec])
                break
            except Exception as e:
                last_err = e
                error_str = str(e)
                
                # Check for quota errors
                if "429" in error_str or "quota" in error_str.lower() or "Quota exceeded" in error_str:
                    quota_error = True
                    if attempt == 0:  # Only log once per text
                        logger.warning("Gemini API quota exceeded for text %d/%d", text_idx + 1, len(texts))
                
                sleep_s = (2 ** attempt) + (0.1 * (1 + attempt))
                if attempt < 4:  # Don't log on last attempt
                    logger.debug("Embedding failed (attempt %d): %s; retrying in %.1fs", attempt + 1, error_str[:100], sleep_s)
                time.sleep(sleep_s)
                attempt += 1
        else:
            # After retries, raise with helpful message
            if last_err is None:
                raise RuntimeError("Embedding failed after retries: unknown error")
            if quota_error:
                raise RuntimeError(
                    f"Embedding failed after retries (quota exhausted). "
                    f"Try again later. Error: {last_err}"
                )
            raise RuntimeError(f"Embedding failed after retries: {last_err}")

    # Validate consistent lengths
    if results:
        dim = len(results[0])
        for vec in results:
            if len(vec) != dim:
                raise RuntimeError("Inconsistent embedding dimensions across chunks")
    return results

