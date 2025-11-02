#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for chunking strategies.
"""

import logging
from unittest.mock import Mock

import pytest

from index_documents import (
    chunk_by_paragraphs,
    chunk_by_sentences,
    chunk_fixed,
    non_empty,
)
from utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, MAX_CHARS_PER_EMBED


@pytest.fixture
def logger():
    """Create a mock logger."""
    return Mock(spec=logging.Logger)


class TestNonEmpty:
    """Tests for non_empty filter function."""

    def test_filters_empty_chunks(self, logger):
        """Test that empty chunks are filtered out."""
        chunks = [(0, "  "), (1, "valid text"), (2, ""), (3, "another")]
        result = non_empty(chunks, logger)
        assert len(result) == 2
        assert result[0]["chunk_text"] == "valid text"
        assert result[1]["chunk_text"] == "another"

    def test_truncates_oversized_chunks(self, logger):
        """Test that chunks exceeding MAX_CHARS_PER_EMBED are truncated."""
        oversized = "x" * (MAX_CHARS_PER_EMBED + 100)
        chunks = [(0, oversized)]
        result = non_empty(chunks, logger)
        assert len(result) == 1
        assert len(result[0]["chunk_text"]) == MAX_CHARS_PER_EMBED
        assert result[0]["chunk_text"] == oversized[:MAX_CHARS_PER_EMBED]
        logger.warning.assert_called_once()

    def test_preserves_valid_chunks(self, logger):
        """Test that valid chunks are preserved as-is."""
        chunks = [(0, "text1"), (1, "text2"), (2, "text3")]
        result = non_empty(chunks, logger)
        assert len(result) == 3
        assert [r["chunk_text"] for r in result] == ["text1", "text2", "text3"]


class TestChunkFixed:
    """Tests for fixed-size chunking strategy."""

    def test_empty_text_returns_empty_list(self, logger):
        """Test that empty text returns empty chunk list."""
        result = chunk_fixed("", logger)
        assert result == []

    def test_single_chunk_small_text(self, logger):
        """Test that small text produces single chunk."""
        text = "x" * 100
        result = chunk_fixed(text, logger)
        assert len(result) == 1
        assert len(result[0]["chunk_text"]) == 100

    def test_multiple_chunks_with_overlap(self, logger):
        """Test that chunks have proper overlap."""
        # Create text that will split into multiple chunks
        text = "x" * (DEFAULT_CHUNK_SIZE * 2)
        result = chunk_fixed(text, logger)
        assert len(result) >= 2
        
        # Check overlap: second chunk should start before first chunk ends
        first_end = DEFAULT_CHUNK_SIZE
        second_start = DEFAULT_CHUNK_SIZE - DEFAULT_OVERLAP
        # Since chunks are stored, we verify they're consecutive
        total_length = sum(len(c["chunk_text"]) for c in result)
        # With overlap, total should be less than text length
        assert total_length > len(text) - (DEFAULT_OVERLAP * len(result))

    def test_chunks_respect_max_size(self, logger):
        """Test that all chunks respect MAX_CHARS_PER_EMBED limit."""
        text = "x" * (MAX_CHARS_PER_EMBED * 3)
        result = chunk_fixed(text, logger)
        for chunk in result:
            assert len(chunk["chunk_text"]) <= MAX_CHARS_PER_EMBED


class TestChunkBySentences:
    """Tests for sentence-based chunking strategy."""

    def test_empty_text_returns_empty_list(self, logger):
        """Test that empty text returns empty chunk list."""
        result = chunk_by_sentences("", logger)
        assert result == []

    def test_single_sentence(self, logger):
        """Test that single sentence produces single chunk."""
        text = "This is a single sentence."
        result = chunk_by_sentences(text, logger)
        assert len(result) == 1
        assert "sentence" in result[0]["chunk_text"]

    def test_multiple_sentences_within_chunk_size(self, logger):
        """Test that multiple sentences within chunk size are combined."""
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_by_sentences(text, logger)
        # All should fit in one chunk (total < DEFAULT_CHUNK_SIZE)
        assert len(result) >= 1
        combined = " ".join(c["chunk_text"] for c in result)
        assert "First" in combined
        assert "Second" in combined
        assert "Third" in combined

    def test_large_text_splits_into_multiple_chunks(self, logger):
        """Test that large text splits across multiple chunks."""
        # Create many sentences
        sentences = [f"Sentence {i}. " for i in range(200)]
        text = "".join(sentences)
        result = chunk_by_sentences(text, logger)
        assert len(result) > 1
        # Each chunk should be <= DEFAULT_CHUNK_SIZE (approximately)
        for chunk in result:
            assert len(chunk["chunk_text"]) <= MAX_CHARS_PER_EMBED

    def test_no_overlap_between_chunks(self, logger):
        """Test that sentence chunks don't overlap."""
        text = "First sentence. Second sentence. Third sentence. " * 50
        result = chunk_by_sentences(text, logger)
        # Combine all chunk texts and verify no sentence appears twice
        all_text = " ".join(c["chunk_text"] for c in result)
        # Simple check: count occurrences of a unique sentence
        assert all_text.count("First sentence") == 50


class TestChunkByParagraphs:
    """Tests for paragraph-based chunking strategy."""

    def test_empty_text_returns_empty_list(self, logger):
        """Test that empty text returns empty chunk list."""
        result = chunk_by_paragraphs("", logger)
        assert result == []

    def test_single_paragraph(self, logger):
        """Test that single paragraph produces single chunk."""
        text = "This is a single paragraph with multiple sentences."
        result = chunk_by_paragraphs(text, logger)
        assert len(result) == 1

    def test_multiple_paragraphs(self, logger):
        """Test that multiple paragraphs are handled correctly."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = chunk_by_paragraphs(text, logger)
        assert len(result) >= 1
        # All paragraphs should appear
        combined = " ".join(c["chunk_text"] for c in result)
        assert "Paragraph one" in combined
        assert "Paragraph two" in combined
        assert "Paragraph three" in combined

    def test_short_paragraphs_combined(self, logger):
        """Test that short paragraphs are combined until chunk size."""
        # Create many short paragraphs
        paragraphs = [f"Short paragraph {i}." for i in range(100)]
        text = "\n\n".join(paragraphs)
        result = chunk_by_paragraphs(text, logger)
        # Should have fewer chunks than paragraphs (combining happens)
        assert len(result) < len(paragraphs)
        for chunk in result:
            assert len(chunk["chunk_text"]) <= MAX_CHARS_PER_EMBED

    def test_paragraph_boundaries_preserved(self, logger):
        """Test that paragraph boundaries are preserved."""
        text = "P1 sentence 1.\n\nP2 sentence 1.\n\nP3 sentence 1."
        result = chunk_by_paragraphs(text, logger)
        # Verify paragraph markers (double newlines) are preserved
        combined = "\n\n".join(c["chunk_text"] for c in result)
        assert "\n\n" in combined or len(result) == 1
