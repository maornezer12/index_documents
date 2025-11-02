#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for embedding generation (Gemini API).
"""

import logging
from unittest.mock import Mock, patch, MagicMock

import pytest

from utils import embeddings_utils
from utils.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_PRIMARY,
    EMBEDDING_MODEL_FALLBACK,
    MAX_CHARS_PER_EMBED,
)


@pytest.fixture
def logger():
    """Create a mock logger."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def mock_embedding_response():
    """Create a mock embedding response."""
    return {
        "embedding": {
            "values": [0.1] * EMBEDDING_DIM
        }
    }


class TestEmbedSingle:
    """Tests for embed_single function."""

    @patch('utils.embeddings_utils.genai.embed_content')
    def test_returns_embedding_vector(self, mock_embed_content, mock_embedding_response):
        """Test that embed_single returns a list of floats."""
        mock_embed_content.return_value = mock_embedding_response
        
        result = embeddings_utils.embed_single("test text", EMBEDDING_MODEL_PRIMARY)
        
        assert isinstance(result, list)
        assert len(result) == EMBEDDING_DIM
        assert all(isinstance(x, (int, float)) for x in result)
        mock_embed_content.assert_called_once_with(
            model=EMBEDDING_MODEL_PRIMARY,
            content="test text"
        )

    @patch('utils.embeddings_utils.genai.embed_content')
    def test_handles_list_response_format(self, mock_embed_content):
        """Test that embed_single handles list format in response."""
        mock_embed_content.return_value = {
            "embedding": [0.1] * EMBEDDING_DIM
        }
        
        result = embeddings_utils.embed_single("test", EMBEDDING_MODEL_PRIMARY)
        
        assert len(result) == EMBEDDING_DIM

    @patch('utils.embeddings_utils.genai.embed_content')
    def test_raises_on_unexpected_format(self, mock_embed_content):
        """Test that embed_single raises error on unexpected response format."""
        mock_embed_content.return_value = {"unexpected": "format"}
        
        with pytest.raises(RuntimeError, match="Unexpected embedding response"):
            embeddings_utils.embed_single("test", EMBEDDING_MODEL_PRIMARY)


class TestEmbedTexts:
    """Tests for embed_texts function."""

    @patch('utils.embeddings_utils.genai.embed_content')
    def test_embeds_empty_list(self, mock_embed_content, logger):
        """Test that empty list returns empty list."""
        result = embeddings_utils.embed_texts([], logger)
        assert result == []
        mock_embed_content.assert_not_called()

    @patch('utils.embeddings_utils.genai.embed_content')
    def test_embeds_single_text(self, mock_embed_content, logger, mock_embedding_response):
        """Test embedding a single text."""
        mock_embed_content.return_value = mock_embedding_response
        
        result = embeddings_utils.embed_texts(["test text"], logger)
        
        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM
        assert mock_embed_content.call_count == 1

    @patch('utils.embeddings_utils.genai.embed_content')
    def test_embeds_multiple_texts(self, mock_embed_content, logger, mock_embedding_response):
        """Test embedding multiple texts."""
        mock_embed_content.return_value = mock_embedding_response
        
        texts = ["text1", "text2", "text3"]
        result = embeddings_utils.embed_texts(texts, logger)
        
        assert len(result) == 3
        assert all(len(vec) == EMBEDDING_DIM for vec in result)
        assert mock_embed_content.call_count == 3

    @patch('utils.embeddings_utils.genai.embed_content')
    def test_raises_on_oversized_chunk(self, mock_embed_content, logger):
        """Test that oversized chunks raise RuntimeError."""
        oversized_text = "x" * (MAX_CHARS_PER_EMBED + 1)
        
        with pytest.raises(RuntimeError, match="exceeds MAX_CHARS_PER_EMBED"):
            embeddings_utils.embed_texts([oversized_text], logger)
        
        mock_embed_content.assert_not_called()

    @patch('utils.embeddings_utils.genai.embed_content')
    @patch('time.sleep')
    def test_retries_on_error(self, mock_sleep, mock_embed_content, logger, mock_embedding_response):
        """Test that embed_texts retries on error."""
        # Primary fails, fallback fails, then both succeed on retry
        call_count = [0]
        def side_effect(model, content):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("Network error")
            return mock_embedding_response
        
        mock_embed_content.side_effect = side_effect
        
        result = embeddings_utils.embed_texts(["test"], logger)
        
        assert len(result) == 1
        # Should try primary + fallback, then retry (primary succeeds)
        assert mock_embed_content.call_count >= 2
        assert mock_sleep.call_count >= 1  # Sleeps between retries

    @patch('utils.embeddings_utils.genai.embed_content')
    @patch('time.sleep')
    def test_uses_fallback_model_on_error(self, mock_sleep, mock_embed_content, logger, mock_embedding_response):
        """Test that embed_texts uses fallback model if primary fails."""
        # Primary fails, fallback succeeds
        def side_effect(model, content):
            if model == EMBEDDING_MODEL_PRIMARY:
                raise Exception("Primary model error")
            return mock_embedding_response
        
        mock_embed_content.side_effect = side_effect
        
        result = embeddings_utils.embed_texts(["test"], logger)
        
        assert len(result) == 1
        # Should call primary first, then fallback
        assert mock_embed_content.call_count == 2
        # Check that both models were called
        call_args_list = mock_embed_content.call_args_list
        models_called = [call[1]['model'] for call in call_args_list]
        assert EMBEDDING_MODEL_PRIMARY in models_called
        assert EMBEDDING_MODEL_FALLBACK in models_called

    @patch('utils.embeddings_utils.genai.embed_content')
    @patch('time.sleep')
    def test_handles_quota_error(self, mock_sleep, mock_embed_content, logger):
        """Test that quota errors are detected and logged."""
        mock_embed_content.side_effect = Exception("HTTP 429: Quota exceeded")
        
        with pytest.raises(RuntimeError, match="quota"):
            embeddings_utils.embed_texts(["test"], logger)
        
        logger.warning.assert_called()

    @patch('utils.embeddings_utils.genai.embed_content')
    @patch('time.sleep')
    def test_fails_after_max_retries(self, mock_sleep, mock_embed_content, logger):
        """Test that function raises after max retries."""
        mock_embed_content.side_effect = Exception("Persistent error")
        
        with pytest.raises(RuntimeError, match="failed after retries"):
            embeddings_utils.embed_texts(["test"], logger)
        
        # Should try 5 times
        assert mock_embed_content.call_count == 10  # 5 primary + 5 fallback attempts

    @patch('utils.embeddings_utils.genai.embed_content')
    def test_validates_consistent_dimensions(self, mock_embed_content, logger):
        """Test that all embeddings have consistent dimensions."""
        # First embedding has correct dim, second has wrong dim
        def side_effect(model, content):
            if content == "text1":
                return {"embedding": {"values": [0.1] * EMBEDDING_DIM}}
            else:
                return {"embedding": {"values": [0.1] * (EMBEDDING_DIM + 10)}}
        
        mock_embed_content.side_effect = side_effect
        
        with pytest.raises(RuntimeError, match="Inconsistent embedding dimensions"):
            embeddings_utils.embed_texts(["text1", "text2"], logger)

    @patch('utils.embeddings_utils.genai.embed_content')
    def test_preserves_ordering(self, mock_embed_content, logger):
        """Test that embeddings preserve input text ordering."""
        responses = [
            {"embedding": {"values": [0.1] * EMBEDDING_DIM}},
            {"embedding": {"values": [0.2] * EMBEDDING_DIM}},
            {"embedding": {"values": [0.3] * EMBEDDING_DIM}},
        ]
        mock_embed_content.side_effect = responses
        
        texts = ["text1", "text2", "text3"]
        result = embeddings_utils.embed_texts(texts, logger)
        
        assert len(result) == 3
        # Verify ordering (check first value of each embedding)
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2
        assert result[2][0] == 0.3
