#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for all tests.
"""

import logging
from unittest.mock import Mock

import pytest
import nltk


@pytest.fixture(scope="session", autouse=True)
def setup_nltk():
    """Ensure NLTK punkt tokenizer is downloaded before tests."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


@pytest.fixture
def logger():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def sample_text():
    """Sample text for chunking tests."""
    return """This is the first paragraph. It contains multiple sentences. Each sentence adds to the meaning.

This is the second paragraph. It also has multiple sentences. Testing chunking strategies.

Short paragraph.

This is a longer paragraph that contains more text. It should be split appropriately based on the chunking strategy used. The text continues here to make it longer."""
