#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface and argument parsing.
"""

import argparse

from utils.config import DEFAULT_STRATEGY


def validate_strategy(strategy: str) -> str:
    """Validate chunking strategy."""
    allowed = {"fixed", "sentence", "paragraph"}
    if strategy not in allowed:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {sorted(allowed)}")
    return strategy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    p = argparse.ArgumentParser(description="Index a document into PostgreSQL with Gemini embeddings.")
    p.add_argument("--file", required=True, help="Path to PDF or DOCX file")
    p.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=["fixed", "sentence", "paragraph"], help="Chunking strategy")
    return p.parse_args()

