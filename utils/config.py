#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants and configuration for the document indexer.
All constants are centralized here.
"""

# =============================
# Chunking Configuration
# =============================
DEFAULT_STRATEGY = "sentence"
# Balanced and safe defaults: avoids hitting model limits
DEFAULT_CHUNK_SIZE = 1500      # Average chunk length (characters)
DEFAULT_OVERLAP = 200          # Context overlap between chunks

# =============================
# File Support
# =============================
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

# =============================
# Embedding Configuration
# =============================
EMBEDDING_MODEL_PRIMARY = "models/text-embedding-004"
EMBEDDING_MODEL_FALLBACK = "models/embedding-001"
EMBEDDING_DIM = 768
# Hard safety limit: ensures no chunk exceeds model limits
MAX_CHARS_PER_EMBED = 3000

# =============================
# Database Configuration
# =============================
TABLE_NAME = "document_chunks"

