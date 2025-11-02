#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error handling and friendly error message formatting.
"""


def format_friendly_error(error: Exception) -> tuple[str, str, str]:
    """Extract cause, details, and action from an error.
    
    Args:
        error: Exception that occurred
        
    Returns:
        Tuple of (cause, details, action)
    """
    error_str = str(error)
    
    # Quota errors
    if "429" in error_str or "quota" in error_str.lower() or "Quota exceeded" in error_str:
        cause = "Gemini API quota exhausted (HTTP 429)."
        details = error_str[:200] if len(error_str) > 200 else error_str
        action = "Try again later"
        return (cause, details, action)
    
    # Database connection errors
    if "connection" in error_str.lower() or "could not connect" in error_str.lower():
        cause = "Database connection failed."
        details = error_str[:200] if len(error_str) > 200 else error_str
        action = "Check POSTGRES_URL in .env and ensure the server is running"
        return (cause, details, action)
    
    # Missing env vars
    if "Missing required environment variables" in error_str:
        cause = "Missing required environment variables."
        details = error_str
        action = "Create .env file from .env.example and fill in required values"
        return (cause, details, action)
    
    # File not found
    if "not found" in error_str.lower() or "FileNotFoundError" in str(type(error)):
        cause = "File not found."
        details = error_str[:200] if len(error_str) > 200 else error_str
        action = "Verify the file path is correct"
        return (cause, details, action)
    
    # Generic error
    cause = error_str.split("\n")[0][:100] if "\n" in error_str else error_str[:100]
    details = error_str[:300] if len(error_str) > 100 else ""
    action = ""
    return (cause, details, action)

