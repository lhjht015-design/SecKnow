from __future__ import annotations


class InferenceError(Exception):
    """Base exception for 4.4 inference module."""


class QueryEncodingError(InferenceError):
    """Raised when a query cannot be encoded."""


class UnsupportedLanguageError(InferenceError):
    """Raised when a code scan language is not supported."""
