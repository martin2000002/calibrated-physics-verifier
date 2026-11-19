"""The error of every freeze query."""

from __future__ import annotations


class QueryError(Exception):
    """A check failed or an input is unusable (exit 2)."""
