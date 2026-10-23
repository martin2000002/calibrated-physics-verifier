"""The error of every freeze query: a check failed or an input is unusable (exit 2)."""

from __future__ import annotations


class QueryError(Exception):
    """A check failed or an input is unusable (exit 2)."""
