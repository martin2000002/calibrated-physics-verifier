"""Exceptions raised by schema records (container-safe: stdlib only)."""

from __future__ import annotations


class SchemaError(ValueError):
    """A record violates its schema: shape, dtype, invariant or file format."""


class SchemaVersionError(SchemaError):
    """A file was written with a different SCHEMA_VERSION."""


class ScaleBindingError(ValueError):
    """A scale estimate is applied to a table it was not fitted on (lift_key != source_key)."""
