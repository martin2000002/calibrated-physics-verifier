"""Shape, dtype and value checks shared by every schema record."""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from physverify.schema.errors import SchemaError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


class ArraySpec(NamedTuple):
    """Expected dtype and shape of an array field.

    ``dtype`` is a numpy dtype string, or ``"U"`` for a unicode array of any width. Shape entries
    are exact sizes (int) or dimension names (str) that must agree across the fields of a record.
    """

    dtype: str
    shape: tuple[int | str, ...]
    optional: bool = False


def check_arrays(
    owner: str, values: Mapping[str, Any], specs: Mapping[str, ArraySpec]
) -> dict[str, int]:
    """Validate every array field of a record and return the bound dimension sizes."""
    dims: dict[str, int] = {}
    for name, spec in specs.items():
        value = values[name]
        if value is None and spec.optional:
            continue
        _check_array(f"{owner}.{name}", value, spec, dims)
    return dims


def _check_array(label: str, value: object, spec: ArraySpec, dims: dict[str, int]) -> None:
    if not isinstance(value, np.ndarray):
        raise SchemaError(f"{label}: expected a numpy array, got {type(value).__name__}")
    ok = value.dtype.kind == "U" if spec.dtype == "U" else value.dtype == np.dtype(spec.dtype)
    if not ok:
        raise SchemaError(f"{label}: expected dtype {spec.dtype}, got {value.dtype.str}")
    if value.ndim != len(spec.shape):
        raise SchemaError(f"{label}: expected {len(spec.shape)} dims, got shape {value.shape}")
    for axis, expected in enumerate(spec.shape):
        _check_dim(label, axis, value.shape[axis], expected, dims)


def _check_dim(label: str, axis: int, size: int, expected: int | str, dims: dict[str, int]) -> None:
    if isinstance(expected, int):
        if size != expected:
            raise SchemaError(f"{label}: axis {axis} must be {expected}, got {size}")
        return
    bound = dims.setdefault(expected, size)
    if bound != size:
        raise SchemaError(f"{label}: axis {axis} is {size} but {expected}={bound} elsewhere")


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_number(label: str, value: object) -> float:
    if not is_number(value):
        raise SchemaError(f"{label}: expected a number, got {type(value).__name__}")
    return float(value)  # type: ignore[arg-type]


def check_finite(label: str, value: object) -> None:
    if not math.isfinite(check_number(label, value)):
        raise SchemaError(f"{label}: must be finite, got {value}")


def check_optional_finite(label: str, value: object) -> None:
    if value is not None:
        check_finite(label, value)


def check_positive(label: str, value: object) -> None:
    number = check_number(label, value)
    if not (math.isfinite(number) and number > 0):
        raise SchemaError(f"{label}: must be finite and > 0, got {value}")


def check_sigma(label: str, value: object) -> None:
    """A standard deviation is NaN (unknown) or finite and >= 0; never negative or infinite."""
    number = check_number(label, value)
    if not (math.isnan(number) or (math.isfinite(number) and number >= 0)):
        raise SchemaError(f"{label}: sigma must be NaN or finite >= 0, got {value}")


def check_sigma_array(label: str, values: NDArray[np.float64]) -> None:
    if bool(np.any(values < 0)) or bool(np.any(np.isinf(values))):
        raise SchemaError(f"{label}: sigma entries must be NaN or finite >= 0")


def check_int(label: str, value: object, minimum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SchemaError(f"{label}: expected an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise SchemaError(f"{label}: must be >= {minimum}, got {value}")


def check_bool(label: str, value: object) -> None:
    if not isinstance(value, bool):
        raise SchemaError(f"{label}: expected a bool, got {type(value).__name__}")


def check_str(label: str, value: object, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise SchemaError(f"{label}: expected a str, got {type(value).__name__}")
    if not allow_empty and not value:
        raise SchemaError(f"{label}: must not be empty")


def check_optional_str(label: str, value: object) -> None:
    if value is not None:
        check_str(label, value)


def check_str_tuple(label: str, value: object) -> None:
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        raise SchemaError(f"{label}: expected a tuple of str")


def check_float_tuple(label: str, value: object, length: int | None = None) -> None:
    if not isinstance(value, tuple) or not all(is_number(item) for item in value):
        raise SchemaError(f"{label}: expected a tuple of numbers")
    if length is not None and len(value) != length:
        raise SchemaError(f"{label}: expected {length} numbers, got {len(value)}")


def check_enum(label: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise SchemaError(f"{label}: expected {enum_type.__name__}, got {value!r}")


def check_instance(label: str, value: object, kind: type) -> None:
    if not isinstance(value, kind):
        raise SchemaError(f"{label}: expected {kind.__name__}, got {type(value).__name__}")


def check_unit_vector(label: str, value: NDArray[np.float64], allow_nan: bool = True) -> None:
    """A direction is all-NaN (unknown, when allowed) or finite with unit norm (1e-6)."""
    if allow_nan and bool(np.all(np.isnan(value))):
        return
    if not bool(np.all(np.isfinite(value))):
        raise SchemaError(f"{label}: must be finite or entirely NaN")
    if abs(float(np.linalg.norm(value)) - 1.0) > 1e-6:
        raise SchemaError(f"{label}: must have unit norm")


def check_json_mapping(label: str, value: object) -> None:
    """Free-form mappings hold only JSON-native values so they round-trip exactly."""
    if not isinstance(value, dict) or not _is_json_native(value):
        raise SchemaError(f"{label}: expected a dict of JSON-native values (no tuples or arrays)")


def _is_json_native(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, list):
        return all(_is_json_native(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_native(v) for k, v in value.items())
    return False


def check_span(label: str, span: object) -> None:
    """A frame span is an inclusive (first, last) pair of native frame indices."""
    if not isinstance(span, tuple) or len(span) != 2:
        raise SchemaError(f"{label}: expected a (first, last) tuple")
    for value in span:
        check_int(label, value, minimum=0)
    if span[0] > span[1]:
        raise SchemaError(f"{label}: first must be <= last")
