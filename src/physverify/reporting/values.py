"""A reported value with its source, and the readers that take one from a ``Doc``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from physverify.reporting.sources import Doc
from physverify.stats.intervals import Rate

Scalar = float | int | str | bool | None
CI_LOW, CI_HIGH = "low", "high"
EXACT_CI = Rate.model_fields["ci_method"].default


@dataclass(frozen=True)
class Value:
    """``source`` names ``value`` (for an interval-only value, the ``[low, high]`` pair)."""

    key: str
    value: Scalar
    source: str
    ci_low: float | None = None
    ci_high: float | None = None
    k: int | None = None
    n: int | None = None
    unit: str | None = None
    estimated: bool = False
    censored: str | None = None
    bound: float | None = None
    note: str | None = None
    reportable: bool = True


@dataclass(frozen=True)
class Section:
    key: str
    values: tuple[Value, ...] = ()
    missing: str | None = None
    caveat: str | None = None


def _finite(value: Any) -> Any:
    return None if isinstance(value, float) and math.isnan(value) else value


def leaf(key: str, doc: Doc, where: str, **extra: Any) -> Value:
    return Value(key=key, value=_finite(doc.at(where)), source=doc.source(where), **extra)


def interval(doc: Doc, where: str) -> tuple[float | None, float | None]:
    """A ``{low, high}`` object or a ``[low, high]`` pair."""
    found = doc.at(where)
    low, high = (found[CI_LOW], found[CI_HIGH]) if isinstance(found, Mapping) else found
    return _finite(low), _finite(high)


def with_ci(key: str, doc: Doc, where: str, ci: str, **extra: Any) -> Value:
    low, high = interval(doc, ci)
    return leaf(key, doc, where, ci_low=low, ci_high=high, **extra)


def rate(key: str, doc: Doc, where: str, **extra: Any) -> Value:
    found = doc.at(where)
    method = found.get("ci_method", EXACT_CI)
    note = None if method == EXACT_CI else f"{found.get('n_scenes')} scenes, {method}"
    return with_ci(
        key, doc, f"{where}/value", f"{where}/ci", k=found["k"], n=found["n"], note=note, **extra
    )


def detection_limit(key: str, doc: Doc, where: str) -> Value:
    found = doc.at(where)
    return with_ci(
        key,
        doc,
        f"{where}/value",
        f"{where}/ci",
        n=found["n"],
        unit=found["unit"],
        estimated=True,
        censored=found["censored"],
        bound=found["bound"],
        note=found.get("reason"),
    )
