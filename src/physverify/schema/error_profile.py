"""Measured per-stage error (written by the diagnosis cycle, read by light providers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from physverify.schema._validate import (
    check_enum,
    check_finite,
    check_instance,
    check_int,
    check_json_mapping,
    check_sigma,
    check_str,
)
from physverify.schema.errors import SchemaError
from physverify.schema.io import SCHEMA_VERSION, Record, stable_hash
from physverify.schema.stages import Stage

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Self

MAX_RESIDUALS = 1000
"""Residuals kept per entry for Monte Carlo resampling."""


@dataclass(frozen=True, eq=False)
class ErrorEntry(Record):
    """Error of one quantity of one provider (e.g. ``scale_rel``), with its residual sample."""

    stage: Stage
    provider: str
    provider_version: str
    quantity: str
    unit: str
    bias: float
    sigma: float
    n: int
    residuals: tuple[float, ...] = ()
    conditions: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[Stage, str]:
        return (self.stage, self.provider)

    @classmethod
    def exact(cls, stage: Stage, provider: str, quantity: str, unit: str = "") -> Self:
        """Zero error: what a ground-truth upstream provider contributes."""
        return cls(
            stage=stage,
            provider=provider,
            provider_version="exact",
            quantity=quantity,
            unit=unit,
            bias=0.0,
            sigma=0.0,
            n=0,
        )

    def validate(self) -> None:
        check_enum("stage", self.stage, Stage)
        for label in ("provider", "provider_version", "quantity"):
            check_str(label, getattr(self, label))
        check_str("unit", self.unit, allow_empty=True)
        check_finite("bias", self.bias)
        check_sigma("sigma", self.sigma)
        check_int("n", self.n, minimum=0)
        if not isinstance(self.residuals, tuple) or len(self.residuals) > MAX_RESIDUALS:
            raise SchemaError(f"residuals must be a tuple of at most {MAX_RESIDUALS} numbers")
        for value in self.residuals:
            check_finite("residuals", value)
        check_json_mapping("conditions", self.conditions)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            stage=Stage(data["stage"]),
            provider=data["provider"],
            provider_version=data["provider_version"],
            quantity=data["quantity"],
            unit=data["unit"],
            bias=data["bias"],
            sigma=data["sigma"],
            n=data["n"],
            residuals=tuple(data["residuals"]),
            conditions=data["conditions"],
        )


@dataclass(frozen=True, eq=False)
class ErrorProfile(Record):
    """All measured entries of one diagnosis run."""

    created_utc: str
    run_id: str
    commit: str
    entries: tuple[ErrorEntry, ...]
    schema_version: str = SCHEMA_VERSION

    def lookup(self, stage: Stage, provider: str, quantity: str) -> ErrorEntry | None:
        for entry in self.entries:
            if entry.key == (stage, provider) and entry.quantity == quantity:
                return entry
        return None

    def entries_hash(self, keys: Iterable[tuple[Stage, str]]) -> str:
        """Hash of the entries of the given (stage, provider) keys only."""
        wanted = set(keys)
        selected = [e.to_dict() for e in self.entries if e.key in wanted]
        return stable_hash(
            sorted(selected, key=lambda d: (d["stage"], d["provider"], d["quantity"]))
        )

    def validate(self) -> None:
        for label in ("created_utc", "run_id", "schema_version"):
            check_str(label, getattr(self, label))
        check_str("commit", self.commit, allow_empty=True)
        if not isinstance(self.entries, tuple):
            raise SchemaError("entries must be a tuple")
        for entry in self.entries:
            check_instance("entries", entry, ErrorEntry)
        keys = [(e.stage, e.provider, e.quantity) for e in self.entries]
        if len(set(keys)) != len(keys):
            raise SchemaError("duplicate (stage, provider, quantity) entry")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            created_utc=data["created_utc"],
            run_id=data["run_id"],
            commit=data["commit"],
            entries=tuple(ErrorEntry.from_dict(e) for e in data["entries"]),
            schema_version=data["schema_version"],
        )


def no_profile_hash() -> str:
    """Hash used for a missing profile; equal to a profile without the declared entries."""
    return stable_hash([])
