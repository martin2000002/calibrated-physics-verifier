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
PARAMS_HASH_CONDITION = "params_hash"
"""``conditions`` key: hash of the provider params the entry was measured with."""
PROFILE_CONDITION = "profile"
"""``conditions`` key: extraction profile (fast, precise) the entry was measured under."""


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


@dataclass(frozen=True)
class EntryFilter:
    """The entries a plan may read: those measured under its profile with its stage params.

    An unset constraint (None, or a stage absent from ``params_hashes``) admits everything; an
    entry that does not declare a condition (hand-written) is not bound by it."""

    profile: str | None = None
    params_hashes: Mapping[Stage, str] = field(default_factory=dict)

    def admits(self, entry: ErrorEntry) -> bool:
        return _unbound(entry, PROFILE_CONDITION, self.profile) and _unbound(
            entry, PARAMS_HASH_CONDITION, self.params_hashes.get(entry.stage)
        )


def _unbound(entry: ErrorEntry, condition: str, expected: str | None) -> bool:
    declared = entry.conditions.get(condition)
    return expected is None or declared is None or declared == expected


@dataclass(frozen=True, eq=False)
class ErrorProfile(Record):
    """All measured entries of one diagnosis run."""

    created_utc: str
    run_id: str
    commit: str
    entries: tuple[ErrorEntry, ...]
    schema_version: str = SCHEMA_VERSION

    def lookup(
        self, stage: Stage, provider: str, quantity: str, accept: EntryFilter | None = None
    ) -> ErrorEntry | None:
        """The entry, or None when absent or not admitted by ``accept`` (no fallback)."""
        for entry in self.entries:
            if entry.key == (stage, provider) and entry.quantity == quantity:
                return entry if accept is None or accept.admits(entry) else None
        return None

    def selected(
        self, keys: Iterable[tuple[Stage, str]], accept: EntryFilter | None = None
    ) -> tuple[ErrorEntry, ...]:
        """Entries of the given (stage, provider) keys that ``lookup`` can return."""
        wanted = set(keys)
        return tuple(
            e for e in self.entries if e.key in wanted and (accept is None or accept.admits(e))
        )

    def rejected(
        self, keys: Iterable[tuple[Stage, str]], accept: EntryFilter
    ) -> tuple[ErrorEntry, ...]:
        """Entries of the given keys that ``lookup`` refuses under ``accept``."""
        wanted = set(keys)
        return tuple(e for e in self.entries if e.key in wanted and not accept.admits(e))

    def entries_hash(
        self, keys: Iterable[tuple[Stage, str]], accept: EntryFilter | None = None
    ) -> str:
        """Hash of the entries ``lookup`` can return for the given (stage, provider) keys only.

        When ``accept`` admits every entry the hash equals the unfiltered one."""
        selected = [e.to_dict() for e in self.selected(keys, accept)]
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
