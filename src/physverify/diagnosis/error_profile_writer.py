"""Level-1 samples turned into the ErrorProfile read by light providers (ADR 0005 §5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.core.errors import ConfigError
from physverify.diagnosis.atomic_io import write_atomic
from physverify.diagnosis.cases import sample_stats
from physverify.schema.error_profile import (
    MAX_RESIDUALS,
    PARAMS_HASH_CONDITION,
    PROFILE_CONDITION,
    ErrorEntry,
    ErrorProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from physverify.diagnosis.cases import FloatArray, MetricOutcome
    from physverify.schema.stages import Stage

PROFILE_UNIT = "1"
UPSTREAM_CONDITION = "ground_truth"

PROFILE_QUANTITIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "track/mask": ("centroid_px_rel",),
        "depth/object_rel": ("depth_rel",),
        "lift/position": ("centroid_rel", "size_rel"),
        "scale/relative": ("scale_rel",),
        "kinematics/derivatives": ("vel_rel", "acc_rel"),
        "contacts/events": ("gap_rel",),
    }
)
"""metric_key -> sample names that become profile entries."""


@dataclass(frozen=True)
class ProviderRef:
    """The provider a set of samples measures, as named in the profile (no ``stage/`` prefix)."""

    stage: Stage
    name: str
    version: str
    params_hash: str


@dataclass(frozen=True)
class MetricSamples:
    """Per-clip outcomes of one metric for one provider, measured pred variant vs truth variant."""

    metric: str
    provider: ProviderRef
    pred_variant: str
    truth_variant: str
    outcomes: Mapping[str, MetricOutcome]


StageSamples = tuple[MetricSamples, ...]


@dataclass(frozen=True)
class ProfileMeta:
    created_utc: str
    run_id: str
    commit: str
    split: str
    subset: str
    dataset: str
    dataset_version: str
    profile: str


@dataclass(frozen=True)
class SkippedEntry:
    """An entry left out of the profile because it has fewer than ``min_samples`` samples."""

    stage: Stage
    provider: str
    quantity: str
    n: int
    bias: float
    sigma: float


@dataclass(frozen=True)
class ProfileBuild:
    profile: ErrorProfile
    skipped: tuple[SkippedEntry, ...]


def finite_samples(arrays: Iterable[FloatArray]) -> FloatArray:
    parts = [np.asarray(a, dtype=np.float64).ravel() for a in arrays]
    joined = np.concatenate(parts) if parts else np.zeros(0)
    return joined[np.isfinite(joined)]


def subsample(residuals: FloatArray, limit: int = MAX_RESIDUALS) -> FloatArray:
    """All residuals sorted when they fit; otherwise ``limit`` evenly spaced mid-quantiles."""
    values = np.sort(np.asarray(residuals, dtype=np.float64).ravel())
    if values.size <= limit:
        return values
    return np.asarray(np.quantile(values, (np.arange(limit) + 0.5) / limit), dtype=np.float64)


def entry_from_samples(
    stage: Stage,
    provider: str,
    version: str,
    quantity: str,
    residuals: FloatArray,
    conditions: Mapping[str, Any],
) -> ErrorEntry:
    values = finite_samples((residuals,))
    bias, sigma, _, _ = sample_stats(values)
    return ErrorEntry(
        stage=stage,
        provider=provider,
        provider_version=version,
        quantity=quantity,
        unit=PROFILE_UNIT,
        bias=bias,
        sigma=sigma,
        n=int(values.size),
        residuals=tuple(float(v) for v in subsample(values)),
        conditions=dict(conditions),
    )


def entry_conditions(meta: ProfileMeta, samples: MetricSamples) -> dict[str, Any]:
    return {
        "split": meta.split,
        "subset": meta.subset,
        "dataset": meta.dataset,
        "dataset_version": meta.dataset_version,
        PROFILE_CONDITION: meta.profile,
        "n_clips": len(samples.outcomes),
        "upstream": UPSTREAM_CONDITION,
        PARAMS_HASH_CONDITION: samples.provider.params_hash,
    }


def quantity_samples(samples: MetricSamples, quantity: str) -> FloatArray:
    return finite_samples(
        o.samples[quantity] for o in samples.outcomes.values() if quantity in o.samples
    )


def build_profile(samples: StageSamples, meta: ProfileMeta, min_samples: int) -> ProfileBuild:
    entries: list[ErrorEntry] = []
    skipped: list[SkippedEntry] = []
    for measured in samples:
        ref = measured.provider
        for quantity in PROFILE_QUANTITIES.get(measured.metric, ()):
            values = quantity_samples(measured, quantity)
            if values.size < max(min_samples, 1):
                bias, sigma, _, _ = sample_stats(values)
                skipped.append(
                    SkippedEntry(ref.stage, ref.name, quantity, values.size, bias, sigma)
                )
                continue
            conditions = entry_conditions(meta, measured)
            entries.append(
                entry_from_samples(ref.stage, ref.name, ref.version, quantity, values, conditions)
            )
    profile = ErrorProfile(
        created_utc=meta.created_utc,
        run_id=meta.run_id,
        commit=meta.commit,
        entries=tuple(entries),
    )
    return ProfileBuild(profile=profile, skipped=tuple(skipped))


def entry_identity(entry: ErrorEntry) -> tuple[str, str, str]:
    return (entry.stage.value, entry.provider, entry.quantity)


IDENTITY_CONDITIONS = (PROFILE_CONDITION, PARAMS_HASH_CONDITION)
"""Conditions an entry may only be replaced under: one file per extraction profile."""


def conflicting_entries(
    previous: ErrorProfile, fresh: ErrorProfile
) -> tuple[tuple[ErrorEntry, ErrorEntry], ...]:
    """(previous, fresh) pairs of one identity whose profile or params_hash differ."""
    renewed = {entry_identity(e): e for e in fresh.entries}
    return tuple(
        (old, new)
        for old in previous.entries
        if (new := renewed.get(entry_identity(old))) is not None
        and any(old.conditions.get(c) != new.conditions.get(c) for c in IDENTITY_CONDITIONS)
    )


def merge_profiles(
    previous: ErrorProfile | None, fresh: ErrorProfile, *, replace_mismatched: bool = False
) -> ErrorProfile:
    """Entries of ``fresh`` replace the same (stage, provider, quantity); the rest are kept.

    Replacing an entry measured under another profile or other params is refused unless
    ``replace_mismatched``: that is a different file (``paths.error_profile``) or a deliberate
    re-measurement."""
    kept = () if previous is None else previous.entries
    conflicts = (
        () if previous is None or replace_mismatched else conflicting_entries(previous, fresh)
    )
    if conflicts:
        raise ConfigError(_conflict_message(conflicts))
    renewed = {entry_identity(e) for e in fresh.entries}
    merged = [e for e in kept if entry_identity(e) not in renewed] + list(fresh.entries)
    return ErrorProfile(
        created_utc=fresh.created_utc,
        run_id=fresh.run_id,
        commit=fresh.commit,
        entries=tuple(sorted(merged, key=entry_identity)),
    )


def _conflict_message(conflicts: Iterable[tuple[ErrorEntry, ErrorEntry]]) -> str:
    lines = [
        f"{'/'.join(entry_identity(old))}: "
        + ", ".join(
            f"{c} {old.conditions.get(c)!r} -> {new.conditions.get(c)!r}"
            for c in IDENTITY_CONDITIONS
        )
        for old, new in conflicts
    ]
    return (
        "refusing to replace error-profile entries measured under another profile or params "
        "(use one paths.error_profile per profile):\n  " + "\n  ".join(lines)
    )


def write_profile(path: Path, profile: ErrorProfile) -> None:
    """Atomic write: readers never see a half-written profile."""
    write_atomic(path, profile.save_json)
