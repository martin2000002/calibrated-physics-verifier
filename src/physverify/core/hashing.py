"""Cache keys of stage artifacts (ADR 0003, section 2.8)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from physverify.core.errors import MissingInputError
from physverify.schema.clip import FrameSampling
from physverify.schema.error_profile import EntryFilter, ErrorProfile, no_profile_hash
from physverify.schema.io import SCHEMA_VERSION, stable_hash
from physverify.schema.stages import Stage


def sampling_hash(sampling: FrameSampling) -> str:
    """Only what changes the pixels a provider sees: frame indices and processing size."""
    return stable_hash(
        {"frame_indices": sampling.frame_indices, "w": sampling.width, "h": sampling.height}
    )


def params_hash(params: Mapping[str, Any]) -> str:
    return stable_hash(dict(params))


def error_profile_hash(
    profile: ErrorProfile | None,
    keys: tuple[tuple[Stage, str], ...],
    accept: EntryFilter | None = None,
) -> str:
    """Hash of the declared entries a lookup can return; a missing profile equals an empty one.

    ``accept`` drops the entries measured under another profile or with other params."""
    if profile is None:
        return no_profile_hash()
    return profile.entries_hash(keys, accept)


def upstream_keys(requires: tuple[Stage, ...], upstream: Mapping[Stage, str]) -> dict[Stage, str]:
    """Cache keys of the required stages only; a missing one is an error."""
    missing = [stage.value for stage in requires if stage not in upstream]
    if missing:
        raise MissingInputError(f"missing upstream cache keys for {missing}")
    return {stage: upstream[stage] for stage in requires}


def cache_key(
    *,
    clip_id: str,
    stage: Stage,
    provider: str,
    version: str,
    params: Mapping[str, Any],
    sampling: FrameSampling,
    requires: tuple[Stage, ...],
    upstream: Mapping[Stage, str],
    is_ground_truth: bool,
    gt_version: str | None,
    error_profile: ErrorProfile | None,
    error_profile_keys: tuple[tuple[Stage, str], ...],
    heavy: bool,
    device: str,
    error_profile_filter: EntryFilter | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> str:
    """sha256 of everything that can change a stage output, and nothing else.

    ``inputs`` holds clip data the provider reads besides pixels (e.g. object queries); None
    leaves the payload, and so every key computed without it, unchanged."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "stage": stage,
        "provider": provider,
        "version": version,
        "params": params_hash(params),
        "sampling": sampling_hash(sampling),
        "upstream": upstream_keys(requires, upstream),
        "error_profile": error_profile_hash(
            error_profile, error_profile_keys, error_profile_filter
        ),
    }
    if inputs is not None:
        payload["inputs"] = dict(inputs)
    if is_ground_truth:
        payload["gt_version"] = gt_version
    if heavy:
        payload["device"] = device
    return stable_hash(payload)
