from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
import records

from physverify.core.cache_keys import cache_key
from physverify.core.errors import MissingInputError
from physverify.extraction._base import GROUND_TRUTH, error_profile_keys
from physverify.schema.error_profile import ErrorEntry, ErrorProfile
from physverify.schema.stages import Stage


def _profile(depth_sigma: float = 0.05, detect_sigma: float = 0.2) -> ErrorProfile:
    entries = (
        ErrorEntry(
            stage=Stage.DEPTH,
            provider="vda_small",
            provider_version="1",
            quantity="abs_rel",
            unit="",
            bias=0.0,
            sigma=depth_sigma,
            n=10,
        ),
        ErrorEntry(
            stage=Stage.DETECT,
            provider="grounding_dino_tiny",
            provider_version="1",
            quantity="box_iou",
            unit="",
            bias=0.0,
            sigma=detect_sigma,
            n=10,
        ),
    )
    return ErrorProfile(
        created_utc="2026-09-24T00:00:00+00:00", run_id="r", commit="", entries=entries
    )


PLAN = {Stage.DEPTH: "vda_small", Stage.DETECT: "grounding_dino_tiny", Stage.TRACK: "edgetam"}


def _key(**changes: Any) -> str:
    base: dict[str, Any] = {
        "clip_id": "clip",
        "stage": Stage.LIFT,
        "provider": "backproject",
        "version": "1",
        "params": {"min_valid": 0.2},
        "sampling": records.sampling(),
        "requires": (Stage.TRACK, Stage.DEPTH),
        "upstream": {Stage.TRACK: "t1", Stage.DEPTH: "d1", Stage.DETECT: "x1"},
        "is_ground_truth": False,
        "gt_version": None,
        "error_profile": _profile(),
        "error_profile_keys": error_profile_keys((Stage.DEPTH,), PLAN),
        "heavy": False,
        "device": "cpu",
    }
    base.update(changes)
    return cache_key(**base)


def test_key_is_deterministic() -> None:
    assert _key() == _key()


@pytest.mark.parametrize(
    "changes",
    [
        {"version": "2"},
        {"params": {"min_valid": 0.3}},
        {"upstream": {Stage.TRACK: "t2", Stage.DEPTH: "d1", Stage.DETECT: "x1"}},
        {"error_profile": _profile(depth_sigma=0.06)},
        {"error_profile": None},
    ],
)
def test_key_changes_with_what_changes_the_output(changes: dict[str, Any]) -> None:
    assert _key(**changes) != _key()


def test_key_changes_with_sampling() -> None:
    grid = records.sampling()
    shifted = dataclasses.replace(grid, frame_indices=grid.frame_indices + 1)
    assert _key(sampling=shifted) != _key()


def test_key_changes_with_ground_truth_version() -> None:
    gt = {"is_ground_truth": True, "provider": GROUND_TRUTH}
    assert _key(**gt, gt_version="v1") != _key(**gt, gt_version="v2")


@pytest.mark.parametrize(
    "changes",
    [
        {"upstream": {Stage.TRACK: "t1", Stage.DEPTH: "d1", Stage.DETECT: "x2"}},
        {"error_profile": _profile(detect_sigma=0.9)},
        {"device": "cuda"},
        {"gt_version": "ignored-when-not-ground-truth"},
    ],
)
def test_key_ignores_what_the_provider_does_not_read(changes: dict[str, Any]) -> None:
    assert _key(**changes) == _key()


def test_device_matters_only_for_heavy_providers() -> None:
    assert _key(heavy=True, device="cpu") != _key(heavy=True, device="cuda")


def test_ground_truth_upstream_declares_no_profile_entry() -> None:
    plan = {**PLAN, Stage.DEPTH: GROUND_TRUTH}
    assert error_profile_keys((Stage.DEPTH, Stage.DETECT), plan) == (
        (Stage.DETECT, "grounding_dino_tiny"),
    )


def test_missing_required_upstream_raises() -> None:
    with pytest.raises(MissingInputError):
        _key(upstream={Stage.TRACK: "t1"})


def test_sampling_hash_ignores_native_metadata() -> None:
    grid = records.sampling()
    same_pixels = dataclasses.replace(grid, seconds_per_frame=grid.seconds_per_frame * 2)
    assert _key(sampling=same_pixels) == _key()
    assert np.array_equal(same_pixels.frame_indices, grid.frame_indices)
