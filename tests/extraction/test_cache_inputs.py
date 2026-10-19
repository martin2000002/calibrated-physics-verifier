"""Clip inputs and the params filter in stage keys (review items 0a.2, 0a.3, 0a.4)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from pydantic import BaseModel

from physverify.core.hashing import error_profile_hash, params_hash
from physverify.extraction._base import GROUND_TRUTH, StageProvider
from physverify.extraction.cache import stage_key
from physverify.extraction.plan import StagePlan
from physverify.schema.clip import FrameSampling, ObjectQuery
from physverify.schema.error_profile import EntryFilter, ErrorEntry, ErrorProfile
from physverify.schema.stages import Stage


class FakeParams(BaseModel):
    alpha: float = 0.5


PH = params_hash(FakeParams().model_dump(mode="json"))
NAMES = {
    Stage.DETECT: "dino",
    Stage.TRACK: "sam",
    Stage.DEPTH: "vda",
    Stage.CAMERA: "fov",
    Stage.LIFT: "bp",
    Stage.SCALE: "sp",
    Stage.KINEMATICS: "sg",
    Stage.CONTACTS: "px",
}
FILTER = EntryFilter(profile="fast", params_hashes=dict.fromkeys(NAMES, PH))
SAMPLING = FrameSampling(
    frame_indices=np.arange(0, 24, 2, dtype=np.int64),
    width=64,
    height=48,
    native_width=128,
    native_height=96,
    seconds_per_frame=1 / 24,
    step=2,
    max_side=64,
)
QUERIES = (ObjectQuery("red ball", 1), ObjectQuery("table"))
UPSTREAM = {stage: f"k-{stage.value}" for stage in Stage}
PROFILE_KEYS = ((Stage.TRACK, "sam"), (Stage.DEPTH, "vda"), (Stage.LIFT, "bp"))


def golden_entry(stage: Stage, quantity: str, sigma: float, params: str = PH) -> ErrorEntry:
    return ErrorEntry(
        stage=stage,
        provider=NAMES[stage],
        provider_version="1",
        quantity=quantity,
        unit="1",
        bias=0.0,
        sigma=sigma,
        n=10,
        conditions={"params_hash": params, "profile": "fast"},
    )


PROFILE = ErrorProfile(
    created_utc="2026-10-19T00:00:00+00:00",
    run_id="golden",
    commit="",
    entries=(
        golden_entry(Stage.TRACK, "centroid_px_rel", 0.1),
        golden_entry(Stage.DEPTH, "depth_rel", 0.2),
        golden_entry(Stage.LIFT, "centroid_rel", 0.3),
    ),
)


def fake(
    stage: Stage,
    *,
    gt: bool,
    requires: tuple[Stage, ...] = (),
    reads: tuple[Stage, ...] = (),
    heavy: bool = False,
    own: bool = False,
) -> type[StageProvider]:
    attrs = {
        "stage": stage,
        "name": GROUND_TRUTH if gt else NAMES[stage],
        "version": "1",
        "requires": requires,
        "heavy": heavy,
        "is_ground_truth": gt,
        "error_profile_reads": reads,
        "weights": (),
        "Params": FakeParams,
        "reads_own_error": own,
    }
    return cast("type[StageProvider]", type(f"Fake{stage.value}", (), attrs))


LIFT_READS = (Stage.TRACK, Stage.DEPTH)
CASES: dict[str, type[StageProvider]] = {
    "detect": fake(Stage.DETECT, gt=False, heavy=True),
    "detect_gt": fake(Stage.DETECT, gt=True),
    "track": fake(Stage.TRACK, gt=False, requires=(Stage.DETECT,), heavy=True),
    "track_gt": fake(Stage.TRACK, gt=True, requires=(Stage.DETECT,)),
    "depth": fake(Stage.DEPTH, gt=False, heavy=True),
    "depth_gt": fake(Stage.DEPTH, gt=True),
    "camera": fake(Stage.CAMERA, gt=False),
    "camera_gt": fake(Stage.CAMERA, gt=True),
    "lift": fake(
        Stage.LIFT, gt=False, requires=(*LIFT_READS, Stage.CAMERA), reads=LIFT_READS, own=True
    ),
    "kinematics": fake(Stage.KINEMATICS, gt=False, requires=(Stage.LIFT,)),
}

GOLDEN_786FCDD = {
    "detect": "6c5039ec3c84761dc0374aa434d1207bf6cf02e2f7fbd53d1222ef877ed38793",
    "detect_gt": "82f82e66c74960891a2d66261a1a3f10cd6490278fef9463b539798de128039a",
    "track": "9ccaa538de505aaf0b666e693741865c52193e0a8734a099747daf6d2e41715d",
    "track_gt": "318718d554750da8ffd3dc122b30e7c4abdb4ff703935294974eae2c76352064",
    "depth": "f2641664189263b0e1734a192accfbf5174b5d7c9c2808096afa6ad280a96d7b",
    "depth_gt": "d7f59b0afbbd883cc237aa4946b20e8aee25892921a539617d251db2d06bba77",
    "camera": "1cafac2a744290d0aa37ebcbbe0fe7f1e674474f30d5e3b927f6df6968831b85",
    "camera_gt": "ff77f14fb35b0e2775b03e8f09de2224c232fb8857cd4f8c3bd8f07b36f82ef4",
    "lift": "5211b073781601104e0dd9537ab384a636f3904fbbdda43ddf454940df3f9a90",
    "kinematics": "a78c1697c0d4c4b5c8c9a32fa2c49a93f90b12ccc8b54faa009d33f18e7d8bd4",
}
"""Keys computed by ``stage_key`` at commit 786fcdd, before clip inputs entered the key."""
GOLDEN_PROFILE_HASH = "a3d666d20951dab738469699ca5faf882978b5560457ec323228cc088cf4bd90"
UNCHANGED = ["detect_gt", "track", "track_gt", "depth", "depth_gt", "camera", "camera_gt"]
"""Stages that read neither the queries nor the time base: their artifacts stay valid."""


def key(case: str, **changes: Any) -> str:
    provider = CASES[case]
    kwargs: dict[str, Any] = {
        "clip_id": "clip-golden",
        "object_queries": QUERIES,
        "sampling": SAMPLING,
        "upstream": UPSTREAM,
        "plan": {**NAMES, provider.stage: provider.name},
        "entry_filter": FILTER,
        "gt_version": "gt-v1" if provider.is_ground_truth else None,
        "error_profile": PROFILE,
        "device": "cuda",
    }
    kwargs.update(changes)
    return stage_key(StagePlan(provider.stage, provider, FakeParams()), **kwargs)


@pytest.mark.parametrize("case", UNCHANGED)
def test_stages_without_new_inputs_keep_their_786fcdd_keys(case: str) -> None:
    assert key(case) == GOLDEN_786FCDD[case]
    assert key(case, object_queries=()) == GOLDEN_786FCDD[case]


@pytest.mark.parametrize("case", ["detect", "lift", "kinematics"])
def test_stages_with_new_inputs_get_new_keys(case: str) -> None:
    assert key(case) != GOLDEN_786FCDD[case]


def test_the_queries_change_only_the_perceived_detector_key() -> None:
    other = (ObjectQuery("red ball", 2), ObjectQuery("table"))
    reordered = tuple(reversed(QUERIES))
    assert len({key("detect"), key("detect", object_queries=other)}) == 2
    assert key("detect") != key("detect", object_queries=reordered)
    for case in ("detect_gt", "track", "depth", "camera", "lift"):
        assert key(case) == key(case, object_queries=other), case


def test_the_time_base_changes_only_the_light_stages_from_lift() -> None:
    slower = replace(SAMPLING, seconds_per_frame=1 / 12)
    for case in ("lift", "kinematics"):
        assert key(case) != key(case, sampling=slower), case
    for case in ("detect", "detect_gt", "track", "track_gt", "depth", "depth_gt", "camera"):
        assert key(case) == key(case, sampling=slower), case


def test_matching_params_keep_the_786fcdd_profile_hash() -> None:
    assert error_profile_hash(PROFILE, PROFILE_KEYS) == GOLDEN_PROFILE_HASH
    assert error_profile_hash(PROFILE, PROFILE_KEYS, FILTER) == GOLDEN_PROFILE_HASH


def test_an_entry_measured_with_other_params_leaves_the_key_and_the_lookup() -> None:
    stale = golden_entry(Stage.DEPTH, "depth_rel", 0.2, params="other")
    mixed = replace(PROFILE, entries=(PROFILE.entries[0], stale, PROFILE.entries[2]))
    without = replace(PROFILE, entries=(PROFILE.entries[0], PROFILE.entries[2]))
    assert key("lift", error_profile=mixed) != key("lift")
    assert key("lift", error_profile=mixed) == key("lift", error_profile=without)
    assert mixed.lookup(Stage.DEPTH, "vda", "depth_rel", FILTER) is None
    assert mixed.lookup(Stage.DEPTH, "vda", "depth_rel") is stale
    assert mixed.rejected(PROFILE_KEYS, FILTER) == (stale,)
    assert PROFILE.rejected(PROFILE_KEYS, FILTER) == ()


def test_an_entry_of_another_profile_is_refused_even_with_the_same_params() -> None:
    precise = replace(FILTER, profile="precise")
    assert key("lift", entry_filter=precise) != key("lift")
    assert PROFILE.lookup(Stage.DEPTH, "vda", "depth_rel", precise) is None
    assert PROFILE.rejected(PROFILE_KEYS, precise) == PROFILE.entries
    assert PROFILE.selected(PROFILE_KEYS, precise) == ()


def test_an_entry_that_declares_no_condition_is_admitted() -> None:
    bare = replace(PROFILE.entries[1], conditions={})
    assert EntryFilter(profile="precise", params_hashes={Stage.DEPTH: "other"}).admits(bare)
    assert EntryFilter().admits(PROFILE.entries[1])
