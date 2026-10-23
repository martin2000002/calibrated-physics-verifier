"""AC12: what enters a stage key, and the on-disk stage cache."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeTrack
from settings_factory import SettingsFactory
from support import (
    chain_keys,
    entry,
    profile_of,
    sampling_of,
    truth_outputs,
    with_sigma,
)
from synthetic_clip import SyntheticClip
from wiring import resolve_plan

from physverify.extraction._base import GROUND_TRUTH, StageProvider
from physverify.extraction.cache import StageCache, stage_key
from physverify.extraction.plan import StagePlan
from physverify.extraction.providers.gt_lift import GroundTruthLift
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.schema.clip import SamplingSpec
from physverify.schema.error_profile import EntryFilter
from physverify.schema.io import stable_hash
from physverify.schema.stages import STAGE_ORDER, Stage, StageArtifactMeta

PLAN = {
    **dict.fromkeys(STAGE_ORDER, GROUND_TRUTH),
    Stage.TRACK: "edgetam",
    Stage.DEPTH: "vda_small",
    Stage.LIFT: "backproject",
}
UPSTREAM = {Stage.DETECT: "k-detect", Stage.TRACK: "k-track", Stage.DEPTH: "k-depth"}
UPSTREAM = {**UPSTREAM, Stage.CAMERA: "k-camera"}
P0 = profile_of(
    entry(Stage.TRACK, "edgetam", "centroid_px_rel", 0.1),
    entry(Stage.DEPTH, "vda_small", "depth_rel", 0.1),
    entry(Stage.LIFT, "backproject", "centroid_rel", 0.1),
    entry(Stage.DETECT, "grounding_dino_tiny", "box_iou", 0.1),
)


def key_of(
    clip: SyntheticClip,
    provider: type[StageProvider],
    **changes: Any,
) -> str:
    params = changes.pop("params", {})
    step = StagePlan(provider.stage, provider, provider.Params(**params))
    kwargs: dict[str, Any] = {
        "clip_id": clip.manifest.clip_id,
        "object_queries": clip.manifest.object_queries,
        "sampling": sampling_of(clip.manifest),
        "upstream": UPSTREAM,
        "plan": PLAN,
        "entry_filter": EntryFilter(),
        "gt_version": None,
        "error_profile": P0,
        "device": "cpu",
    }
    kwargs.update(changes)
    return stage_key(step, **kwargs)


def versioned(provider: type[StageProvider], version: str) -> type[StageProvider]:
    return type(f"{provider.__name__}V{version}", (provider,), {"version": version})


def test_the_key_changes_with_every_ingredient(fixture_clip: SyntheticClip) -> None:
    base = key_of(fixture_clip, BackprojectLift)
    coarse = sampling_of(fixture_clip.manifest, SamplingSpec(max_side=48))
    changed = {
        "params": key_of(fixture_clip, BackprojectLift, params={"erode_px": 2}),
        "version": key_of(
            fixture_clip, versioned(BackprojectLift, BackprojectLift.version + "-next")
        ),
        "sampling": key_of(fixture_clip, BackprojectLift, sampling=coarse),
        "upstream": key_of(fixture_clip, BackprojectLift, upstream={**UPSTREAM, Stage.DEPTH: "x"}),
        "read entry": key_of(
            fixture_clip,
            BackprojectLift,
            error_profile=with_sigma(P0, (Stage.DEPTH, "vda_small", "depth_rel"), 0.2),
        ),
        "own entry": key_of(
            fixture_clip,
            BackprojectLift,
            error_profile=with_sigma(P0, (Stage.LIFT, "backproject", "centroid_rel"), 0.2),
        ),
    }
    assert all(key != base for key in changed.values()), changed
    assert len(set(changed.values())) == len(changed)
    gt = key_of(fixture_clip, GroundTruthLift, gt_version="a")
    assert gt != key_of(fixture_clip, GroundTruthLift, gt_version="b")
    heavy = key_of(fixture_clip, FakeTrack, upstream={Stage.DETECT: "k"})
    assert heavy != key_of(fixture_clip, FakeTrack, upstream={Stage.DETECT: "k"}, device="cuda")


def test_the_key_ignores_what_cannot_change_the_output(fixture_clip: SyntheticClip) -> None:
    base = key_of(fixture_clip, BackprojectLift)
    outside = with_sigma(P0, (Stage.DETECT, "grounding_dino_tiny", "box_iou"), 0.9)
    assert key_of(fixture_clip, BackprojectLift, error_profile=outside) == base
    assert key_of(fixture_clip, BackprojectLift, upstream={**UPSTREAM, Stage.DETECT: "x"}) == base
    assert key_of(fixture_clip, BackprojectLift, device="cuda") == base


def test_a_size_prior_override_invalidates_only_the_scale(
    make_settings: SettingsFactory,
    real_providers: Mapping[str, type[StageProvider]],
    fixture_clip: SyntheticClip,
) -> None:
    """KINEMATICS and CONTACTS read LIFT, not SCALE, so a prior edit reaches SCALE alone."""
    before = resolve_plan(make_settings(), real_providers, dataset="x")
    after = resolve_plan(make_settings("size_priors.ball.mean_m=0.2"), real_providers, dataset="x")
    keys = [
        chain_keys(plan, fixture_clip.manifest, gt_version=None, profile=None)
        for plan in (before, after)
    ]
    changed = [stage for stage in STAGE_ORDER if keys[0][stage] != keys[1][stage]]
    assert changed == [Stage.SCALE]


def test_the_precise_detector_reuses_depth_and_camera(
    make_settings: SettingsFactory,
    real_providers: Mapping[str, type[StageProvider]],
    fixture_clip: SyntheticClip,
) -> None:
    before = resolve_plan(
        make_settings("extraction.plan.detect=grounding_dino_tiny", profile="precise"),
        real_providers,
        dataset="x",
    )
    after = resolve_plan(make_settings(profile="precise"), real_providers, dataset="x")
    keys = [
        chain_keys(plan, fixture_clip.manifest, gt_version=None, profile=P0)
        for plan in (before, after)
    ]
    reused = [stage for stage in STAGE_ORDER if keys[0][stage] == keys[1][stage]]
    assert reused == [Stage.DEPTH, Stage.CAMERA]


def meta_for(stage: Stage, provider: str, key: str, clip_id: str) -> StageArtifactMeta:
    return StageArtifactMeta(
        stage=stage,
        provider=provider,
        provider_version="1",
        params_hash="p",
        sampling_hash="s",
        upstream={},
        error_profile_hash="e",
        gt_version=None,
        device=None,
        cache_key=key,
        clip_id=clip_id,
        is_ground_truth=False,
        wall_s=0.0,
        peak_rss_mb=0.0,
        created_utc="2026-09-24T00:00:00+00:00",
    )


def test_every_output_class_round_trips(fixture_clip: SyntheticClip, tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    clip_id = fixture_clip.manifest.clip_id
    outputs = truth_outputs(fixture_clip)
    assert len({type(o) for o in outputs.values()}) == 7
    for stage, output in outputs.items():
        key = f"{stage.value}-key-0123456789"
        cache.store(
            "ds",
            clip_id,
            stage,
            output,
            meta_for(stage, "p", key, clip_id),
            BackprojectLift.Params(),
        )
        assert cache.contains("ds", clip_id, stage, "p", key)
        loaded = cache.load("ds", clip_id, stage, "p", key)
        assert type(loaded) is type(output)
        assert stable_hash(loaded.to_dict()) == stable_hash(output.to_dict()), stage


def test_half_written_entries_and_foreign_keys_miss(
    fixture_clip: SyntheticClip, tmp_path: Path
) -> None:
    cache = StageCache(tmp_path)
    clip_id = fixture_clip.manifest.clip_id
    scale = truth_outputs(fixture_clip)[Stage.SCALE]
    key = "0123456789ab-first"
    final = cache.directory("ds", clip_id, Stage.SCALE, "p", key)
    staging = final.with_name(f"{final.name}.tmp-999")
    staging.mkdir(parents=True)
    scale.save_json(staging / "output.json")
    assert not cache.contains("ds", clip_id, Stage.SCALE, "p", key)
    assert cache.load("ds", clip_id, Stage.SCALE, "p", key) is None
    cache.store(
        "ds",
        clip_id,
        Stage.SCALE,
        scale,
        meta_for(Stage.SCALE, "p", key, clip_id),
        BackprojectLift.Params(),
    )
    other = "0123456789ab-other"
    assert cache.directory("ds", clip_id, Stage.SCALE, "p", other) == final
    assert not cache.contains("ds", clip_id, Stage.SCALE, "p", other)
    assert cache.load("ds", clip_id, Stage.SCALE, "p", other) is None
    assert cache.contains("ds", clip_id, Stage.SCALE, "p", key)


@pytest.mark.parametrize("stage", [Stage.SCALE, Stage.CONTACTS])
def test_scale_and_contacts_are_json(stage: Stage) -> None:
    from physverify.extraction.cache import OUTPUT_FORMAT

    assert OUTPUT_FORMAT[stage] == "json"
    assert {OUTPUT_FORMAT[s] for s in STAGE_ORDER if s not in (Stage.SCALE, Stage.CONTACTS)} == {
        "npz"
    }
