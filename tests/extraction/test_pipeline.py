"""AC13: the pipeline over the fixture with ground truth and heavy fakes."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import replace

import pytest
from fakes import FAKE, JOURNAL
from settings_factory import SettingsFactory
from support import (
    CountingLock,
    entry,
    load_output,
    plan_overrides,
    profile_of,
    run_pipeline,
)
from synthetic_clip import SyntheticFixtureDataset
from wiring import resolve_plan

from physverify.config.models import Settings
from physverify.extraction._base import StageProvider
from physverify.extraction.pipeline import ExtractionReport
from physverify.schema.scale import ScaleEstimate
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import LengthUnit, ObjectStateTable

Providers = Mapping[str, type[StageProvider]]
HEAVY = (Stage.DETECT, Stage.TRACK, Stage.DEPTH)
TWIN = "fixture_twin"
FAKE_HEAVIES = plan_overrides(detect=FAKE, track=FAKE, depth=FAKE)
PROFILE = profile_of(
    entry(Stage.TRACK, FAKE, "centroid_px_rel", 0.05),
    entry(Stage.DEPTH, FAKE, "depth_rel", 0.1),
)


def cell(track_gt: bool, depth_gt: bool, scale_gt: bool) -> tuple[list[str], tuple[Stage, ...]]:
    """Overrides and ``--gt`` stages of one factorial cell; CAMERA is always ground truth."""
    gt = {Stage.CAMERA}
    gt |= {Stage.DETECT, Stage.TRACK} if track_gt else set()
    gt |= {Stage.DEPTH} if depth_gt else set()
    gt |= {Stage.SCALE} if scale_gt else set()
    return FAKE_HEAVIES, tuple(stage for stage in STAGE_ORDER if stage in gt)


def completed(report: ExtractionReport, clip_id: str) -> list[Stage]:
    return [event.stage for event in report.events if event.clip_id == clip_id]


def assert_bound(
    settings: Settings, providers: Providers, report: ExtractionReport, gt: tuple[Stage, ...]
) -> None:
    plan = resolve_plan(settings, providers, dataset="synthetic_fixture", gt_stages=gt)
    clip_id = report.events[0].clip_id
    lift_key = report.keys[(clip_id, Stage.LIFT)]
    lift = load_output(settings, report, plan, clip_id, Stage.LIFT)
    scale = load_output(settings, report, plan, clip_id, Stage.SCALE)
    assert isinstance(lift, ObjectStateTable) and isinstance(scale, ScaleEstimate)
    assert lift.source_key == scale.lift_key == lift_key
    for stage in (Stage.KINEMATICS, Stage.CONTACTS):
        output = load_output(settings, report, plan, clip_id, stage)
        assert output.source_key == lift_key, stage  # type: ignore[union-attr]
    if Stage.LIFT not in gt:
        assert lift.unit is LengthUnit.RELATIVE


def test_all_ground_truth_plan_completes(
    make_settings: SettingsFactory, providers: Providers, fixture_dataset: SyntheticFixtureDataset
) -> None:
    settings = make_settings()
    report = run_pipeline(settings, providers, fixture_dataset, gt_stages=STAGE_ORDER)
    assert report.failures == ()
    assert completed(report, report.events[0].clip_id) == list(STAGE_ORDER)
    assert_bound(settings, providers, report, STAGE_ORDER)


@pytest.mark.parametrize(
    ("track_gt", "depth_gt", "scale_gt"), list(itertools.product((True, False), repeat=3))
)
def test_every_factorial_cell_completes_with_a_relative_lift(
    make_settings: SettingsFactory,
    providers: Providers,
    fixture_dataset: SyntheticFixtureDataset,
    track_gt: bool,
    depth_gt: bool,
    scale_gt: bool,
) -> None:
    overrides, gt = cell(track_gt, depth_gt, scale_gt)
    settings = make_settings(*overrides)
    report = run_pipeline(settings, providers, fixture_dataset, gt_stages=gt)
    assert report.failures == ()
    assert completed(report, report.events[0].clip_id) == list(STAGE_ORDER)
    assert_bound(settings, providers, report, gt)
    assert JOURNAL.max_live <= 1


def test_the_lock_is_taken_once_around_the_heavy_stages_and_not_on_a_rerun(
    make_settings: SettingsFactory, providers: Providers, fixture_dataset: SyntheticFixtureDataset
) -> None:
    settings = make_settings(*FAKE_HEAVIES)
    lock = CountingLock()
    first = run_pipeline(settings, providers, fixture_dataset, gt_stages=(Stage.CAMERA,), lock=lock)
    clip_id = first.events[0].clip_id
    calls = JOURNAL.calls
    assert lock.calls == 1
    assert calls.count("lock:acquire") == calls.count("lock:release") == 1
    assert calls.index("start:detect:" + clip_id) < calls.index("lock:acquire")
    assert calls.index("lock:acquire") < calls.index("init:detect")
    assert calls.index("run:depth:" + clip_id) < calls.index("lock:release")
    assert calls.index("lock:release") < calls.index("start:camera:" + clip_id)
    assert not any(event.cached for event in first.events)

    rerun_lock = CountingLock()
    again = run_pipeline(
        settings, providers, fixture_dataset, gt_stages=(Stage.CAMERA,), lock=rerun_lock
    )
    assert again.events and all(event.cached for event in again.events)
    assert rerun_lock.calls == 0
    assert dict(again.keys) == dict(first.keys)


def test_a_new_profile_recomputes_only_light_stages(
    make_settings: SettingsFactory, providers: Providers, fixture_dataset: SyntheticFixtureDataset
) -> None:
    settings = make_settings(*FAKE_HEAVIES)
    run_pipeline(settings, providers, fixture_dataset, gt_stages=(Stage.CAMERA,))
    lock = CountingLock()
    report = run_pipeline(
        settings, providers, fixture_dataset, gt_stages=(Stage.CAMERA,), lock=lock, profile=PROFILE
    )
    recomputed = {event.stage for event in report.events if not event.cached}
    assert Stage.LIFT in recomputed
    assert recomputed.isdisjoint(HEAVY)
    assert lock.calls == 0


def test_a_failing_clip_leaves_the_others_complete(
    make_settings: SettingsFactory, providers: Providers, fixture_dataset: SyntheticFixtureDataset
) -> None:
    settings = make_settings(*FAKE_HEAVIES, f"extraction.providers.depth.fake.fail_clip={TWIN}")
    clip = fixture_dataset.clips()[0]
    clips = [clip, replace(clip, clip_id=TWIN)]
    report = run_pipeline(
        settings, providers, fixture_dataset, gt_stages=(Stage.CAMERA,), clips=clips
    )
    assert [(f.clip_id, f.stage) for f in report.failures] == [(TWIN, Stage.DEPTH)]
    assert "fails on" in report.failures[0].error
    assert completed(report, clip.clip_id) == list(STAGE_ORDER)
    assert completed(report, TWIN) == [Stage.DETECT, Stage.TRACK]
