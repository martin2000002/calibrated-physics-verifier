"""ErrorProfile built from Level-1 samples: statistics, merge, atomic write (ADR 0005 §5)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from synthetic_clip import SyntheticFixtureDataset, manifest

from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.diagnosis.error_profile_writer import (
    PROFILE_UNIT,
    MetricSamples,
    ProfileMeta,
    ProviderRef,
    build_profile,
    entry_from_samples,
    merge_profiles,
    write_profile,
)
from physverify.diagnosis.measurement_cases import MetricOutcome
from physverify.extraction._base import GROUND_TRUTH, StageContext
from physverify.schema.clip import FrameSampling
from physverify.schema.error_profile import MAX_RESIDUALS, EntryFilter, ErrorEntry, ErrorProfile
from physverify.schema.stages import Stage

META = ProfileMeta(
    created_utc="2026-09-24T12:00:00+00:00",
    run_id="run-fresh",
    commit="c0ffee0",
    split="calib",
    subset="local",
    dataset="synthetic_fixture",
    dataset_version="v1",
    profile="fast",
)
PARAMS_HASH = "abc123"
CONTRACT = (
    (Stage.TRACK, "centroid_px_rel"),
    (Stage.DEPTH, "depth_rel"),
    (Stage.LIFT, "centroid_rel"),
    (Stage.LIFT, "size_rel"),
)
"""The four quantities A5's light providers read from the profile."""


def provider(stage: Stage, name: str) -> ProviderRef:
    return ProviderRef(stage=stage, name=name, version="1", params_hash=PARAMS_HASH)


def samples(
    metric: str, ref: ProviderRef, per_clip: dict[str, dict[str, Sequence[float]]]
) -> MetricSamples:
    outcomes = {
        clip_id: MetricOutcome(
            samples=MappingProxyType({q: np.array(v, dtype=np.float64) for q, v in values.items()})
        )
        for clip_id, values in per_clip.items()
    }
    return MetricSamples(
        metric=metric,
        provider=ref,
        pred_variant="V1",
        truth_variant="V2",
        outcomes=MappingProxyType(outcomes),
    )


def entry(stage: Stage, provider_name: str, quantity: str, bias: float) -> ErrorEntry:
    return ErrorEntry(
        stage=stage,
        provider=provider_name,
        provider_version="1",
        quantity=quantity,
        unit=PROFILE_UNIT,
        bias=bias,
        sigma=0.1,
        n=40,
    )


def profile(run_id: str, *entries: ErrorEntry) -> ErrorProfile:
    return ErrorProfile(
        created_utc="2026-09-01T00:00:00+00:00", run_id=run_id, commit="", entries=entries
    )


def test_entry_statistics_use_finite_samples_only() -> None:
    residuals = np.array([3.0, 1.0, np.nan, 2.0, np.inf])
    built = entry_from_samples(Stage.TRACK, "diag_fake", "1", "centroid_px_rel", residuals, {})
    assert (built.n, built.bias, built.sigma) == (3, 2.0, 1.0)
    assert built.residuals == (1.0, 2.0, 3.0)
    assert built.unit == PROFILE_UNIT


def test_large_samples_keep_evenly_spaced_quantiles() -> None:
    values = np.random.default_rng(0).normal(0.1, 0.5, 2500)
    built = entry_from_samples(Stage.DEPTH, "vda", "1", "depth_rel", values, {})
    assert built.n == 2500
    assert len(built.residuals) == MAX_RESIDUALS
    levels = (np.arange(MAX_RESIDUALS) + 0.5) / MAX_RESIDUALS
    np.testing.assert_allclose(built.residuals, np.quantile(values, levels))
    assert built.sigma == pytest.approx(float(np.std(values, ddof=1)))


def test_build_profile_keeps_contract_quantities_and_skips_small_ones() -> None:
    track = samples(
        "track/mask",
        provider(Stage.TRACK, "diag_fake"),
        {
            "a": {"centroid_px_rel": [0.1, 0.2, 0.3], "mask_iou": [0.9]},
            "b": {"centroid_px_rel": [0.4, math.nan]},
        },
    )
    depth = samples(
        "depth/object_rel", provider(Stage.DEPTH, "vda"), {"a": {"depth_rel": [0.1, 0.3]}}
    )
    boxes = samples("detect/box", provider(Stage.DETECT, "dino"), {"a": {"box_iou": [1.0] * 5}})
    build = build_profile((track, depth, boxes), META, min_samples=3)
    (only,) = build.profile.entries
    assert (only.stage, only.provider, only.quantity, only.n) == (
        Stage.TRACK,
        "diag_fake",
        "centroid_px_rel",
        4,
    )
    assert only.bias == pytest.approx(0.25)
    assert only.conditions == {
        "split": "calib",
        "subset": "local",
        "dataset": "synthetic_fixture",
        "dataset_version": "v1",
        "profile": "fast",
        "n_clips": 2,
        "upstream": "ground_truth",
        "params_hash": "abc123",
    }
    (skipped,) = build.skipped
    assert (skipped.stage, skipped.provider, skipped.quantity, skipped.n) == (
        Stage.DEPTH,
        "vda",
        "depth_rel",
        2,
    )
    assert skipped.bias == pytest.approx(0.2)
    assert build.profile.run_id == META.run_id


def test_empty_quantities_are_skipped_even_without_a_minimum() -> None:
    lift = samples(
        "lift/position",
        provider(Stage.LIFT, "backproject"),
        {"a": {"centroid_rel": [0.1], "size_rel": []}},
    )
    build = build_profile((lift,), META, min_samples=0)
    assert [e.quantity for e in build.profile.entries] == ["centroid_rel"]
    assert [(s.quantity, s.n) for s in build.skipped] == [("size_rel", 0)]


def test_merge_replaces_only_the_renewed_entries() -> None:
    previous = profile(
        "run-old",
        entry(Stage.TRACK, "diag_fake", "centroid_px_rel", 9.0),
        entry(Stage.DEPTH, "vda", "depth_rel", 0.5),
    )
    fresh = profile("run-fresh", entry(Stage.TRACK, "diag_fake", "centroid_px_rel", 0.25))
    merged = merge_profiles(previous, fresh)
    assert merged.run_id == "run-fresh"
    assert [(e.stage, e.bias) for e in merged.entries] == [(Stage.DEPTH, 0.5), (Stage.TRACK, 0.25)]
    assert merge_profiles(None, fresh).entries == fresh.entries


def test_write_round_trips_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "error_profile.json"
    written = profile("run-fresh", entry(Stage.TRACK, "diag_fake", "centroid_px_rel", 0.25))
    write_profile(path, written)
    assert ErrorProfile.load_json(path).to_dict() == written.to_dict()
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_failed_write_keeps_the_previous_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "error_profile.json"
    old = profile("run-old", entry(Stage.DEPTH, "vda", "depth_rel", 0.5))
    write_profile(path, old)

    def fail(self: ErrorProfile, target: object) -> None:
        Path(str(target)).write_text("{half", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(ErrorProfile, "save_json", fail)
    with pytest.raises(OSError, match="disk full"):
        write_profile(path, profile("run-new"))
    monkeypatch.undo()
    assert ErrorProfile.load_json(path).run_id == "run-old"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def stage_context(
    settings: Settings,
    plan: dict[Stage, str],
    loaded: ErrorProfile,
    params_hash: str = PARAMS_HASH,
    profile_name: str = "fast",
) -> StageContext:
    clip = manifest()
    return StageContext(
        clip=clip,
        video=SyntheticFixtureDataset().video(clip),
        sampling=FrameSampling(
            frame_indices=np.arange(clip.n_frames, dtype=np.int64),
            width=clip.width,
            height=clip.height,
            native_width=clip.width,
            native_height=clip.height,
            seconds_per_frame=clip.seconds_per_frame,
            step=1,
            max_side=None,
        ),
        upstream=MappingProxyType({}),
        plan=MappingProxyType(plan),
        entry_filter=EntryFilter(profile_name, dict.fromkeys(plan, params_hash)),
        gt=None,
        error_profile=loaded,
        device="cpu",
        threads=1,
        paths=settings.paths,
    )


def test_light_providers_read_the_contract_entries(
    tmp_path: Path, make_settings: Callable[..., Settings]
) -> None:
    track = samples(
        "track/mask", provider(Stage.TRACK, "diag_fake"), {"a": {"centroid_px_rel": [0.1, 0.3]}}
    )
    depth = samples(
        "depth/object_rel", provider(Stage.DEPTH, "diag_fake"), {"a": {"depth_rel": [0.2, 0.4]}}
    )
    lift = samples(
        "lift/position",
        provider(Stage.LIFT, "backproject"),
        {"a": {"centroid_rel": [0.5, 0.7], "size_rel": [-0.1, 0.1]}},
    )
    path = tmp_path / "error_profile.json"
    write_profile(path, build_profile((track, depth, lift), META, min_samples=2).profile)
    plan = {Stage.TRACK: "diag_fake", Stage.DEPTH: "diag_fake", Stage.LIFT: "backproject"}
    ctx = stage_context(make_settings(), plan, ErrorProfile.load_json(path))
    found = {(stage, q): ctx.error_entry(stage, q) for stage, q in CONTRACT}
    assert all(e is not None and e.n == 2 and "/" not in e.provider for e in found.values())
    biases = {key: e.bias for key, e in found.items() if e is not None}
    assert biases == pytest.approx(
        {CONTRACT[0]: 0.2, CONTRACT[1]: 0.3, CONTRACT[2]: 0.6, CONTRACT[3]: 0.0}
    )
    exact = stage_context(
        make_settings(), {Stage.TRACK: GROUND_TRUTH}, ErrorProfile.load_json(path)
    )
    track_entry = exact.error_entry(Stage.TRACK, "centroid_px_rel")
    assert track_entry is not None and track_entry.sigma == 0.0


def conditioned(stage: Stage, quantity: str, bias: float, **conditions: str) -> ErrorEntry:
    base = entry(stage, "diag_fake", quantity, bias)
    return replace(base, conditions={"profile": "fast", "params_hash": PARAMS_HASH, **conditions})


@pytest.mark.parametrize("changed", [{"profile": "precise"}, {"params_hash": "other"}])
def test_merge_refuses_an_entry_of_another_profile_or_params(changed: dict[str, str]) -> None:
    previous = profile("run-old", conditioned(Stage.TRACK, "centroid_px_rel", 9.0))
    fresh = profile("run-fresh", conditioned(Stage.TRACK, "centroid_px_rel", 0.25, **changed))
    with pytest.raises(ConfigError, match="track/diag_fake/centroid_px_rel"):
        merge_profiles(previous, fresh)
    merged = merge_profiles(previous, fresh, replace_mismatched=True)
    assert [e.bias for e in merged.entries] == [0.25]


def test_merge_then_lookup_returns_the_fresh_entry_for_its_params() -> None:
    previous = profile(
        "run-old",
        conditioned(Stage.TRACK, "centroid_px_rel", 9.0),
        conditioned(Stage.DEPTH, "depth_rel", 0.5),
    )
    fresh = profile("run-fresh", conditioned(Stage.TRACK, "centroid_px_rel", 0.25))
    merged = merge_profiles(previous, fresh)
    plan = EntryFilter("fast", {Stage.TRACK: PARAMS_HASH})
    found = merged.lookup(Stage.TRACK, "diag_fake", "centroid_px_rel", plan)
    assert found is not None and found.bias == 0.25
    other = EntryFilter("fast", {Stage.TRACK: "other"})
    assert merged.lookup(Stage.TRACK, "diag_fake", "centroid_px_rel", other) is None


def test_a_reader_with_other_params_gets_no_entry(
    tmp_path: Path, make_settings: Callable[..., Settings]
) -> None:
    track = samples(
        "track/mask", provider(Stage.TRACK, "diag_fake"), {"a": {"centroid_px_rel": [0.1, 0.3]}}
    )
    loaded = build_profile((track,), META, min_samples=2).profile
    plan = {Stage.TRACK: "diag_fake"}
    same = stage_context(make_settings(), plan, loaded)
    other = stage_context(make_settings(), plan, loaded, params_hash="other")
    precise = stage_context(make_settings(), plan, loaded, profile_name="precise")
    assert same.error_entry(Stage.TRACK, "centroid_px_rel") is not None
    assert other.error_entry(Stage.TRACK, "centroid_px_rel") is None
    assert precise.error_entry(Stage.TRACK, "centroid_px_rel") is None
