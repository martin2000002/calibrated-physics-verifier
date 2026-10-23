"""Level 1 on the split fixture: pairs (D6), recovered errors, calib-only profile (D3)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import pytest
from synthetic_clip import DATASET

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.models import Settings
from physverify.diagnosis.chain_runner import ChainRunner
from physverify.diagnosis.chain_variants import ladder, oracle_variant
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.measurement_cases import SampledMetric
from physverify.diagnosis.metrics.depth_object import ObjectDepthMetric
from physverify.diagnosis.metrics.track_mask import MaskMetric
from physverify.diagnosis.report import StageErrorReport
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.stage_errors import (
    StagePair,
    measure,
    pair_variants,
    run_stage_errors,
    stage_pairs,
)
from physverify.extraction._base import StageProvider
from physverify.schema.clip import Split
from physverify.schema.error_profile import ErrorEntry, ErrorProfile
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import STAGE_ORDER, Stage

BASE = (f"diagnosis.dataset={DATASET}", "diagnosis.min_samples=3")
EVAL_CACHE = "data/cache_eval"
"""A cache of its own: the eval-error fakes share the plain fakes' cache keys."""
TRACK_SHIFT_PX = 2
"""The fake TRACK default: truth masks moved 2 px along +x."""


def all_metrics(settings: Settings) -> dict[str, SampledMetric]:
    options = diagnosis_settings(settings).metrics
    return {key: cls(options) for key, cls in PLUGIN_KINDS["metrics"]().items()}


def pair_of(settings: Settings, stage: Stage) -> StagePair:
    return next(p for p in stage_pairs(diagnosis_settings(settings)) if p.stage is stage)


def run(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
    *overrides: str,
    write: bool = True,
    profile: ErrorProfile | None = None,
    stage_providers: Mapping[str, type[StageProvider]] | None = None,
) -> tuple[Settings, StageErrorReport]:
    settings = make_settings(*BASE, *overrides)
    ctx = make_diagnosis_context(
        settings, metrics=all_metrics(settings), profile=profile, stage_providers=stage_providers
    )
    return settings, run_stage_errors(ctx, ctx.dataset().clips(), write_profile=write)


def profile_numbers(report: StageErrorReport) -> dict[tuple[Stage, str, str], tuple[float, ...]]:
    return {
        (r.stage, r.provider, r.quantity): (r.bias, r.sigma, r.rms, r.n) for r in report.profile
    }


def row_value(report: StageErrorReport, metric: str, split: Split, key: str) -> float:
    row = next(r for r in report.rows if r.metric == metric and r.split is split)
    return row.values[key]


def test_each_stage_is_measured_against_the_next_rung(
    make_settings: Callable[..., Settings],
) -> None:
    pairs = stage_pairs(diagnosis_settings(make_settings(*BASE)))
    names = [v.name for v in ladder()]
    ladder_pairs = [(p.stage, p.pred.name, p.truth.name) for p in pairs[: len(STAGE_ORDER)]]
    assert ladder_pairs == [(s, names[k], names[k + 1]) for k, s in enumerate(STAGE_ORDER)]
    scale = pairs[len(STAGE_ORDER) :]
    assert [p.pred.name for p in scale] == ["V5/scale=size_prior", "V5/scale=free_fall"]
    assert all(p.truth.name == "V6" and p.stage is Stage.SCALE for p in scale)
    read = [v.name for v in pair_variants(pairs)]
    assert oracle_variant().name == names[-1]
    assert sorted(read) == sorted({*names, *(p.pred.name for p in scale)})


def test_level_one_recovers_the_injected_track_shift(
    make_settings: Callable[..., Settings], make_chain: Callable[..., ChainRunner]
) -> None:
    settings = make_settings(*BASE)
    options = diagnosis_settings(settings).metrics
    runner = make_chain(settings)
    pair = pair_of(settings, Stage.TRACK)
    clip = runner.dataset.clips()[0]
    (measured,) = measure(runner, [clip], {"track/mask": MaskMetric(options)}, [pair], options)
    assert (measured.provider.name, measured.pred_variant, measured.truth_variant) == (
        "diag_fake",
        "V1",
        "V2",
    )
    truth_outputs = runner.outputs(clip, pair.truth)
    assert truth_outputs is not None
    truth = truth_outputs.outputs[Stage.TRACK]
    assert isinstance(truth, MaskTrack)
    radii = [
        math.sqrt(float(frame.sum()) / math.pi)
        for n in range(truth.masks.shape[1])
        for frame in truth.masks[:, n]
        if frame.any()
    ]
    centroid = measured.outcomes[clip.clip_id].samples["centroid_px_rel"]
    np.testing.assert_allclose(centroid[0::2], TRACK_SHIFT_PX / np.array(radii), atol=1e-9)
    np.testing.assert_allclose(centroid[1::2], 0.0, atol=1e-9)


def test_level_one_recovers_the_injected_object_depth_error(
    make_settings: Callable[..., Settings], make_chain: Callable[..., ChainRunner]
) -> None:
    eps = 0.1
    settings = make_settings(
        *BASE,
        "extraction.providers.depth.diag_fake.object_id=ball",
        f"extraction.providers.depth.diag_fake.object_eps={eps}",
    )
    options = diagnosis_settings(settings).metrics
    runner = make_chain(settings)
    clip = runner.dataset.clips()[0]
    metric = {"depth/object_rel": ObjectDepthMetric(options)}
    (measured,) = measure(runner, [clip], metric, [pair_of(settings, Stage.DEPTH)], options)
    ratio = 1.0 + measured.outcomes[clip.clip_id].samples["depth_rel"]
    assert ratio.size > 0
    assert float(ratio.max() / ratio.min()) == pytest.approx(1.0 + eps, abs=1e-6)


def test_calib_profile_is_merged_written_and_blind_to_eval_changes(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
    eval_error_providers: Mapping[str, type[StageProvider]],
) -> None:
    kept = ErrorEntry(
        stage=Stage.DEPTH,
        provider="video_depth_anything",
        provider_version="1",
        quantity="depth_rel",
        unit="1",
        bias=0.3,
        sigma=0.2,
        n=50,
    )
    previous = ErrorProfile(
        created_utc="2026-09-01T00:00:00+00:00", run_id="old", commit="", entries=(kept,)
    )
    settings, before = run(make_settings, make_diagnosis_context, profile=previous)
    path = settings.paths.error_profile
    assert before.profile_written and before.profile_path == str(path)
    loaded = ErrorProfile.load_json(path)
    written = {(r.stage, r.provider, r.quantity) for r in before.profile if r.written}
    assert {(e.stage, e.provider, e.quantity) for e in loaded.entries} == written | {
        (kept.stage, kept.provider, kept.quantity)
    }
    merged = loaded.lookup(kept.stage, kept.provider, kept.quantity)
    track = loaded.lookup(Stage.TRACK, "diag_fake", "centroid_px_rel")
    depth = loaded.lookup(Stage.DEPTH, "diag_fake", "depth_rel")
    assert merged is not None and merged.bias == kept.bias
    assert track is not None and track.bias > 0
    assert depth is not None and abs(depth.bias) <= 1e-6
    fresh = [e for e in loaded.entries if e.provider != kept.provider]
    assert all(e.conditions["split"] == "calib" and e.conditions["n_clips"] == 1 for e in fresh)
    assert {r.split for r in before.rows} == {Split.CALIB, Split.EVAL}
    lift = next(r for r in before.rows if r.metric == "lift/position" and r.split is Split.EVAL)
    assert [c.k for c in lift.coverage["position_z"]] == [1.0, 2.0]

    snapshot = path.read_bytes()
    _, after = run(
        make_settings,
        make_diagnosis_context,
        f"paths.cache_dir={EVAL_CACHE}",
        write=False,
        profile=previous,
        stage_providers=eval_error_providers,
    )
    assert after.profile_path is None and not after.profile_written
    assert not any(r.written for r in after.profile)
    assert path.read_bytes() == snapshot
    np.testing.assert_equal(profile_numbers(after), profile_numbers(before))
    assert after.provenance.error_profile_hash == before.provenance.error_profile_hash
    key = "centroid_px_rel.mean"
    assert row_value(after, "track/mask", Split.EVAL, key) > row_value(
        before, "track/mask", Split.EVAL, key
    )
    assert row_value(after, "track/mask", Split.CALIB, key) == row_value(
        before, "track/mask", Split.CALIB, key
    )
