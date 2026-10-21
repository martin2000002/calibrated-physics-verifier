"""In-frame state and occlusion inside interior mask gaps (it1 change A, J1: T1-T5, T2b)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
from support import SettingsFactory, make_context, run_provider
from synthetic_clip import SyntheticClip

from physverify.checks._base import CheckInputs
from physverify.checks.r4_integrity import PermanenceCheck, PermanenceParams
from physverify.cli.context import PLUGIN_KINDS
from physverify.core.errors import ConfigError
from physverify.extraction._base import GROUND_TRUTH, StageProvider
from physverify.extraction.plan import occlusion_thresholds, resolve_plan
from physverify.extraction.providers._visibility import (
    Box,
    covered_fraction,
    gap_visibility,
    interior_gaps,
    interpolated_boxes,
    mask_box,
    touches_border,
)
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.contacts import ContactTable
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable
from physverify.verification.runner import CHECKS_SECTION, ChecksSettings, build_checks

AT = 0.9
SIZE = 64
ROWS = 12
GAP = (4, 7)  # rows 4, 5, 6 have no mask


def square(x0: int, y0: int, side: int = 8) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[y0 : y0 + side, x0 : x0 + side] = True
    return mask


def moving_object(
    gap: tuple[int, int] = GAP, x0: int = 10, start_x: int | None = None
) -> np.ndarray:
    """(ROWS, H, W) masks of a square moving right by 2 px per row, empty on the gap rows."""
    first = x0 if start_x is None else start_x
    masks = np.stack([square(first + 2 * t, 20) for t in range(ROWS)])
    masks[gap[0] : gap[1]] = False
    return masks


def visible_of(masks: np.ndarray) -> np.ndarray:
    return masks.any(axis=(2, 3))


def visibility(*objects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = np.stack(objects, axis=1)
    visible = visible_of(masks)
    found = gap_visibility(masks, visible, AT)
    return visible, found.in_frame, found.occlusion


def table_of(visible: np.ndarray, in_frame: np.ndarray, occlusion: np.ndarray) -> ObjectStateTable:
    n_t, n_obj = visible.shape
    blank = ObjectStateTable.blank(
        clip_id="gap",
        frame=CoordFrame.CAMERA_OPENCV,
        unit=LengthUnit.RELATIVE,
        source_key="test",
        t=np.arange(n_t, dtype=np.float64) / 24.0,
        frame_index=np.arange(n_t, dtype=np.int64),
        object_ids=[f"o{n}" for n in range(n_obj)],
        labels=["ball"] * n_obj,
        primitive=["sphere"] * n_obj,
    )
    sigma = np.full((n_t, n_obj, 3), 0.01)
    return replace(
        blank, visible=visible, in_frame=in_frame, occlusion=occlusion, position_sigma=sigma
    )


def r4_missing(fixture_clip: SyntheticClip, table: ObjectStateTable) -> list[CheckResult]:
    inputs = CheckInputs(
        clip=fixture_clip.manifest,
        states=table,
        contacts=ContactTable("gap", CoordFrame.CAMERA_OPENCV, LengthUnit.RELATIVE, "test", ()),
        scale=None,
        input_level=InputLevel.PERCEIVED,
        provenance={},
        scene=None,
    )
    return PermanenceCheck(PermanenceParams()).evaluate(inputs)


def test_helpers_find_gaps_boxes_borders_and_cover() -> None:
    shown = np.array([False, True, False, False, True, True, False, True, False])
    assert interior_gaps(shown) == [(2, 4), (6, 7)]
    assert interior_gaps(np.zeros(4, dtype=bool)) == []
    assert touches_border(square(0, 20)) and not touches_border(square(10, 20))
    assert mask_box(square(10, 20)) == Box(10.0, 20.0, 18.0, 28.0)
    assert mask_box(np.zeros((SIZE, SIZE), dtype=bool)) is None
    (middle,) = interpolated_boxes(Box(0, 0, 4, 4), Box(8, 0, 16, 8), 1)
    assert middle == Box(4.0, 0.0, 10.0, 6.0)
    assert covered_fraction(Box(10, 20, 18, 28), square(10, 20)) == 1.0
    assert covered_fraction(Box(10, 20, 18, 28), square(14, 20)) == 0.5
    assert math.isnan(covered_fraction(Box(5, 5, 5, 9), square(0, 0)))


def test_t1_an_isolated_interior_gap_is_in_frame_and_unoccluded(
    fixture_clip: SyntheticClip,
) -> None:
    visible, in_frame, occlusion = visibility(moving_object())
    rows = slice(*GAP)
    assert bool(in_frame[rows, 0].all()) and bool((occlusion[rows, 0] == 0.0).all())
    (result,) = r4_missing(fixture_clip, table_of(visible, in_frame, occlusion))
    assert result.magnitude == GAP[1] - GAP[0]
    assert result.frame_span == (GAP[0], GAP[1] - 1)


def test_t2_a_mask_covering_the_interpolated_box_explains_the_gap(
    fixture_clip: SyntheticClip,
) -> None:
    cover = np.stack([square(8, 18, 30) for _ in range(ROWS)])
    visible, in_frame, occlusion = visibility(moving_object(), cover)
    assert bool((occlusion[slice(*GAP), 0] >= AT).all())
    first, _ = r4_missing(fixture_clip, table_of(visible, in_frame, occlusion))
    assert first.magnitude == 0


def test_t2b_a_half_covered_box_is_unknown_and_the_object_abstains(
    fixture_clip: SyntheticClip,
) -> None:
    half = np.zeros((ROWS, SIZE, SIZE), dtype=bool)
    for i, row in enumerate(range(*GAP)):
        x0 = 18 + 2 * i  # the interpolated box of the gap row spans x0 .. x0 + 8
        half[row, 20:28, x0 : x0 + 4] = True
    visible, in_frame, occlusion = visibility(moving_object(), half)
    assert bool(np.isnan(occlusion[slice(*GAP), 0]).all())
    assert bool(in_frame[slice(*GAP), 0].all())
    first, _ = r4_missing(fixture_clip, table_of(visible, in_frame, occlusion))
    assert first.verdict is Verdict.ABSTAIN
    assert first.abstain_reason is AbstainReason.OCCLUDED


@pytest.mark.parametrize("side", ["before", "after"])
def test_t3_an_enclosing_mask_on_the_border_keeps_the_gap_out_of_frame(side: str) -> None:
    masks = moving_object()
    row = GAP[0] - 1 if side == "before" else GAP[1]
    masks[row] = square(0, 20)
    _, in_frame, occlusion = visibility(masks)
    assert not bool(in_frame[slice(*GAP), 0].any())
    assert bool(np.isnan(occlusion[slice(*GAP), 0]).all())


def test_t4_gaps_at_the_ends_and_a_never_seen_object_are_out_of_frame() -> None:
    masks = moving_object()
    masks[:2] = False
    masks[-2:] = False
    never = np.zeros((ROWS, SIZE, SIZE), dtype=bool)
    _, in_frame, _ = visibility(masks, never)
    assert not bool(in_frame[:2, 0].any()) and not bool(in_frame[-2:, 0].any())
    assert bool(in_frame[slice(*GAP), 0].all())
    assert not bool(in_frame[:, 1].any())


def test_visible_cells_keep_in_frame_and_an_unknown_occlusion() -> None:
    visible, in_frame, occlusion = visibility(moving_object())
    assert bool(in_frame[visible].all())
    assert bool(np.isnan(occlusion[visible]).all())


# --- T5: LIFT version 2 ------------------------------------------------------------------------


def gapped_track(fixture_clip: SyntheticClip) -> Mapping[Stage, object]:
    """The fixture's truth inputs with one interior mask gap cut into the first object."""
    s = make_context(fixture_clip).sampling
    gt = fixture_clip.ground_truth
    track = gt.masks(s)
    masks = track.masks.copy()
    shown = np.flatnonzero(masks[:, 0].any(axis=(1, 2)))
    lo = int(shown[len(shown) // 2])
    masks[lo : lo + 2, 0] = False
    return {
        Stage.TRACK: replace(track, masks=masks),
        Stage.DEPTH: gt.depth(s),
        Stage.CAMERA: gt.camera(s),
    }


def lift(fixture_clip: SyntheticClip, upstream: Mapping[Stage, object], **params: object):
    names = {**dict.fromkeys(STAGE_ORDER, GROUND_TRUTH), Stage.LIFT: "backproject"}
    ctx = make_context(fixture_clip, upstream=upstream, plan=names)  # type: ignore[arg-type]
    table = run_provider(BackprojectLift, ctx, **params)
    assert isinstance(table, ObjectStateTable)
    return table


def test_t5_visible_cells_match_and_none_is_version_one(fixture_clip: SyntheticClip) -> None:
    upstream = gapped_track(fixture_clip)
    new = lift(fixture_clip, upstream)
    old = lift(fixture_clip, upstream, gap_visibility="none")
    track = upstream[Stage.TRACK]
    assert isinstance(track, MaskTrack)
    for name in ("position", "position_sigma", "size", "size_sigma", "visible", "confidence"):
        np.testing.assert_array_equal(getattr(new, name), getattr(old, name))
    np.testing.assert_array_equal(old.in_frame, old.visible)
    assert bool(np.isnan(old.occlusion).all())
    shown = new.visible
    assert bool(new.in_frame[shown].all()) and bool(np.isnan(new.occlusion[shown]).all())
    expected = gap_visibility(track.masks, new.visible, AT)
    np.testing.assert_array_equal(new.in_frame, expected.in_frame)
    np.testing.assert_array_equal(new.occlusion, expected.occlusion)
    assert bool((new.in_frame & ~new.visible).any())


def test_the_lift_version_and_params_separate_the_cache() -> None:
    assert BackprojectLift.version == "2"
    params = BackprojectLift.Params()
    assert params.gap_visibility == "interpolate"  # type: ignore[attr-defined]


def test_the_gap_threshold_is_the_checks_occlusion_threshold(
    make_settings: SettingsFactory, real_providers: Mapping[str, type[StageProvider]]
) -> None:
    settings = make_settings()
    plan = resolve_plan(settings, real_providers, dataset="kubric_synthetic")
    (step,) = [s for s in plan.stages if s.stage is Stage.LIFT]
    checks = build_checks(
        PLUGIN_KINDS["checks"](), settings.section(CHECKS_SECTION, ChecksSettings)
    )
    thresholds = {
        c.check_id: c.params.occluded_at  # type: ignore[attr-defined]
        for c in checks
        if hasattr(getattr(c, "params", None), "occluded_at")
    }
    assert thresholds
    assert set(thresholds.values()) == {step.params.gap_occluded_at}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "overrides",
    [
        ("checks.R4_missing.occluded_at=0.8",),
        ("extraction.providers.lift.backproject.gap_occluded_at=0.8",),
    ],
)
def test_a_split_occlusion_threshold_fails_at_plan_resolution(
    make_settings: SettingsFactory,
    real_providers: Mapping[str, type[StageProvider]],
    overrides: tuple[str, ...],
) -> None:
    """Audit m3: the effective configuration, --set included, must agree."""
    with pytest.raises(ConfigError, match="gap_occluded_at"):
        resolve_plan(make_settings(*overrides), real_providers, dataset="kubric_synthetic")


def test_one_moved_threshold_everywhere_is_accepted(
    make_settings: SettingsFactory, real_providers: Mapping[str, type[StageProvider]]
) -> None:
    moved = [f"checks.{c}.occluded_at=0.8" for c in occlusion_thresholds(make_settings())]
    settings = make_settings(*moved, "extraction.providers.lift.backproject.gap_occluded_at=0.8")
    plan = resolve_plan(settings, real_providers, dataset="kubric_synthetic")
    assert plan.step(Stage.LIFT).params.gap_occluded_at == 0.8  # type: ignore[attr-defined]
    truth = resolve_plan(
        make_settings("checks.R4_missing.occluded_at=0.8"),
        real_providers,
        dataset="kubric_synthetic",
        gt_stages=(Stage.LIFT,),
    )
    assert truth.step(Stage.LIFT).provider.is_ground_truth
