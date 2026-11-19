"""Rest in the image plane for ``contacts/kinematic_scene``: the mask centroid's velocity against
sigma_px = max(e_c r_eq, 1 / sqrt(12)), objects matched by id and rows by frame index."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import scene_contact_cases as cases

from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.providers._image_motion import ObjectImage, PixelError, mask_motion
from physverify.extraction.providers._image_rest import ImageRest, image_rest
from physverify.extraction.providers._savgol import SavgolParams
from physverify.extraction.providers._velocity_tests import rest_runs
from physverify.extraction.providers.gt_kinematics import Params as TruthParams
from physverify.physics.row_time import reference_seconds, seconds_per_row
from physverify.physics.tracks import PIXEL_QUANTIZATION_SIGMA
from physverify.schema.contacts import ContactKind
from physverify.schema.error_profile import CENTROID_PX_REL, ErrorEntry
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import Stage

cs = cases.cs
ALPHA = 0.05
MIN_ROWS = 3
FPS = 24.0
FRAMES = 49
SIZE = 160
RADIUS = 12.0
E_C = 0.3526
ERROR = PixelError(E_C)
EXACT_ENTRY = ErrorEntry.exact(Stage.TRACK, GROUND_TRUTH, CENTROID_PX_REL)
SPEC = SavgolParams().spec(1.0 / FPS)


def times() -> np.ndarray:
    return np.arange(FRAMES, dtype=np.float64) / FPS


def discs(centres: np.ndarray, radius: float = RADIUS) -> np.ndarray:
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    return np.stack([(cols - x) ** 2 + (rows - y) ** 2 <= radius**2 for x, y in centres]).astype(
        np.bool_
    )


def runs_of(masks: np.ndarray, error: PixelError = ERROR) -> np.ndarray:
    image = ObjectImage.of(masks, np.ones(len(masks), dtype=np.bool_))
    motion = mask_motion(image, times(), SPEC, error)
    rest = np.zeros(len(masks), dtype=np.bool_)
    for run in rest_runs(
        motion.velocity, image.valid, motion.velocity_sigma[:, 0], ALPHA, MIN_ROWS
    ):
        rest[run] = True
    return rest


def test_sigma_px_is_the_entry_rms_times_the_equivalent_radius() -> None:
    error = PixelError.of(cases.CENTROID_PX)
    assert error.known and error.relative == pytest.approx(E_C)
    assert error.sigma(np.array([10.0])) == pytest.approx([E_C * 10.0])
    assert not PixelError.of(None).known


def test_an_exact_track_keeps_the_pixel_quantization_floor() -> None:
    exact = PixelError.of(EXACT_ENTRY)
    assert exact.known and exact.relative == 0.0
    assert exact.sigma(np.array([0.0, RADIUS])) == pytest.approx([1.0 / np.sqrt(12.0)] * 2)
    assert ERROR.sigma(np.array([0.5])) == pytest.approx([PIXEL_QUANTIZATION_SIGMA])
    assert np.isnan(PixelError.of(None).sigma(np.array([RADIUS]))).all()


def test_under_an_exact_track_a_still_disc_rests_and_a_throw_apex_does_not() -> None:
    exact = PixelError.of(EXACT_ENTRY)
    assert runs_of(discs(np.tile([80.0, 80.0], (FRAMES, 1))), exact).all()
    t = times() - times()[FRAMES // 2]
    gravity = 0.3 * RADIUS * FPS**2
    centres = np.stack([np.full(FRAMES, 80.0), 60.0 + 0.5 * gravity * t**2], axis=-1)
    assert not runs_of(discs(centres), exact).any()


def test_the_centroid_velocity_and_its_sigma_follow_a_moving_disc() -> None:
    t = times()
    centres = np.stack([40.0 + 30.0 * t, np.full(FRAMES, 80.0)], axis=-1)
    image = ObjectImage.of(discs(centres), np.ones(FRAMES, bool))
    motion = mask_motion(image, t, SPEC, ERROR)
    inner = slice(SPEC.window, FRAMES - SPEC.window)
    assert image.valid.all()
    assert motion.velocity[inner, 0] == pytest.approx(30.0, abs=3.0)
    assert np.allclose(motion.velocity_sigma[:, 0], motion.velocity_sigma[:, 1])
    assert np.all(motion.velocity_sigma[inner, 0] > 0.0)


def test_the_apex_of_a_throw_is_never_at_rest() -> None:
    """Vertical throw with the apex mid-clip and g f / Z = 0.3 r_eq per frame squared, the
    median at the apex of the calibration clips."""
    t = times() - times()[FRAMES // 2]
    gravity = 0.3 * RADIUS * FPS**2
    centres = np.stack([np.full(FRAMES, 80.0), 60.0 + 0.5 * gravity * t**2], axis=-1)
    assert not runs_of(discs(centres)).any()


def test_a_still_disc_with_centroid_noise_below_sigma_px_is_one_run() -> None:
    noise = np.random.default_rng(3).normal(0.0, 0.3 * E_C * RADIUS, (FRAMES, 2))
    centres = np.array([80.0, 80.0]) + noise
    assert runs_of(discs(centres)).all()


def test_rows_without_a_mask_split_the_run() -> None:
    masks = discs(np.tile([80.0, 80.0], (FRAMES, 1)))
    masks[20] = False
    rest = runs_of(masks)
    assert not rest[20] and rest[:20].all() and rest[21:].all()


def still_boxes() -> tuple[cs.sc.Case, MaskTrack]:
    case = cases.boxes_at([0.0, 0.0, 0.0])
    return case, cs.disk_track(case.states, cases.sampling(case))


def test_objects_are_matched_by_id_and_rows_by_frame_index() -> None:
    case, track = still_boxes()
    order = [2, 0]
    dropped = 10
    keep = np.flatnonzero(np.arange(track.masks.shape[0]) != dropped)
    shuffled = replace(
        track,
        sampling=replace(track.sampling, frame_indices=track.sampling.frame_indices[keep]),
        object_ids=track.object_ids[order],
        labels=track.labels[order],
        masks=track.masks[keep][:, order],
        scores=track.scores[keep][:, order],
    )
    rest = ImageRest(shuffled, ERROR, SPEC).mask(case.states, ALPHA, MIN_ROWS)
    assert not rest[:, 1].any()
    for n in (0, 2):
        assert not rest[dropped, n] and rest[:dropped, n].all() and rest[dropped + 1 :, n].all()


def test_an_unknown_centroid_error_has_no_rest_and_says_so() -> None:
    case, track = still_boxes()
    unknown = ImageRest(track, PixelError.of(None), SPEC)
    assert not unknown.known and not unknown.mask(case.states, ALPHA, MIN_ROWS).any()
    found = cases.outcome(case, rest=unknown)
    assert not found.rest_known
    assert not [r for r in found.records if r.kind is ContactKind.RESTING]


def test_the_provider_rests_on_masks_and_needs_the_track_entry() -> None:
    case = cases.stacked_boxes()
    floor = cs.sc.build("floor", [cases.box("box2", 3 * cases.SPACING, 0.0)], [], 25.0).states
    states = cs.joined(case.states, floor)
    entries = cases.measured_entries()
    records = cases.scene_table(case, entries, states=states).records
    assert {r.kind for r in records} == {ContactKind.RESTING}
    assert cases.pair_records(records) and cases.env_records(records)
    del entries[(Stage.TRACK, CENTROID_PX_REL)]
    assert cases.scene_table(case, entries, states=states).records == ()


def test_the_provider_rests_under_a_ground_truth_track() -> None:
    case = cases.stacked_boxes()
    floor = cs.sc.build("floor", [cases.box("box2", 3 * cases.SPACING, 0.0)], [], 25.0).states
    states = cs.joined(case.states, floor)
    entries = cases.measured_entries()
    entries[(Stage.TRACK, CENTROID_PX_REL)] = EXACT_ENTRY
    records = cases.scene_table(case, entries, states=states).records
    assert {r.kind for r in records} == {ContactKind.RESTING}
    assert cases.pair_records(records) and cases.env_records(records)


def test_the_rest_window_is_the_plans_kinematics_window() -> None:
    case, track = still_boxes()
    sampling = cases.sampling(case)
    row = seconds_per_row(case.states.frame_index, sampling.seconds_per_frame)
    wide = SavgolParams(window_s=reference_seconds(11))

    def spec_of(stage_params: dict[Stage, object]) -> object:
        ctx = cs.StubContext({Stage.TRACK: track}, sampling, stage_params=stage_params)
        return image_rest(ctx, case.states).spec  # type: ignore[arg-type]

    assert wide.spec(row) != SavgolParams().spec(row)
    assert spec_of({Stage.KINEMATICS: wide}) == wide.spec(row)
    assert spec_of({Stage.KINEMATICS: TruthParams()}) == SavgolParams().spec(row)
    assert spec_of({}) == SavgolParams().spec(row)
