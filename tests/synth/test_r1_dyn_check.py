"""R1_dyn on the committed f5 Bullet clip: clean versus r1_pass_through, ground-truth inputs."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.row_sigma import contact_rows, row_residuals
from physverify.checks._base import Check, CheckInputs
from physverify.checks.separate.r1_dyn_pass_through import image_track
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest, SamplingSpec, Split, TimeSource, resolve_sampling
from physverify.schema.matching import IdMap, MatchMethod
from physverify.schema.perception import CameraModel, IntrinsicsSource
from physverify.schema.stages import STAGE_ORDER
from physverify.synth.gt.camera import camera_states
from physverify.synth.perturb import SeedKey, perturb_trajectory
from physverify.synth.perturbations._penetration import PEN_LEVELS
from physverify.synth.placement import spec_intrinsics, spec_world_from_camera

FAMILY = "f5_contact"
PAIR = ("cube_0", "ball_0")
K_FRAMES = 3
# A pass-through keeps at least 0.8 of its speed; the clean hit loses nearly all of it.
NO_IMPULSE = 0.8


@pytest.fixture
def check() -> Check:
    cls = PLUGIN_KINDS["separate_checks"]()["R1_dyn"]
    instance: Check = cls(cls.Params())
    return instance


def oracle_inputs(clip: Any, *, with_camera: bool = True, depth_scale: float = 1.0) -> CheckInputs:
    scene = clip.scene
    pose = spec_world_from_camera(scene.camera)
    states = camera_states(clip.states, pose)
    states = dataclasses.replace(
        states, position=states.position * depth_scale, size=states.size * depth_scale
    )
    width, height = scene.resolution
    manifest = ClipManifest(
        dataset="kubric_synthetic",
        clip_id=states.clip_id,
        split=Split.CALIB,
        video_path=f"kubric_synthetic/{states.clip_id}",
        n_frames=states.n_frames,
        width=width,
        height=height,
        container_fps=float(scene.frame_rate),
        seconds_per_frame=1.0 / scene.frame_rate,
        time_source=TimeSource.GROUND_TRUTH,
        gt_stages=STAGE_ORDER,
    )
    camera = CameraModel(
        sampling=resolve_sampling(manifest, SamplingSpec()),
        K=spec_intrinsics(scene.camera, scene.resolution),
        source=IntrinsicsSource.GROUND_TRUTH,
        fov_deg=60.0,
        fov_sigma_deg=0.0,
    )
    return CheckInputs(
        clip=manifest,
        states=states,
        contacts=None,
        scale=None,
        input_level=InputLevel.ORACLE,
        provenance={},
        scene=scene,
        camera=camera if with_camera else None,
    )


def pair_result(results: Sequence[CheckResult], pair: tuple[str, str] = PAIR) -> CheckResult:
    return next(r for r in results if set(r.subject) == set(pair))


def passed_through(plugins: dict[str, Any], clip: Any, level: int) -> Any:
    states, contacts, _ = perturb_trajectory(
        plugins["r1_pass_through"],
        clip.scene,
        clip.states,
        clip.contacts,
        PEN_LEVELS[level - 1],
        SeedKey(2046980616, FAMILY, 0, level),
    )
    return dataclasses.replace(clip, states=states, contacts=contacts)


def test_the_clean_hit_scores_low(check: Check, bank: Any) -> None:
    result = pair_result(check.evaluate(oracle_inputs(bank.recorded(FAMILY))))
    assert result.verdict is Verdict.UNSCORED
    assert result.magnitude is not None and result.magnitude < K_FRAMES * (1.0 - NO_IMPULSE)
    assert result.sigma == 0.0


@pytest.mark.parametrize("level", [2, 3, 4])
def test_a_pass_through_keeps_its_speed(
    check: Check, bank: Any, plugins: dict[str, Any], level: int
) -> None:
    clip = passed_through(plugins, bank.recorded(FAMILY), level)
    result = pair_result(check.evaluate(oracle_inputs(clip)))
    assert result.verdict is Verdict.UNSCORED, result.abstain_reason
    assert result.details["mover"] == "ball_0"
    assert result.details["matched_frames"] >= K_FRAMES
    assert result.details["impulse"] < 1.0 - NO_IMPULSE
    assert result.magnitude is not None and result.magnitude >= K_FRAMES * NO_IMPULSE
    assert result.sigma == 0.0


def test_the_magnitude_does_not_depend_on_the_depth_scale(
    check: Check, bank: Any, plugins: dict[str, Any]
) -> None:
    clip = passed_through(plugins, bank.recorded(FAMILY), 3)
    base = pair_result(check.evaluate(oracle_inputs(clip)))
    scaled = pair_result(check.evaluate(oracle_inputs(clip, depth_scale=2.5)))
    assert base.magnitude is not None and scaled.magnitude is not None
    assert scaled.magnitude == pytest.approx(base.magnitude, abs=1e-9)
    assert scaled.details["impulse"] == pytest.approx(base.details["impulse"], abs=1e-9)


def test_pairs_that_never_meet_score_zero(check: Check, bank: Any) -> None:
    results = check.evaluate(oracle_inputs(bank.recorded(FAMILY)))
    assert len(results) == 6
    assert pair_result(results, ("cube_1", "ball_0")).magnitude == 0.0


def test_no_camera_abstains(check: Check, bank: Any) -> None:
    results = check.evaluate(oracle_inputs(bank.recorded(FAMILY), with_camera=False))
    assert {r.abstain_reason for r in results} == {AbstainReason.MISSING_INPUT}


def test_an_unknown_sigma_abstains(check: Check, bank: Any, plugins: dict[str, Any]) -> None:
    inputs = oracle_inputs(passed_through(plugins, bank.recorded(FAMILY), 3))
    unknown = np.full_like(inputs.states.position_sigma, np.nan)
    states = dataclasses.replace(inputs.states, position_sigma=unknown)
    result = pair_result(check.evaluate(dataclasses.replace(inputs, states=states)))
    assert result.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY


def test_the_row_is_kept_out_of_the_nine_checks() -> None:
    assert "R1_dyn" not in PLUGIN_KINDS["checks"]()
    assert list(PLUGIN_KINDS["separate_checks"]()) == ["R1_dyn"]


def calibrated(sigma: float, size_sigma: float | None = None, **lags: tuple[float, ...]) -> Check:
    cls = PLUGIN_KINDS["separate_checks"]()["R1_dyn"]
    params = {"row_sigma": sigma, "size_sigma": size_sigma, **lags}
    instance: Check = cls(cls.Params.model_validate(params))
    return instance


def without_sigma(inputs: CheckInputs) -> CheckInputs:
    unknown = np.full_like(inputs.states.position_sigma, np.nan)
    return dataclasses.replace(
        inputs, states=dataclasses.replace(inputs.states, position_sigma=unknown)
    )


def lag_value(lags: Sequence[float], lag: int) -> float:
    return 1.0 if lag == 0 else (lags[lag - 1] if lag <= len(lags) else 0.0)


def hand_statistic(
    inputs: CheckInputs,
    first_frame: int,
    row: float,
    size: float,
    row_lags: Sequence[float] = (),
    size_lags: Sequence[float] = (),
) -> float:
    """max_b q_b computed by hand: per body the two least-squares rates of D around the first
    matched row and q = Delta' Sigma^-1 Delta, Sigma = sum_t sum_t' c_t c_t' C_tt' over the rows
    of both windows (c = (-w_before, w_after)), C_tt' = R_r(|t - t'|) diag(row^2 / (s_t s_t'),
    row^2 / (s_t s_t'), 0) + R_s(|t - t'|) size^2 D_t D_t'^T, R the lags (1 at 0, 0 beyond)."""
    states = inputs.states
    k = np.asarray(inputs.camera.K) if inputs.camera is not None else np.eye(3)
    ids = [str(x) for x in states.object_ids]
    tracks = [image_track(k, states, ids.index(body)) for body in PAIR]
    usable = tracks[0].usable & tracks[1].usable
    start = int(np.flatnonzero(states.frame_index == first_frame)[0])
    windows = [np.flatnonzero(usable[:start]), start + np.flatnonzero(usable[start:])]
    out = []
    for track in tracks:
        d = np.column_stack(
            [
                (track.contact[:, 0] - k[0, 2]) / track.size,
                (track.contact[:, 1] - k[1, 2]) / track.size,
                k[0, 0] / track.size,
            ]
        )
        rates, terms = [], []
        for sign, rows in zip((-1.0, 1.0), windows, strict=True):
            rates.append(np.polyfit(states.t[rows], d[rows], 1)[0])
            centred = states.t[rows] - states.t[rows].mean()
            terms += [
                (int(r), sign * w) for w, r in zip(centred / (centred @ centred), rows, strict=True)
            ]
        sigma = np.zeros((3, 3))
        for r_i, c_i in terms:
            for r_j, c_j in terms:
                lag = abs(r_i - r_j)
                per_pair = lag_value(size_lags, lag) * size**2 * np.outer(d[r_i], d[r_j])
                scale = lag_value(row_lags, lag) * row**2 / (track.size[r_i] * track.size[r_j])
                per_pair[:2, :2] += scale * np.eye(2)
                sigma += c_i * c_j * per_pair
        change = rates[1] - rates[0]
        out.append(float(change @ np.linalg.inv(sigma) @ change))
    return max(out)


def test_a_calibrated_row_sigma_replaces_the_unknown_one(
    bank: Any, plugins: dict[str, Any]
) -> None:
    inputs = without_sigma(oracle_inputs(passed_through(plugins, bank.recorded(FAMILY), 3)))
    check = calibrated(1.5, 0.05, row_autocorrelation=(0.6, 0.2), size_autocorrelation=(0.4,))
    result = pair_result(check.evaluate(inputs))
    assert result.verdict is Verdict.UNSCORED, result.abstain_reason
    assert result.magnitude is not None and result.magnitude >= K_FRAMES * (1.0 - NO_IMPULSE)
    assert result.sigma is not None and 0.0 < result.sigma < float("inf")
    assert result.details["impulse_statistic"] >= 0.0


@pytest.mark.parametrize("level", [0, 3])
def test_the_impulse_statistic_is_q_of_both_windows(
    bank: Any, plugins: dict[str, Any], level: int
) -> None:
    clip = bank.recorded(FAMILY)
    inputs = without_sigma(oracle_inputs(passed_through(plugins, clip, level) if level else clip))
    result = pair_result(calibrated(1.5, 0.05).evaluate(inputs))
    assert result.verdict is Verdict.UNSCORED, result.abstain_reason
    expected = hand_statistic(inputs, result.details["first_matched_frame"], 1.5, 0.05)
    assert result.details["impulse_statistic"] == pytest.approx(expected, rel=1e-9)


ROW_LAGS = (0.6, 0.3, 0.1)
SIZE_LAGS = (0.5,)


@pytest.mark.parametrize("level", [0, 3])
def test_the_impulse_statistic_reads_each_autocorrelation_at_its_own_lags(
    bank: Any, plugins: dict[str, Any], level: int
) -> None:
    clip = bank.recorded(FAMILY)
    inputs = without_sigma(oracle_inputs(passed_through(plugins, clip, level) if level else clip))
    check = calibrated(1.5, 0.05, row_autocorrelation=ROW_LAGS, size_autocorrelation=SIZE_LAGS)
    result = pair_result(check.evaluate(inputs))
    assert result.verdict is Verdict.UNSCORED, result.abstain_reason
    first = result.details["first_matched_frame"]
    expected = hand_statistic(inputs, first, 1.5, 0.05, ROW_LAGS, SIZE_LAGS)
    swapped = hand_statistic(inputs, first, 1.5, 0.05, SIZE_LAGS, ROW_LAGS)
    assert result.details["impulse_statistic"] == pytest.approx(expected, rel=1e-9)
    assert swapped != pytest.approx(expected, rel=1e-6)


def test_a_separate_rate_sigma_moves_q_but_not_the_match(
    bank: Any, plugins: dict[str, Any]
) -> None:
    inputs = without_sigma(oracle_inputs(passed_through(plugins, bank.recorded(FAMILY), 3)))
    cls = PLUGIN_KINDS["separate_checks"]()["R1_dyn"]
    params = {"row_sigma": 1.5, "rate_row_sigma": 0.75, "size_sigma": 0.05}
    split = pair_result(cls(cls.Params.model_validate(params)).evaluate(inputs))
    same = pair_result(calibrated(1.5, 0.05).evaluate(inputs))
    assert split.details["matched_frames"] == same.details["matched_frames"]
    assert split.details["impulse_statistic"] > same.details["impulse_statistic"]


def test_without_overlapping_rows_sigma_is_exactly_zero(bank: Any) -> None:
    inputs = without_sigma(oracle_inputs(bank.recorded(FAMILY)))
    results = calibrated(1.5, 0.05).evaluate(inputs)
    apart = [r for r in results if r.details.get("overlap_rows") == 0]
    assert apart and all(r.magnitude == 0.0 and r.sigma == 0.0 for r in apart)


def test_a_row_sigma_without_a_size_sigma_abstains(bank: Any, plugins: dict[str, Any]) -> None:
    inputs = oracle_inputs(passed_through(plugins, bank.recorded(FAMILY), 3))
    result = pair_result(calibrated(1.5).evaluate(inputs))
    assert result.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY


def test_the_default_row_sigma_keeps_the_delta_method(check: Check, bank: Any) -> None:
    inputs = oracle_inputs(bank.recorded(FAMILY))
    cls = PLUGIN_KINDS["separate_checks"]()["R1_dyn"]
    assert cls.Params().row_sigma is None
    explicit = cls(cls.Params.model_validate({"row_sigma": None}))
    assert repr(explicit.evaluate(inputs)) == repr(check.evaluate(inputs))
    assert repr(calibrated(0.0).evaluate(inputs)) == repr(check.evaluate(inputs))


def test_matching_rows_have_no_residual(bank: Any) -> None:
    truth = oracle_inputs(bank.recorded(FAMILY))
    rows = contact_rows(truth)
    assert set(rows) == set(truth.states.object_ids)
    residuals = row_residuals(truth, truth, None)
    assert residuals.size == sum(len(r) for r in rows.values()) > 0
    assert not residuals.any()


def test_residuals_follow_the_id_map(bank: Any) -> None:
    truth = oracle_inputs(bank.recorded(FAMILY))
    states = truth.states
    renamed = np.asarray([f"p{i}" for i in range(len(states.object_ids))])
    lowered = states.position.copy()
    lowered[..., 1] += 0.05
    perceived = dataclasses.replace(
        truth, states=dataclasses.replace(states, object_ids=renamed, position=lowered)
    )
    ids = IdMap(
        pairs=tuple((str(p), str(g), 1.0) for p, g in zip(renamed, states.object_ids, strict=True)),
        unmatched_pred=(),
        unmatched_gt=(),
        method=MatchMethod.MASK_IOU,
        frames_used=(),
        min_score=0.5,
    )
    residuals = row_residuals(perceived, truth, ids)
    assert residuals.size == sum(len(r) for r in contact_rows(truth).values()) > 0
    assert (residuals > 0.0).all()
    assert row_residuals(perceived, truth, None).size == 0


def test_residuals_use_the_frames_both_read(bank: Any) -> None:
    truth = oracle_inputs(bank.recorded(FAMILY))
    offset = truth.states.n_frames // 2
    shifted = dataclasses.replace(truth.states, frame_index=truth.states.frame_index + offset)
    perceived = dataclasses.replace(truth, states=shifted)
    seen, true = contact_rows(perceived), contact_rows(truth)
    shared = sum(len(seen[o].keys() & true[o].keys()) for o in true)
    assert 0 < row_residuals(perceived, truth, None).size == shared < sum(map(len, true.values()))
