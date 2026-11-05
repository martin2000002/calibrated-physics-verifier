"""R1_dyn on the committed f5 Bullet clip: clean versus r1_pass_through, ground-truth inputs."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks._base import Check, CheckInputs
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest, SamplingSpec, Split, TimeSource, resolve_sampling
from physverify.schema.perception import CameraModel, IntrinsicsSource
from physverify.schema.stages import STAGE_ORDER
from physverify.synth.gt.camera import camera_states
from physverify.synth.perturb import SeedKey, perturb_trajectory
from physverify.synth.perturbations._penetration import PEN_LEVELS
from physverify.synth.placement import spec_intrinsics, spec_world_from_camera

FAMILY = "f5_contact"
PAIR = ("cube_0", "ball_0")
K_FRAMES = 3
# A pass-through keeps at least 0.8 of its speed; the clean hit is a full impulse.
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
    assert result.magnitude is not None and result.magnitude < 1.0 - NO_IMPULSE


@pytest.mark.parametrize("level", [2, 3, 4])
def test_a_pass_through_keeps_its_speed(
    check: Check, bank: Any, plugins: dict[str, Any], level: int
) -> None:
    clip = passed_through(plugins, bank.recorded(FAMILY), level)
    result = pair_result(check.evaluate(oracle_inputs(clip)))
    assert result.verdict is Verdict.UNSCORED, result.abstain_reason
    assert result.details["mover"] == "ball_0"
    assert result.details["matched_frames"] >= K_FRAMES
    assert result.magnitude is not None and result.magnitude >= NO_IMPULSE
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
