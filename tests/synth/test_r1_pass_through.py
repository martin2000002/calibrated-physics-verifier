"""r1_pass_through (ADR 0047 section 1) on the committed f5 clips."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from physverify.schema.states import ObjectStateTable
from physverify.synth.frame_windows import rest_mask
from physverify.synth.perturb import PerturbationNotApplicableError, SeedKey, perturb_trajectory
from physverify.synth.perturbations._penetration import PEN_LEVELS
from physverify.synth.perturbations.r1_pass_through import (
    absent_partner,
    first_rolling_hit,
    overlap_frames,
)
from physverify.synth.placement import UnreachableTargetError

FAMILY = "f5_contact"
SEED = 2046980616


@pytest.fixture
def plugin(plugins: dict[str, Any]) -> Any:
    return plugins["r1_pass_through"]


def clip_of(bank: Any) -> Any:
    """The Bullet clip: the analytic twin rolls with friction and stops short of the cube."""
    return bank.recorded(FAMILY)


def applied(plugin: Any, clip: Any, magnitude: float, seed: int = 7) -> Any:
    return plugin.apply(
        clip.scene, clip.states, clip.contacts, magnitude, np.random.default_rng(seed)
    )


def test_each_level_is_reached_by_bisection(plugin: Any, bank: Any) -> None:
    clip = clip_of(bank)
    for level, target in enumerate(PEN_LEVELS, start=1):
        key = SeedKey(SEED, FAMILY, 0, level)
        states, contacts, record = perturb_trajectory(
            plugin, clip.scene, clip.states, clip.contacts, target, key
        )
        assert record.level == level
        assert record.target_checks == ("R1_dyn",)
        assert abs(record.magnitude - target) <= 1e-4 * max(1.0, target), (level, record.magnitude)
        assert plugin.measure(states, contacts, record) == record.magnitude
        assert states.position.shape == clip.states.position.shape


def test_the_mover_rolls_on_and_the_rest_stand_still(plugin: Any, bank: Any) -> None:
    clip = clip_of(bank)
    mover, partner, start = first_rolling_hit(clip.states, clip.contacts)
    states, _, record = applied(plugin, clip, PEN_LEVELS[2])
    offset = np.asarray(record.params["offset_m"])
    assert record.params["free_from_frame"] == start
    assert abs(float(offset[2])) < 1e-12
    np.testing.assert_allclose(
        states.position[: start + 1, mover], clip.states.position[: start + 1, mover] + offset
    )
    steps = np.diff(states.position[start:, mover], axis=0)
    np.testing.assert_allclose(steps, np.broadcast_to(steps[0], steps.shape), atol=1e-12)
    for j in range(states.n_objects):
        if j != mover:
            assert bool(np.all(rest_mask(states, j)[start:]))
            np.testing.assert_array_equal(
                states.position[start:, j],
                np.broadcast_to(clip.states.position[start, j], (states.n_frames - start, 3)),
            )
    assert len(overlap_frames(states, mover, partner)) > 0


def test_the_sign_is_one_draw_and_the_edit_is_deterministic(plugin: Any, bank: Any) -> None:
    clip = clip_of(bank)
    rng = np.random.default_rng(11)
    plugin.apply(clip.scene, clip.states, clip.contacts, PEN_LEVELS[1], rng)
    reference = np.random.default_rng(11)
    reference.uniform()
    assert rng.uniform() == reference.uniform()
    first, second = (
        applied(plugin, clip, PEN_LEVELS[1], 11),
        applied(plugin, clip, PEN_LEVELS[1], 11),
    )
    assert first[2] == second[2]
    assert np.array_equal(first[0].position, second[0].position, equal_nan=True)
    offsets = {
        float(np.sign(applied(plugin, clip, PEN_LEVELS[1], seed)[2].params["offset_m"][0]))
        for seed in range(8)
    }
    assert offsets == {-1.0, 1.0}


def test_an_unreachable_target_is_refused(plugin: Any, bank: Any) -> None:
    with pytest.raises(UnreachableTargetError):
        applied(plugin, clip_of(bank), 50.0)


def test_a_third_body_on_the_free_path_is_refused(plugin: Any, bank: Any) -> None:
    clip = clip_of(bank)
    mover, partner, start = first_rolling_hit(clip.states, clip.contacts)
    free = absent_partner(clip.states, mover, start)
    others = [j for j in range(clip.states.n_objects) if j not in (mover, partner)]
    third = max(others, key=lambda j: float(clip.states.size[0, j, 0]))
    position = np.array(clip.states.position, copy=True)
    position[:, third, :2] = free.position[-1, mover, :2]
    blocked = replace(clip, states=_with_position(clip.states, position))
    with pytest.raises(PerturbationNotApplicableError, match="third body"):
        applied(plugin, blocked, PEN_LEVELS[3])


def test_a_pair_out_of_the_image_is_refused(plugin: Any, bank: Any) -> None:
    clip = clip_of(bank)
    camera = clip.scene.camera
    away = tuple(2.0 * np.asarray(camera.position) - np.asarray(camera.look_at))
    scene = replace(clip.scene, camera=replace(camera, look_at=away))
    with pytest.raises(PerturbationNotApplicableError, match="leaves the image"):
        applied(plugin, replace(clip, scene=scene), PEN_LEVELS[0])


def test_a_clip_without_a_rolling_hit_is_refused(plugin: Any, bank: Any) -> None:
    for clip in (bank.recorded("f4_stack"), bank.analytic(FAMILY)):
        with pytest.raises(PerturbationNotApplicableError, match="no rolling sphere"):
            applied(plugin, clip, PEN_LEVELS[0])


def _with_position(states: ObjectStateTable, position: np.ndarray) -> ObjectStateTable:
    return replace(states, position=position)


def test_only_the_contact_family_is_perturbed(plugin: Any, bank: Any) -> None:
    for family in ("f1_projectile", "f2_bounce", "f3_collision", "f4_stack"):
        clip = bank.recorded(family)
        with pytest.raises(PerturbationNotApplicableError, match="not applicable"):
            perturb_trajectory(
                plugin, clip.scene, clip.states, clip.contacts, 0.1, SeedKey(SEED, family, 0, 1)
            )
