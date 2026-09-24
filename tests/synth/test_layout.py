"""Sizes, placement, camera framing and projection used by the scene families."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.config.models import SizePrior
from physverify.schema.states import PrimitiveKind
from physverify.synth.layout import (
    CameraFraming,
    UnreachableTargetError,
    camera_axes,
    extents_for,
    footprint_clear,
    footprint_radius,
    frame_camera,
    lognormal_params,
    mass_for,
    place_on_floor,
    project,
    solve_offset,
    spec_intrinsics,
    spec_world_from_camera,
)


def test_lognormal_matches_the_prior_moments() -> None:
    prior = SizePrior(mean_m=0.2, rel_spread=0.3)
    mu, sigma = lognormal_params(prior)
    assert math.exp(mu + sigma**2 / 2.0) == pytest.approx(0.2)
    assert math.sqrt(math.exp(sigma**2) - 1.0) == pytest.approx(0.3)


@pytest.mark.parametrize("kind", list(PrimitiveKind))
def test_extents_keep_the_characteristic_length(kind: PrimitiveKind) -> None:
    extents = extents_for(kind, 0.1, np.random.default_rng(0), (0.5, 2.0))
    assert math.prod(extents) ** (1.0 / 3.0) == pytest.approx(0.1)


def test_mass_uses_the_primitive_volume() -> None:
    assert mass_for(PrimitiveKind.SPHERE, (0.1, 0.1, 0.1), 1000.0) == pytest.approx(
        1000.0 * math.pi * 0.1**3 / 6.0
    )
    assert mass_for(PrimitiveKind.BOX, (0.1, 0.2, 0.3), 10.0) == pytest.approx(0.06)
    assert footprint_radius(PrimitiveKind.BOX, (0.3, 0.4, 1.0)) == pytest.approx(0.25)


def test_placement_keeps_clearance() -> None:
    rng = np.random.default_rng(1)
    placed: list[tuple[np.ndarray, float]] = []
    for _ in range(5):
        xy = place_on_floor(rng, 0.05, placed, 0.5, 0.02)
        assert xy is not None
        placed.append((xy, 0.05))
    for n, (xy, r) in enumerate(placed):
        assert footprint_clear(xy, r, placed[:n] + placed[n + 1 :], 0.02)
    assert place_on_floor(rng, 1.0, placed, 0.01, 0.0, tries=5) is None


def test_camera_axes_are_a_right_handed_rotation() -> None:
    rotation = camera_axes((2.0, -1.0, 1.5), (0.0, 0.0, 0.2))
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation[2, 1] < 0.0


def test_framed_envelope_fits_inside_the_image() -> None:
    envelope = np.array([[-0.3, -0.2, 0.0], [0.4, 0.3, 0.6], [0.1, -0.1, 0.3]])
    framing = CameraFraming(fill=0.8)
    camera = frame_camera(np.random.default_rng(2), envelope, (256, 192), framing)
    K = spec_intrinsics(camera, (256, 192))  # noqa: N806
    uv, z = project(envelope, K, spec_world_from_camera(camera))
    assert np.all(z > 0.0)
    half = np.array([128.0, 96.0])
    assert np.max(np.abs(uv - half) / half) <= 0.8 + 1e-3
    look_uv, _ = project(camera.look_at, K, spec_world_from_camera(camera))
    assert look_uv[0] == pytest.approx(half, abs=1e-9)


def test_solve_offset_brackets_or_refuses() -> None:
    assert solve_offset(lambda s: s**2, 2.0, 0.0, 3.0) == pytest.approx(math.sqrt(2.0))
    with pytest.raises(UnreachableTargetError):
        solve_offset(lambda s: s**2, -1.0, 0.0, 3.0)
