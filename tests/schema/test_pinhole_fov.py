"""The one pinhole FOV <-> focal conversion."""

from __future__ import annotations

import math

import pytest

from physverify.schema.perception import (
    focal_from_fov,
    focal_rel_sigma,
    fov_sigma_from_phi,
    horizontal_fov_deg,
)


@pytest.mark.parametrize(("focal", "width", "fov"), [(0.5, 1.0, 90.0), (320.0, 640.0, 90.0)])
def test_known_fields_of_view(focal: float, width: float, fov: float) -> None:
    assert horizontal_fov_deg(focal, width) == pytest.approx(fov)


@pytest.mark.parametrize("fov", [20.0, 49.13, 60.0, 120.0])
def test_the_two_conversions_invert_each_other(fov: float) -> None:
    assert horizontal_fov_deg(focal_from_fov(fov, 512.0), 512.0) == pytest.approx(fov)


@pytest.mark.parametrize("fov", [20.0, 49.13, 90.0, 120.0])
def test_the_focal_sigma_is_the_derivative_of_the_focal(fov: float) -> None:
    step = 1e-4
    slope = (focal_from_fov(fov - step, 1.0) - focal_from_fov(fov + step, 1.0)) / (2 * step)
    assert focal_rel_sigma(fov, 1.0) == pytest.approx(slope / focal_from_fov(fov, 1.0), rel=1e-6)


@pytest.mark.parametrize("fov", [20.0, 49.13, 120.0])
def test_the_focal_and_fov_sigmas_invert_each_other(fov: float) -> None:
    assert fov_sigma_from_phi(fov, focal_rel_sigma(fov, 2.5)) == pytest.approx(2.5)
    assert math.isnan(fov_sigma_from_phi(fov, math.nan))
