"""The one pinhole FOV <-> focal conversion."""

from __future__ import annotations

import pytest

from physverify.schema.perception import focal_from_fov, horizontal_fov_deg


@pytest.mark.parametrize(("focal", "width", "fov"), [(0.5, 1.0, 90.0), (320.0, 640.0, 90.0)])
def test_known_fields_of_view(focal: float, width: float, fov: float) -> None:
    assert horizontal_fov_deg(focal, width) == pytest.approx(fov)


@pytest.mark.parametrize("fov", [20.0, 49.13, 60.0, 120.0])
def test_the_two_conversions_invert_each_other(fov: float) -> None:
    assert horizontal_fov_deg(focal_from_fov(fov, 512.0), 512.0) == pytest.approx(fov)
