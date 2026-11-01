"""ADR 0033: the along-ray sigma of a perceived sphere from its apparent-size constancy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.extraction.lift_params import SizeConstancyLiftParams, size_constancy_params
from physverify.extraction.size_constancy import (
    SizeConstancyParams,
    along_ray_sigma,
    apparent_diameter,
    replace_along_ray_sigma,
)

PARAMS = size_constancy_params(SizeConstancyLiftParams())
SIDE = 64
DIAMETERS = (20, 24, 30, 40)
DEPTHS = (6.0, 5.0, 4.0, 3.0)


def square_masks(diameters: tuple[int, ...], *, offset: int = 8) -> np.ndarray:
    """(T, 1, H, W) square masks of the given sides, away from the border."""
    masks = np.zeros((len(diameters), 1, SIDE, SIDE), dtype=np.bool_)
    for t, d in enumerate(diameters):
        masks[t, 0, offset : offset + d, offset : offset + d] = True
    return masks


def pixel_sigma(d: float, params: SizeConstancyParams = PARAMS) -> float:
    return math.hypot(params.diameter_noise_px / d, params.shape_sigma_floor)


def sphere_sigma(
    masks: np.ndarray, depth: tuple[float, ...], *, visible: np.ndarray | None = None
) -> np.ndarray:
    z = np.asarray(depth, dtype=np.float64)[:, None]
    seen = np.ones(z.shape, dtype=np.bool_) if visible is None else visible
    return along_ray_sigma(masks, z, seen, np.array([True]), PARAMS)


def test_the_constants_are_the_a_priori_ones() -> None:
    a_priori = SizeConstancyParams(
        min_diameter_px=12.0,
        diameter_noise_px=math.sqrt(2.0),
        shape_sigma_floor=0.05,
        min_clean_rows=3,
        border_margin_px=1,
    )
    assert size_constancy_params(SizeConstancyLiftParams()) == a_priori


def test_the_diameter_is_the_longer_box_side_and_nan_when_empty() -> None:
    masks = np.zeros((2, 1, 20, 20), dtype=np.bool_)
    masks[0, 0, 3:8, 4:13] = True
    np.testing.assert_array_equal(apparent_diameter(masks), [[9.0], [np.nan]])


def test_a_sphere_whose_size_follows_its_depth_gets_the_pixel_sigma() -> None:
    sigma = sphere_sigma(square_masks(DIAMETERS), DEPTHS)
    expected = [[pixel_sigma(d)] for d in DIAMETERS]
    np.testing.assert_allclose(sigma, expected, rtol=1e-12)


def test_a_row_whose_depth_disagrees_with_its_size_gets_the_log_ratio() -> None:
    depth = (6.0, 5.0 * 1.2, 4.0, 3.0)
    sigma = sphere_sigma(square_masks(DIAMETERS), depth)
    assert sigma[1, 0] == pytest.approx(math.hypot(math.log(1.2), pixel_sigma(24)), rel=1e-12)
    assert sigma[0, 0] == pytest.approx(pixel_sigma(20), rel=1e-12)


def test_the_sigma_never_falls_below_the_shape_floor() -> None:
    sigma = sphere_sigma(square_masks((48, 48, 48), offset=4), (2.0, 2.0, 2.0))
    assert bool((sigma >= PARAMS.shape_sigma_floor).all())
    assert float(sigma[0, 0]) == pytest.approx(pixel_sigma(48), rel=1e-12)


def test_only_sphere_tracks_get_a_sigma() -> None:
    masks = np.concatenate([square_masks(DIAMETERS)] * 2, axis=1)
    z = np.repeat(np.asarray(DEPTHS)[:, None], 2, axis=1)
    sigma = along_ray_sigma(masks, z, np.ones(z.shape, np.bool_), np.array([True, False]), PARAMS)
    assert bool(np.isfinite(sigma[:, 0]).all())
    assert bool(np.isnan(sigma[:, 1]).all())


def test_hidden_rows_are_left_out() -> None:
    visible = np.array([[True], [False], [True], [True]])
    sigma = sphere_sigma(square_masks(DIAMETERS), DEPTHS, visible=visible)
    assert math.isnan(sigma[1, 0])
    assert bool(np.isfinite(sigma[[0, 2, 3], 0]).all())


def test_rows_cut_by_the_frame_are_left_out() -> None:
    masks = square_masks(DIAMETERS)
    masks[2, 0, 0, 10] = True
    sigma = sphere_sigma(masks, DEPTHS)
    assert math.isnan(sigma[2, 0])
    assert bool(np.isfinite(sigma[[0, 1, 3], 0]).all())


def test_rows_narrower_than_the_minimum_are_left_out() -> None:
    sigma = sphere_sigma(square_masks((11, 24, 30, 40)), DEPTHS)
    assert math.isnan(sigma[0, 0])
    assert bool(np.isfinite(sigma[1:, 0]).all())


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0])
def test_rows_without_a_positive_finite_depth_are_left_out(bad: float) -> None:
    sigma = sphere_sigma(square_masks(DIAMETERS), (6.0, bad, 4.0, 3.0))
    assert math.isnan(sigma[1, 0])
    assert bool(np.isfinite(sigma[[0, 2, 3], 0]).all())


def test_an_object_with_too_few_clean_rows_gets_no_sigma() -> None:
    visible = np.array([[True], [True], [False], [False]])
    sigma = sphere_sigma(square_masks(DIAMETERS), DEPTHS, visible=visible)
    assert bool(np.isnan(sigma).all())


def test_the_along_ray_part_is_swapped_and_the_lateral_part_kept() -> None:
    position = np.array([[[1.0, -2.0, 4.0]]])
    old = np.array([[[0.3, 0.4, 0.5]]])
    sigma = replace_along_ray_sigma(position, old, np.array([[0.1]]))
    relative = 0.5 / 4.0
    expected_x = math.sqrt(0.3**2 - (relative * 1.0) ** 2 + (0.1 * 1.0) ** 2)
    expected_y = math.sqrt(max(0.4**2 - (relative * 2.0) ** 2, 0.0) + (0.1 * 2.0) ** 2)
    np.testing.assert_allclose(sigma[0, 0], [expected_x, expected_y, 0.4], rtol=1e-12)
    np.testing.assert_array_equal(old, [[[0.3, 0.4, 0.5]]])


@pytest.mark.parametrize(
    ("position", "old", "along"),
    [
        ([1.0, 2.0, 4.0], [0.3, 0.4, 0.5], math.nan),
        ([1.0, 2.0, math.nan], [0.3, 0.4, 0.5], 0.1),
        ([1.0, 2.0, -4.0], [0.3, 0.4, 0.5], 0.1),
        ([1.0, 2.0, 4.0], [math.nan, math.nan, 0.5], 0.1),
    ],
)
def test_cells_without_a_usable_input_are_returned_unchanged(
    position: list[float], old: list[float], along: float
) -> None:
    sigma = replace_along_ray_sigma(np.array([[position]]), np.array([[old]]), np.array([[along]]))
    np.testing.assert_array_equal(sigma, [[old]])
