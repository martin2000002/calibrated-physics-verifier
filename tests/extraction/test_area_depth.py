"""Depth from apparent area: the area fixes how depth changes along a track, the track keeps its
median depth, and rows or tracks without a trusted area keep the lift's values bit for bit."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray

from physverify.extraction.area_depth import (
    SCALED_FIELDS,
    area_depth_factor,
    area_depth_rows,
    mask_area,
    scale_depth_fields,
)
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable, PrimitiveKind

MIN_ROWS = 3
T = 9
Z_TRUE = np.linspace(2.0, 6.0, T)
AREA_AT_ONE_METER = 4.0e4
RAYS = np.array([[0.3, -0.2, 1.0], [-0.1, 0.4, 1.0], [0.2, 0.2, 1.0]])
ANGULAR_SIZE = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.3], [0.05, 0.05, 0.05]])


def measured_z() -> NDArray[np.float64]:
    """Depth anticorrelated with the truth plus a wobble, as a relative depth model can give."""
    return 9.0 - Z_TRUE + 0.3 * np.sin(np.arange(T) * 1.7)


def table_of(z: NDArray[np.float64]) -> ObjectStateTable:
    n = z.shape[1]
    table = ObjectStateTable.blank(
        clip_id="clip",
        frame=CoordFrame.CAMERA_OPENCV,
        unit=LengthUnit.RELATIVE,
        source_key="lift-key",
        t=np.arange(T, dtype=np.float64),
        frame_index=np.arange(T, dtype=np.int64),
        object_ids=[f"o{i}" for i in range(n)],
        labels=["ball"] * n,
        primitive=[PrimitiveKind.SPHERE.value] * n,
    )
    position = RAYS[None, :n] * z[..., None]
    size = ANGULAR_SIZE[None, :n] * z[..., None]
    return dataclasses.replace(
        table,
        position=position,
        position_sigma=0.1 * np.abs(position) + 0.01,
        size=size,
        size_sigma=0.2 * size,
        nearer_cover=np.zeros((T, n)),
        border_cut=np.zeros((T, n), dtype=np.bool_),
    )


def lifted(table: ObjectStateTable, area: NDArray[np.float64]) -> ObjectStateTable:
    used = area_depth_rows(table, area)
    factor = area_depth_factor(table.position[..., 2], area, used, MIN_ROWS)
    return scale_depth_fields(table, factor)


@pytest.fixture
def receding() -> tuple[ObjectStateTable, NDArray[np.float64]]:
    """One rigid sphere moving away from 2 to 6 m: its area falls as 1/z_true^2."""
    z = measured_z()[:, None]
    area = (AREA_AT_ONE_METER / Z_TRUE**2)[:, None]
    return table_of(z), area


def test_area_depth_has_slope_one_against_true_depth(
    receding: tuple[ObjectStateTable, NDArray[np.float64]],
) -> None:
    table, area = receding
    z_area = lifted(table, area).position[:, 0, 2]
    slope = np.polyfit(np.log(Z_TRUE), np.log(z_area), 1)[0]
    assert abs(slope - 1.0) < 1e-9
    assert np.median(z_area) == pytest.approx(np.median(table.position[:, 0, 2]), abs=1e-12)


def test_the_measured_depth_alone_has_the_wrong_slope(
    receding: tuple[ObjectStateTable, NDArray[np.float64]],
) -> None:
    table, _ = receding
    slope = np.polyfit(np.log(Z_TRUE), np.log(table.position[:, 0, 2]), 1)[0]
    assert slope < 0


def test_untrusted_rows_keep_the_lift_values_bit_for_bit() -> None:
    z = np.repeat(measured_z()[:, None], 2, axis=1)
    area = np.repeat((AREA_AT_ONE_METER / Z_TRUE**2)[:, None], 2, axis=1)
    table = table_of(z)
    cover = table.nearer_cover.copy()  # type: ignore[union-attr]
    cut = table.border_cut.copy()  # type: ignore[union-attr]
    visible = table.visible.copy()
    position = table.position.copy()
    cover[0, 0] = 0.4
    cut[1, 0] = True
    area[2, 0] = 0.0
    position[3, 0, 2] = np.nan
    position[4, 0, 2] = -1.0
    visible[5, 0] = False
    cover[6, 0] = np.nan
    table = dataclasses.replace(
        table, nearer_cover=cover, border_cut=cut, visible=visible, position=position
    )
    out = lifted(table, area)
    for name in SCALED_FIELDS:
        np.testing.assert_array_equal(getattr(out, name)[:6, 0], getattr(table, name)[:6, 0])
    assert not np.array_equal(out.position[6:, 0], table.position[6:, 0])
    assert not np.array_equal(out.position[:, 1], table.position[:, 1])
    np.testing.assert_array_equal(out.nearer_cover, table.nearer_cover)
    np.testing.assert_array_equal(out.border_cut, table.border_cut)


def test_nan_cover_counts_as_uncovered() -> None:
    table = table_of(measured_z()[:, None])
    cover = np.full((T, 1), np.nan)
    area = (AREA_AT_ONE_METER / Z_TRUE**2)[:, None]
    assert area_depth_rows(dataclasses.replace(table, nearer_cover=cover), area).all()


def test_a_track_with_too_few_trusted_rows_is_left_whole() -> None:
    table = table_of(measured_z()[:, None])
    area = np.zeros((T, 1))
    area[: MIN_ROWS - 1, 0] = 100.0
    out = lifted(table, area)
    for name in SCALED_FIELDS:
        np.testing.assert_array_equal(getattr(out, name), getattr(table, name))


def test_scaling_keeps_the_ray_and_the_angular_size(
    receding: tuple[ObjectStateTable, NDArray[np.float64]],
) -> None:
    table, area = receding
    out = lifted(table, area)
    before, after = table.position[:, 0], out.position[:, 0]
    cosine = (before * after).sum(axis=-1) / (
        np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1)
    )
    np.testing.assert_allclose(cosine, 1.0, atol=1e-12)
    np.testing.assert_allclose(
        out.size[:, 0] / after[:, 2:3], table.size[:, 0] / before[:, 2:3], rtol=1e-12
    )


def test_sigmas_scale_by_the_same_factor_as_the_position(
    receding: tuple[ObjectStateTable, NDArray[np.float64]],
) -> None:
    table, area = receding
    out = lifted(table, area)
    factor = (out.position[..., 2] / table.position[..., 2])[..., None]
    for name in ("position_sigma", "size_sigma"):
        np.testing.assert_allclose(getattr(out, name), getattr(table, name) * factor, rtol=1e-12)
        assert (getattr(out, name) >= getattr(table, name) * factor * (1 - 1e-12)).all()


def test_mask_area_counts_pixels_per_row_and_track() -> None:
    masks = np.zeros((2, 3, 4, 5), dtype=np.bool_)
    masks[0, 1, :2, :3] = True
    masks[1, 2, 3, 4] = True
    np.testing.assert_array_equal(mask_area(masks), [[0, 6, 0], [0, 0, 1]])
