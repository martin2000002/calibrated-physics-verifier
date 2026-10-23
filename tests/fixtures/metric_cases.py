"""Builders shared by the diagnosis metric tests: the options, a native sampling, a case of the
analytic clip, and two disjoint squares over a sloped depth."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.diagnosis.measurement_cases import MetricCase
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import FrameSampling
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import DepthKind, DepthMap, MaskTrack
from physverify.schema.stages import Stage

METRIC_OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.125,
)


def native_sampling(frames: int, width: int, height: int) -> FrameSampling:
    """``frames`` frames at 24 fps, sampled at the native ``width`` x ``height``."""
    return FrameSampling(
        frame_indices=np.arange(frames, dtype=np.int64),
        width=width,
        height=height,
        native_width=width,
        native_height=height,
        seconds_per_frame=1.0 / 24.0,
        step=1,
        max_side=None,
    )


def metric_case(
    pred: StageOutput,
    truth: StageOutput,
    options: MetricSettings = METRIC_OPTIONS,
    context: Mapping[Stage, StageOutput] | None = None,
) -> MetricCase:
    """A case of the analytic clip without ground truth or plan."""
    return MetricCase(
        clip=manifest(),
        pred=pred,
        truth=truth,
        context=MappingProxyType(dict(context or {})),
        gt=None,
        options=options,
        plan=None,
    )


def depth_on(
    values: NDArray[np.float64],
    sampling: FrameSampling,
    kind: DepthKind = DepthKind.METRIC_Z,
    valid: NDArray[np.bool_] | None = None,
) -> DepthMap:
    """``values`` as float32 depth, valid everywhere unless ``valid`` says otherwise."""
    return DepthMap(
        sampling=sampling,
        depth=values.astype(np.float32),
        kind=kind,
        valid=np.ones(values.shape, dtype=bool) if valid is None else valid,
    )


def two_squares(frames: int, size: int) -> NDArray[np.bool_]:
    """(T, 2, h, w): "ball" a 7x7 square, "box" an 8x8 square, disjoint."""
    masks = np.zeros((frames, 2, size, size), dtype=bool)
    masks[:, 0, 2:9, 2:9] = True
    masks[:, 1, 10:18, 10:18] = True
    return masks


def squares_track(frames: int, size: int) -> MaskTrack:
    return MaskTrack(
        sampling=native_sampling(frames, size, size),
        object_ids=np.array(["ball", "box"]),
        labels=np.array(["ball", "box"]),
        masks=two_squares(frames, size),
        scores=np.ones((frames, 2)),
    )


def sloped_z(frames: int, size: int) -> NDArray[np.float64]:
    """A depth plane sloped along both image axes that recedes with time."""
    t, rows, cols = np.mgrid[:frames, :size, :size]
    z: NDArray[np.float64] = 2.0 + 0.1 * cols + 0.05 * rows + 0.3 * t
    return z
