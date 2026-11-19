"""Heavy DETECT/TRACK/DEPTH stand-ins built from the synthetic fixture's truth.

Ids are renamed and emitted in reverse truth order, masks are dilated by one pixel and depth is
an affine disparity ``a / z + b``. Every instance and call is written to ``JOURNAL``.
"""

from __future__ import annotations

import functools
import weakref
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.ndimage import binary_dilation, generate_binary_structure
from synthetic_clip import build_clip

from physverify.config.models import RuntimeSettings
from physverify.datasets._base import ArrayGroundTruthStore
from physverify.extraction._base import ProviderParams, StageContext, StageProvider, WeightSpec
from physverify.extraction.providers._common import expect_params
from physverify.schema.perception import DepthKind, DepthMap, Detections, MaskTrack
from physverify.schema.stages import Stage

FAKE = "fake"
FAKE_IDS = {"ball": "obj_1", "box": "obj_0"}
"""Truth id -> perceived id."""
TRUTH_IDS = {fake: truth for truth, fake in FAKE_IDS.items()}
DISPARITY_A = 2.0
DILATION = np.zeros((1, 1, 3, 3), dtype=bool)
DILATION[0, 0] = generate_binary_structure(2, 1)
"""One-pixel 4-connected dilation in the image plane only."""


@dataclass
class Journal:
    """Live fake instances (weak) and the ordered log of calls, locks and stage starts."""

    live: weakref.WeakSet[object] = field(default_factory=weakref.WeakSet)
    calls: list[str] = field(default_factory=list)
    max_live: int = 0

    def created(self, provider: object, stage: Stage) -> None:
        self.live.add(provider)
        self.max_live = max(self.max_live, len(self.live))
        self.calls.append(f"init:{stage.value}")

    def reset(self) -> None:
        self.live = weakref.WeakSet()
        self.calls.clear()
        self.max_live = 0


JOURNAL = Journal()


@functools.cache
def fixture_truth() -> ArrayGroundTruthStore:
    return build_clip(RuntimeSettings()).ground_truth


class FakeParams(ProviderParams):
    fail_clip: str = ""


class FakeDepthParams(FakeParams):
    a: float = DISPARITY_A
    b: float = 0.0


class _FakeHeavy:
    name: ClassVar[str] = FAKE
    version: ClassVar[str] = "1"
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    stage: ClassVar[Stage]
    Params: ClassVar[type[BaseModel]] = FakeParams

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, FakeParams)
        JOURNAL.created(self, self.stage)

    def _start(self, ctx: StageContext) -> None:
        JOURNAL.calls.append(f"run:{self.stage.value}:{ctx.clip.clip_id}")
        if ctx.clip.clip_id == self.params.fail_clip:
            raise ValueError(f"fake {self.stage.value} fails on {ctx.clip.clip_id}")


def _perceived_order(object_ids: NDArray[np.str_]) -> NDArray[np.int64]:
    return np.arange(object_ids.shape[0], dtype=np.int64)[::-1]


class FakeDetect(_FakeHeavy):
    stage: ClassVar[Stage] = Stage.DETECT
    requires: ClassVar[tuple[Stage, ...]] = ()

    def run(self, ctx: StageContext) -> Detections:
        self._start(ctx)
        truth = fixture_truth().detections(ctx.sampling)
        order = _perceived_order(truth.object_ids)
        return Detections(
            sampling=ctx.sampling,
            frame_index=truth.frame_index,
            boxes_xyxy=truth.boxes_xyxy[order],
            scores=truth.scores[order],
            labels=truth.labels[order],
            object_ids=np.array([FAKE_IDS[str(i)] for i in truth.object_ids[order]]),
        )


class FakeTrack(_FakeHeavy):
    stage: ClassVar[Stage] = Stage.TRACK
    requires: ClassVar[tuple[Stage, ...]] = (Stage.DETECT,)

    def run(self, ctx: StageContext) -> MaskTrack:
        self._start(ctx)
        detections = ctx.require(Stage.DETECT, Detections)
        truth = fixture_truth().masks(ctx.sampling)
        truth_ids = truth.object_ids.tolist()
        rows = [truth_ids.index(TRUTH_IDS[str(i)]) for i in detections.object_ids.tolist()]
        return MaskTrack(
            sampling=ctx.sampling,
            object_ids=detections.object_ids.copy(),
            labels=detections.labels.copy(),
            masks=binary_dilation(truth.masks[:, rows], structure=DILATION),
            scores=truth.scores[:, rows],
        )


class FakeDepth(_FakeHeavy):
    stage: ClassVar[Stage] = Stage.DEPTH
    requires: ClassVar[tuple[Stage, ...]] = ()
    Params: ClassVar[type[BaseModel]] = FakeDepthParams

    def run(self, ctx: StageContext) -> DepthMap:
        self._start(ctx)
        params = expect_params(self.params, FakeDepthParams)
        truth = fixture_truth().depth(ctx.sampling)
        return affine_disparity(truth, params.a, params.b)


def affine_disparity(metric: DepthMap, a: float, b: float) -> DepthMap:
    """``a / z + b`` on valid pixels of a metric depth map."""
    z = metric.depth.astype(np.float64)
    valid = metric.valid & (z > 0)
    disparity = np.where(valid, a / np.where(valid, z, 1.0) + b, 0.0)
    return DepthMap(
        sampling=metric.sampling,
        depth=disparity.astype(np.float32),
        kind=DepthKind.AFFINE_DISPARITY,
        valid=valid,
    )


FAKES: tuple[type[StageProvider], ...] = (FakeDetect, FakeTrack, FakeDepth)
