"""Ground-truth masks relabelled with the ids of the upstream detections."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from physverify.extraction._base import GROUND_TRUTH, StageContext, WeightSpec
from physverify.extraction.matching import DEFAULT_MIN_IOU, gt_detections_at, match_boxes
from physverify.extraction.providers._common import ProviderParams, expect_params
from physverify.schema.perception import Detections, MaskTrack
from physverify.schema.stages import Stage


class Params(ProviderParams):
    min_iou: float = Field(default=DEFAULT_MIN_IOU, gt=0.0, le=1.0)


class GroundTruthTrack:
    stage: ClassVar[Stage] = Stage.TRACK
    name: ClassVar[str] = GROUND_TRUTH
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.DETECT,)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = True
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> MaskTrack:
        detections = ctx.require(Stage.DETECT, Detections)
        gt = ctx.require_gt()
        truth = gt.masks(ctx.sampling)
        at_detection = gt_detections_at(gt, ctx.sampling, detections.frame_index)
        gt_of = match_boxes(detections, at_detection, self.params.min_iou).pred_to_gt()
        truth_ids = truth.object_ids.tolist()
        n_t, _, h, w = truth.masks.shape
        k = detections.object_ids.shape[0]
        masks = np.zeros((n_t, k, h, w), dtype=bool)
        scores = np.zeros((n_t, k))
        for i, object_id in enumerate(detections.object_ids.tolist()):
            if object_id in gt_of:
                j = truth_ids.index(gt_of[object_id])
                masks[:, i] = truth.masks[:, j]
                scores[:, i] = truth.scores[:, j]
        return MaskTrack(
            sampling=ctx.sampling,
            object_ids=detections.object_ids,
            labels=detections.labels,
            masks=masks,
            scores=scores,
        )


PLUGINS = (GroundTruthTrack,)
