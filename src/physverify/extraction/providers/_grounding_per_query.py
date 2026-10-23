"""Grounding DINO with one prompt per query and a slot assignment under the expected counts.

A joint prompt lets one returned span ("cube cylinder") claim several objects for the first
query and ``expected_count`` then drops the rest (ADR 0014 §1). Here each query is prompted on
its own, boxes found by several prompts are merged, and boxes are assigned to query slots by a
maximum-score matching. ``prompt_mode = "joint"`` keeps the original selection exactly.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.matching import box_iou_matrix
from physverify.extraction.providers import detect_grounding_dino_tiny as base
from physverify.extraction.providers._common import expect_params
from physverify.schema.clip import ObjectQuery

FloatArray = NDArray[np.float64]


class PromptedParams(base.Params):
    prompt_mode: Literal["joint", "per_query"] = "per_query"
    merge_iou: float = Field(default=0.7, gt=0.0, le=1.0)


@dataclass(frozen=True)
class Candidates:
    """Distinct boxes (K, 4) xyxy and their score under each query (K, Q); 0 when not found."""

    boxes: FloatArray
    scores: FloatArray


def merge_candidates(found: Sequence[base.RawDetections], merge_iou: float) -> Candidates:
    """Greedy merge by decreasing score: a box overlapping a kept one at ``merge_iou`` or more
    adds its score to that box's column; otherwise it becomes a new box."""
    entries = [
        (float(score), q, raw.boxes[i])
        for q, raw in enumerate(found)
        for i, score in enumerate(raw.scores.tolist())
    ]
    entries.sort(key=lambda entry: -entry[0])
    boxes: list[FloatArray] = []
    rows: list[FloatArray] = []
    for score, q, box in entries:
        overlap = box_iou_matrix(box[None, :], np.array(boxes).reshape(-1, 4))[0]
        hit = int(np.argmax(overlap)) if overlap.size and overlap.max() >= merge_iou else None
        if hit is None:
            boxes.append(box)
            rows.append(np.zeros(len(found)))
            hit = len(rows) - 1
        rows[hit][q] = max(rows[hit][q], score)
    return Candidates(
        boxes=np.array(boxes, dtype=np.float64).reshape(-1, 4),
        scores=np.array(rows, dtype=np.float64).reshape(-1, len(found)),
    )


def per_query_candidates(
    model: Any,
    processor: Any,
    frame: NDArray[np.uint8],
    queries: Sequence[ObjectQuery],
    params: PromptedParams,
    device: str,
) -> Candidates:
    found = [base.detect(model, processor, frame, [query], params, device) for query in queries]
    return merge_candidates(found, params.merge_iou)


def query_slots(queries: Sequence[ObjectQuery], n_boxes: int) -> list[int]:
    """Each query repeated ``expected_count`` times; an unknown count gets one slot per box."""
    return [q for q, query in enumerate(queries) for _ in range(query.expected_count or n_boxes)]


def assign(
    candidates: Candidates, queries: Sequence[ObjectQuery], params: PromptedParams
) -> base.Selection:
    """NMS on each box's best score, then the slot matching of maximum total score; boxes left
    without a slot or matched at score 0 are dropped, then at most ``max_objects`` by score."""
    if candidates.boxes.shape[0] == 0:
        return base.Selection(query=(), boxes=np.zeros((0, 4)), scores=np.zeros(0))
    kept = base.non_max_suppression(candidates.boxes, candidates.scores.max(axis=1), params.nms_iou)
    slots = query_slots(queries, len(kept))
    gain = candidates.scores[np.ix_(kept, slots)]
    rows, cols = linear_sum_assignment(-gain)
    chosen = [
        (slots[c], kept[r], float(gain[r, c]))
        for r, c in zip(rows, cols, strict=True)
        if gain[r, c] > 0
    ]
    chosen = sorted(chosen, key=lambda pick: -pick[2])[: params.max_objects]
    chosen.sort(key=lambda pick: (pick[0], -pick[2]))
    return base.Selection(
        query=tuple(q for q, _, _ in chosen),
        boxes=candidates.boxes[[i for _, i, _ in chosen]].reshape(-1, 4),
        scores=np.array([s for _, _, s in chosen], dtype=np.float64),
    )


class PromptedDetector(base.GroundingDinoDetector):
    """Grounding DINO with a choice of prompt mode; subclasses fix the model weights."""

    spec: ClassVar[WeightSpec] = base.WEIGHTS
    Params: ClassVar[type[BaseModel]] = PromptedParams

    def __init__(self, params: BaseModel) -> None:
        self.prompted = expect_params(params, PromptedParams)
        super().__init__(params)

    def loader(self) -> Callable[[StageContext], tuple[Any, Any]]:
        return functools.partial(base.load_detector, spec=self.spec)

    def select_frame(
        self,
        model: Any,
        processor: Any,
        frame: NDArray[np.uint8],
        queries: Sequence[ObjectQuery],
        device: str,
    ) -> base.Selection:
        if self.prompted.prompt_mode == "joint":
            return super().select_frame(model, processor, frame, queries, device)
        candidates = per_query_candidates(model, processor, frame, queries, self.prompted, device)
        return assign(candidates, queries, self.prompted)
