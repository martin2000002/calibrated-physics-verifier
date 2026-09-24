"""Open-vocabulary boxes from the clip's object queries with Grounding DINO tiny (ADR 0005, 3.7).

The model runs on the first ``search_frames`` sampled frames; the frame whose detections cover
the most queries seeds the tracker. torch and transformers are imported inside the functions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.core.errors import MissingInputError
from physverify.extraction._base import StageContext, WeightKind, WeightSpec
from physverify.extraction.matching import box_iou_matrix
from physverify.extraction.providers._common import (
    LazyModel,
    ProviderParams,
    expect_params,
    set_torch_threads,
)
from physverify.extraction.weights import hub_offline, local_path
from physverify.schema.clip import FrameSampling, ObjectQuery
from physverify.schema.perception import Detections
from physverify.schema.stages import Stage

FloatArray = NDArray[np.float64]

WEIGHTS = WeightSpec(
    WeightKind.HF,
    "IDEA-Research/grounding-dino-tiny",
    "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3",
)


class Params(ProviderParams):
    box_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    nms_iou: float = Field(default=0.5, gt=0.0, le=1.0)
    search_frames: int = Field(default=1, ge=1)
    max_objects: int = Field(default=10, ge=1)


@dataclass(frozen=True)
class RawDetections:
    """One frame of model output: xyxy boxes, scores and the matched text spans."""

    boxes: FloatArray
    scores: FloatArray
    texts: tuple[str, ...]


@dataclass(frozen=True)
class Selection:
    """Kept detections: query index per box, in the order they become object ids."""

    query: tuple[int, ...]
    boxes: FloatArray
    scores: FloatArray

    @property
    def coverage(self) -> tuple[int, float]:
        """Queries covered, then total score (the frame-choice criterion)."""
        return len(set(self.query)), float(self.scores.sum())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def assign_query(text: str, queries: Sequence[ObjectQuery]) -> int | None:
    """Index of the query sharing most tokens with a returned span (first on ties)."""
    words = tokens(text)
    overlaps = [len(words & tokens(query.phrase)) for query in queries]
    best = int(np.argmax(overlaps)) if overlaps else 0
    return best if overlaps and overlaps[best] > 0 else None


def non_max_suppression(boxes: FloatArray, scores: FloatArray, iou: float) -> list[int]:
    """Class-agnostic greedy NMS; kept indices by decreasing score."""
    overlap = box_iou_matrix(boxes, boxes)
    kept: list[int] = []
    for index in np.argsort(-scores, kind="stable").tolist():
        if all(overlap[index, other] <= iou for other in kept):
            kept.append(index)
    return kept


def select(raw: RawDetections, queries: Sequence[ObjectQuery], params: Params) -> Selection:
    """NMS, query assignment, top ``expected_count`` per query, at most ``max_objects``."""
    per_query: dict[int, list[int]] = {}
    for index in non_max_suppression(raw.boxes, raw.scores, params.nms_iou):
        query = assign_query(raw.texts[index], queries)
        if query is not None:
            per_query.setdefault(query, []).append(index)
    chosen = [
        (query, index)
        for query, indices in sorted(per_query.items())
        for index in indices[: queries[query].expected_count or len(indices)]
    ]
    chosen = sorted(chosen, key=lambda pair: -raw.scores[pair[1]])[: params.max_objects]
    chosen.sort()
    rows = [index for _, index in chosen]
    return Selection(
        query=tuple(query for query, _ in chosen),
        boxes=raw.boxes[rows].reshape(-1, 4),
        scores=raw.scores[rows],
    )


def slug(phrase: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", phrase.lower())) or "object"


def to_detections(
    sampling: FrameSampling, row: int, selection: Selection, queries: Sequence[ObjectQuery]
) -> Detections:
    counts: dict[int, int] = {}
    ids: list[str] = []
    for query in selection.query:
        ids.append(f"{slug(queries[query].phrase)}_{counts.get(query, 0)}")
        counts[query] = counts.get(query, 0) + 1
    return Detections(
        sampling=sampling,
        frame_index=int(sampling.frame_indices[row]),
        boxes_xyxy=selection.boxes.astype(np.float64),
        scores=selection.scores.astype(np.float64),
        labels=np.array([queries[q].phrase for q in selection.query], dtype=np.str_),
        object_ids=np.array(ids, dtype=np.str_),
    )


def best_frame(selections: Sequence[Selection]) -> int:
    """Earliest frame with the best (queries covered, total score)."""
    scores = [selection.coverage for selection in selections]
    return scores.index(max(scores))


def prompt(queries: Sequence[ObjectQuery]) -> str:
    """Grounding DINO's text format: lowercase phrases, each ended by a period."""
    return " ".join(f"{query.phrase.strip().lower()}." for query in queries)


class GroundingDinoDetector:
    stage: ClassVar[Stage] = Stage.DETECT
    name: ClassVar[str] = "grounding_dino_tiny"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = ()
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = (WEIGHTS,)
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)
        self._model: LazyModel[tuple[Any, Any]] = LazyModel(load_detector)

    def run(self, ctx: StageContext) -> Detections:
        queries = ctx.clip.object_queries
        if not queries:
            raise MissingInputError(f"{ctx.clip.clip_id}: no object queries to detect")
        s = ctx.sampling
        count = min(self.params.search_frames, s.n_frames)
        stop = int(s.frame_indices[count - 1]) + 1
        frames = ctx.video.frames(s.start, stop, s.step, s.max_side)
        set_torch_threads(ctx.threads)
        model, processor = self._model.get(ctx)
        raws = [
            detect(model, processor, frame, queries, self.params, ctx.device) for frame in frames
        ]
        selections = [select(raw, queries, self.params) for raw in raws]
        row = best_frame(selections)
        return to_detections(s, row, selections[row], queries)


def load_detector(ctx: StageContext) -> tuple[Any, Any]:
    import transformers

    snapshot = local_path(WEIGHTS, ctx.paths)
    with hub_offline():
        model: Any = transformers.GroundingDinoForObjectDetection.from_pretrained(snapshot)
        processor = transformers.AutoProcessor.from_pretrained(snapshot)
    return model.to(ctx.device).eval(), processor


def detect(
    model: Any,
    processor: Any,
    frame: NDArray[np.uint8],
    queries: Sequence[ObjectQuery],
    params: Params,
    device: str,
) -> RawDetections:
    import torch

    inputs = processor(images=frame, text=prompt(queries), return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=params.box_threshold,
        text_threshold=params.text_threshold,
        target_sizes=[frame.shape[:2]],
    )[0]
    return RawDetections(
        boxes=result["boxes"].float().cpu().numpy().astype(np.float64).reshape(-1, 4),
        scores=result["scores"].float().cpu().numpy().astype(np.float64),
        texts=tuple(str(text) for text in result["text_labels"]),
    )


PLUGINS = (GroundingDinoDetector,)
