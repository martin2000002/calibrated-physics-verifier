"""The box-prompted tracker maps the model's object ids back to detection columns."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from physverify.extraction.providers._sam_video import track_boxes
from physverify.schema.perception import Detections

torch = pytest.importorskip("torch")

N_FRAMES, N_OBJECTS, HEIGHT, WIDTH = 4, 3, 5, 6
ROW = 1


class ConsumingProcessor:
    """Keeps and later empties the caller's ``obj_ids`` list, as transformers does."""

    def init_video_session(self, video: list[Any], inference_device: str, dtype: Any) -> Any:
        return SimpleNamespace(video_height=HEIGHT, video_width=WIDTH)

    def add_inputs_to_inference_session(
        self, inference_session: Any, frame_idx: int, obj_ids: list[int], input_boxes: Any
    ) -> None:
        inference_session.pending = obj_ids
        inference_session.ids = list(obj_ids)

    def post_process_masks(
        self, masks: list[Any], original_sizes: Any, mask_threshold: float, binarize: bool
    ) -> list[Any]:
        return [masks[0] > mask_threshold]


class ReversedIdModel:
    """Object ``i`` lights column ``i`` of every frame; ids come back in reverse order."""

    def __call__(self, inference_session: Any, frame_idx: int) -> None:
        inference_session.pending.clear()

    def propagate_in_video_iterator(self, session: Any, start_frame_idx: int, reverse: bool) -> Any:
        frames = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, N_FRAMES)
        ids = session.ids[::-1]
        pred = torch.zeros(len(ids), 1, HEIGHT, WIDTH)
        for k, obj_id in enumerate(ids):
            pred[k, 0, :, obj_id] = 1.0
        for frame in frames:
            logits = torch.full((len(ids), 1), 5.0)
            yield SimpleNamespace(
                frame_idx=frame, object_ids=ids, pred_masks=pred, object_score_logits=logits
            )


def test_every_frame_and_object_lands_in_its_own_column() -> None:
    detections = SimpleNamespace(
        object_ids=np.array([f"o{i}" for i in range(N_OBJECTS)]),
        boxes_xyxy=np.zeros((N_OBJECTS, 4)),
    )
    frames = np.zeros((N_FRAMES, HEIGHT, WIDTH, 3), np.uint8)
    masks, scores = track_boxes(
        ReversedIdModel(),
        ConsumingProcessor(),
        frames,
        cast(Detections, detections),
        ROW,
        "cpu",
        0.0,
    )
    assert masks.shape == (N_FRAMES, N_OBJECTS, HEIGHT, WIDTH)
    for column in range(N_OBJECTS):
        lit = masks[:, column].any(axis=1)
        assert lit[:, column + 1].all() and lit.sum() == N_FRAMES
    assert scores == pytest.approx(np.full((N_FRAMES, N_OBJECTS), 1 / (1 + np.exp(-5.0))))
