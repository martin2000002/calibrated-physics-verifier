"""Box-prompted video segmentation shared by the SAM 2.1 and EdgeTAM trackers.

torch and transformers are imported inside the functions (registry discovery stays light).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from physverify.extraction._base import ProviderParams, StageContext, WeightSpec
from physverify.extraction.providers._common import LazyModel
from physverify.extraction.weights import hub_offline, local_path
from physverify.schema.perception import Detections, MaskTrack
from physverify.schema.stages import Stage

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


class VideoTrackParams(ProviderParams):
    mask_threshold: float = Field(default=0.0)


@dataclass(frozen=True)
class VideoModel:
    """A transformers video segmentation class name and its pinned checkpoint."""

    class_name: str
    weights: WeightSpec


def video_model_loader(video_model: VideoModel) -> LazyModel[tuple[Any, Any]]:
    """The tracker's (model, processor), loaded once per provider instance."""
    return LazyModel(lambda ctx: load_video_model(video_model, ctx))


def run_video_tracker(
    ctx: StageContext, loaded: LazyModel[tuple[Any, Any]], mask_threshold: float
) -> MaskTrack:
    detections = ctx.require(Stage.DETECT, Detections)
    s = ctx.sampling
    frames = ctx.video.frames(s.start, s.stop, s.step, s.max_side)
    row = int(np.flatnonzero(s.frame_indices == detections.frame_index)[0])
    model, processor = loaded.get(ctx)
    masks, scores = track_boxes(
        model, processor, frames, detections, row, ctx.device, mask_threshold
    )
    return MaskTrack(
        sampling=s,
        object_ids=detections.object_ids.copy(),
        labels=detections.labels.copy(),
        masks=masks,
        scores=scores,
    )


def load_video_model(video_model: VideoModel, ctx: StageContext) -> tuple[Any, Any]:
    import transformers

    snapshot = local_path(video_model.weights, ctx.paths)
    with hub_offline():
        model_class = getattr(transformers, video_model.class_name)
        model = model_class.from_pretrained(snapshot).to(ctx.device).eval()
        processor = transformers.Sam2VideoProcessor.from_pretrained(snapshot)
    return model, processor


def track_boxes(
    model: Any,
    processor: Any,
    frames: NDArray[np.uint8],
    detections: Detections,
    row: int,
    device: str,
    mask_threshold: float,
) -> tuple[BoolArray, FloatArray]:
    """Masks (T, N, h, w) and presence scores (T, N) propagated from the boxes at ``row``."""
    import torch

    n_frames, height, width = frames.shape[:3]
    n_objects = int(detections.object_ids.shape[0])
    masks = np.zeros((n_frames, n_objects, height, width), dtype=bool)
    scores = np.zeros((n_frames, n_objects))
    if n_objects == 0:
        return masks, scores
    obj_ids = tuple(range(1, n_objects + 1))
    with torch.inference_mode():
        session = processor.init_video_session(
            video=list(frames), inference_device=device, dtype=torch.float32
        )
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=row,
            obj_ids=list(obj_ids),  # transformers consumes this list in place
            input_boxes=[detections.boxes_xyxy.tolist()],
        )
        model(inference_session=session, frame_idx=row)
        for reverse in (False, True) if row > 0 else (False,):
            for output in model.propagate_in_video_iterator(
                session, start_frame_idx=row, reverse=reverse
            ):
                frame_masks = processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    mask_threshold=mask_threshold,
                    binarize=True,
                )[0]
                _store(output, frame_masks, obj_ids, masks, scores)
    return masks, scores


def _store(
    output: Any, frame_masks: Any, obj_ids: tuple[int, ...], masks: BoolArray, scores: FloatArray
) -> None:
    import torch

    frame = int(output.frame_idx)
    ids = obj_ids if output.object_ids is None else output.object_ids
    columns = [obj_ids.index(int(i)) for i in ids]
    binary = frame_masks[:, 0].cpu().numpy().astype(bool)
    masks[frame, columns] = binary
    logits = getattr(output, "object_score_logits", None)
    presence = None
    if logits is not None:
        presence = torch.sigmoid(logits.float()).reshape(-1).cpu().numpy()
    if presence is not None and presence.shape[0] == len(columns):
        scores[frame, columns] = presence
    else:
        scores[frame, columns] = binary.reshape(len(columns), -1).any(axis=1)
