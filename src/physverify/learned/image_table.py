"""The pair image table a learned variant reads: one ``pair_row`` per unordered object pair of a
clip, keyed as the variant looks it up (``image_key``: clip, mask and depth providers, pair).

The masks and the depth of a row come from the same plan, so they share its sampling; the frames
are the clip's video decoded at that sampling. A table names the crop encoder (sha256) and the
crop settings its tokens were computed with.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np

from physverify.checks.variants._learned import image_key
from physverify.core.errors import PhysverifyError
from physverify.core.feature_table import FeatureTable
from physverify.datasets._base import VideoSource
from physverify.extraction.contact_crops import CropSettings, FloatArray
from physverify.extraction.crop_encoder import CropEncoder
from physverify.extraction.pair_features import PairFrames, dense_columns, pair_row, token_columns
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import DepthMap, MaskTrack
from physverify.schema.stages import Stage

ENCODER_KEY = "encoder_sha256"
CROP_KEY = "crop_settings"


class ImageTableError(PhysverifyError):
    """Masks and depth that cannot be read together, or rows of mixed widths."""


@dataclass(frozen=True)
class ClipImages:
    """What one clip's rows are read from: its masks, depth and frames under one plan."""

    clip_id: str
    provenance: Mapping[str, str]
    track: MaskTrack
    depth: DepthMap
    frames: np.ndarray


def same_sampling(a: FrameSampling, b: FrameSampling) -> bool:
    return a.size == b.size and np.array_equal(a.frame_indices, b.frame_indices)


def clip_images(
    clip_id: str, names: Mapping[Stage, str], track: MaskTrack, depth: DepthMap, video: VideoSource
) -> ClipImages:
    """``names``: the plan's provider per stage, as ``CheckInputs.provenance`` records them."""
    if not same_sampling(track.sampling, depth.sampling):
        raise ImageTableError(f"{clip_id}: masks and depth were sampled differently")
    s = track.sampling
    frames = video.frames(s.start, s.stop, s.step, s.max_side)
    if frames.shape[:3] != track.masks.shape[:1] + track.masks.shape[2:]:
        raise ImageTableError(f"{clip_id}: frames {frames.shape} do not match the masks")
    provenance = {stage.value: name for stage, name in names.items()}
    return ClipImages(clip_id, provenance, track, depth, frames)


def object_pairs(track: MaskTrack) -> list[tuple[int, int]]:
    order = sorted(range(len(track.object_ids)), key=lambda i: str(track.object_ids[i]))
    return list(itertools.combinations(order, 2))


def pair_frames(images: ClipImages, a: int, b: int) -> PairFrames:
    masks = images.track.masks
    depth = np.asarray(images.depth.depth, dtype=np.float64)
    return PairFrames(images.frames, masks[:, a], masks[:, b], depth, images.depth.valid)


def clip_rows(
    images: ClipImages, encoder: CropEncoder, settings: CropSettings
) -> dict[str, FloatArray]:
    ids = images.track.object_ids
    return {
        image_key(images.clip_id, images.provenance, str(ids[a]), str(ids[b])): pair_row(
            pair_frames(images, a, b), encoder, settings
        )
        for a, b in object_pairs(images.track)
    }


def image_table(
    rows: Iterable[Mapping[str, FloatArray]], encoder: CropEncoder, settings: CropSettings
) -> FeatureTable:
    """Every clip's rows in ``dense_columns()`` then ``token_columns(width)`` order."""
    merged = {key: row for clip in rows for key, row in clip.items()}
    n_dense = len(dense_columns())
    widths = {len(row) - n_dense for row in merged.values()}
    if len(widths) > 1:
        raise ImageTableError(f"rows carry token widths {sorted(widths)}")
    width = widths.pop() if widths else 0
    meta = {ENCODER_KEY: encoder.sha256, CROP_KEY: json.dumps(asdict(settings), sort_keys=True)}
    columns = (*dense_columns(), *token_columns(width))
    keys = tuple(sorted(merged))
    values = np.asarray([merged[k] for k in keys], dtype=np.float64).reshape(
        len(keys), len(columns)
    )
    return FeatureTable(keys, columns, values, meta)
