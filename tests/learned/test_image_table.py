from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from physverify.app.learned import read_images
from physverify.checks.variants._learned import image_key
from physverify.diagnosis.chain_runner import VariantOutputs
from physverify.diagnosis.chain_variants import Variant
from physverify.extraction.contact_crops import CropSettings
from physverify.extraction.crop_encoder import CropEncoder
from physverify.extraction.pair_features import PairFrames, dense_columns, pair_row, token_columns
from physverify.learned.arms import ARMS, arm_groups, arm_name, arm_variant, drawn_arm
from physverify.learned.image_table import (
    CROP_KEY,
    ENCODER_KEY,
    ClipImages,
    ImageTableError,
    clip_images,
    clip_rows,
    image_table,
    object_pairs,
)
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import DepthKind, DepthMap, MaskTrack
from physverify.schema.stages import Stage

T, H, W = 3, 24, 32
NAMES = {Stage.TRACK: "sam2", Stage.DEPTH: "ground_truth"}
ENCODER = CropEncoder(lambda batch: batch.mean(axis=(2, 3)), "sha")
SETTINGS = CropSettings(size_px=16)


def sampling(width: int = W, frames: tuple[int, ...] = (0, 2, 4)) -> FrameSampling:
    indices = np.asarray(frames, dtype=np.int64)
    return FrameSampling(indices, width, H, width, H, 1 / 24, 2, None)


def track(ids: tuple[str, ...] = ("c", "a", "b")) -> MaskTrack:
    masks = np.zeros((T, len(ids), H, W), dtype=bool)
    for i in range(len(ids)):
        masks[:, i, 8:16, 6 * i + 2 : 6 * i + 9] = True
    labels = np.asarray(["box"] * len(ids))
    return MaskTrack(sampling(), np.asarray(ids), labels, masks, np.ones((T, len(ids))))


def depth(width: int = W) -> DepthMap:
    values = np.linspace(1.0, 3.0, T * H * width, dtype=np.float32).reshape(T, H, width)
    valid = np.ones((T, H, width), dtype=bool)
    return DepthMap(sampling(width), values, DepthKind.METRIC_Z, valid)


@dataclass
class Video:
    calls: list[tuple[int, int, int, int | None]] = field(default_factory=list)
    frames_shape: tuple[int, ...] = (T, H, W, 3)

    def frames(self, start: int, stop: int, step: int = 1, max_side: int | None = None) -> Any:
        self.calls.append((start, stop, step, max_side))
        return np.arange(np.prod(self.frames_shape), dtype=np.uint8).reshape(self.frames_shape)


def images() -> ClipImages:
    return clip_images("clip-1", NAMES, track(), depth(), Video())  # type: ignore[arg-type]


def test_inputs_are_read_at_the_plan_sampling_under_its_provider_names() -> None:
    video = Video()
    found = clip_images("clip-1", NAMES, track(), depth(), video)  # type: ignore[arg-type]
    assert video.calls == [(0, 5, 2, None)]
    assert found.provenance == {"track": "sam2", "depth": "ground_truth"}


def test_masks_and_depth_of_another_sampling_or_frames_are_refused() -> None:
    with pytest.raises(ImageTableError, match="sampled differently"):
        clip_images("c", NAMES, track(), depth(W + 2), Video())  # type: ignore[arg-type]
    short = Video(frames_shape=(T - 1, H, W, 3))
    with pytest.raises(ImageTableError, match="do not match"):
        clip_images("c", NAMES, track(), depth(), short)  # type: ignore[arg-type]


def test_every_unordered_pair_gets_the_row_and_key_the_variant_looks_up() -> None:
    found = images()
    rows = clip_rows(found, ENCODER, SETTINGS)
    assert len(object_pairs(found.track)) == 3
    key = image_key("clip-1", {"track": "sam2", "depth": "ground_truth"}, "c", "a")
    assert set(rows) == {
        image_key("clip-1", found.provenance, x, y) for x, y in (("a", "b"), ("a", "c"), ("b", "c"))
    }
    masks = found.track.masks
    pair = PairFrames(
        found.frames, masks[:, 1], masks[:, 0], found.depth.depth.astype(float), found.depth.valid
    )
    assert np.array_equal(rows[key], pair_row(pair, ENCODER, SETTINGS))


def test_the_table_names_its_columns_encoder_and_crop() -> None:
    table = image_table([clip_rows(images(), ENCODER, SETTINGS)], ENCODER, SETTINGS)
    assert table.columns == (*dense_columns(), *token_columns(3))
    assert len(table.keys) == 3
    assert table.meta[ENCODER_KEY] == "sha"
    assert '"size_px": 16' in table.meta[CROP_KEY]
    other = {"x": np.zeros(len(dense_columns()) + 5)}
    with pytest.raises(ImageTableError, match="token widths"):
        image_table([clip_rows(images(), ENCODER, SETTINGS), other], ENCODER, SETTINGS)


@dataclass
class Plan:
    names: dict[Stage, str]


@dataclass
class Runner:
    outputs_by_variant: dict[str, VariantOutputs | None]
    asked: list[str] = field(default_factory=list)

    @property
    def dataset(self) -> Any:
        return type("Dataset", (), {"video": lambda _self, clip: Video()})()

    def outputs(self, clip: Any, variant: Variant) -> VariantOutputs | None:
        self.asked.append(variant.name)
        return self.outputs_by_variant.get(variant.name)


def test_a_clip_is_read_under_its_arm_and_fails_without_masks_or_depth() -> None:
    arm = (True, False, True)
    loaded = VariantOutputs(
        arm_variant(arm),
        Plan(NAMES),  # type: ignore[arg-type]
        {Stage.TRACK: track(), Stage.DEPTH: depth()},
    )
    clip = type("Clip", (), {"clip_id": "clip-1"})()
    runner = Runner({arm_name(arm): loaded})
    found = read_images(runner, clip, arm)  # type: ignore[arg-type]
    assert runner.asked == ["A_tpt"]
    assert found is not None and found.provenance["track"] == "sam2"
    no_depth = VariantOutputs(arm_variant(arm), Plan(NAMES), {Stage.TRACK: track()})  # type: ignore[arg-type]
    assert read_images(Runner({"A_tpt": no_depth}), clip, arm) is None  # type: ignore[arg-type]
    assert read_images(Runner({}), clip, arm) is None  # type: ignore[arg-type]


def test_clips_are_grouped_by_the_arm_they_draw_in_arm_order() -> None:
    ids = [f"clip-{i}" for i in range(40)]
    groups = arm_groups(ids, 0, 0.5)
    assert sorted(i for members in groups.values() for i in members) == sorted(ids)
    assert list(groups) == [a for a in ARMS if a in groups]
    assert all(drawn_arm(i, 0, 0.5) == arm for arm, members in groups.items() for i in members)
    assert list(arm_groups(ids, 0, 1.0)) == [(True, True, True)]
