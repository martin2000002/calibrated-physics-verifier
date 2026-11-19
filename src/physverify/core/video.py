"""Streaming video decoding: every frame is reduced before it is kept, and the output size is
checked against a byte budget before decoding starts (never materialise native 4K clips)."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from physverify.core.errors import MissingInputError, PhysverifyError, ResourceBudgetError
from physverify.core.resample import resize_frame
from physverify.schema.clip import scaled_size

Frames = NDArray[np.uint8]
FRAME_SUFFIXES = (".png", ".jpg", ".jpeg")
VIDEO_SUFFIX: Final = ".mp4"
# MPEG-4 Part 2: the codec every opencv-python wheel can write (H.264 needs an external encoder).
MP4_FOURCC = "mp4v"


def check_frame_budget(n_frames: int, size: tuple[int, int], max_frame_bytes: int) -> None:
    """Raise ResourceBudgetError when T' x h x w x 3 bytes exceed the budget."""
    needed = n_frames * size[0] * size[1] * 3
    if needed > max_frame_bytes:
        raise ResourceBudgetError(
            f"decoding {n_frames} frames at {size[0]}x{size[1]} needs {needed / 1e9:.2f} GB "
            f"(runtime.max_frame_bytes = {max_frame_bytes / 1e9:.2f} GB); "
            "lower max_side or target_fps"
        )


def write_mp4(frames: Frames, path: Path, fps: float) -> None:
    """Encode T x H x W x 3 RGB frames as MP4; readers never see a partial file."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError(f"expected T x H x W x 3 frames, got shape {frames.shape}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    # The suffix stays .mp4 so OpenCV picks the container from the temporary name.
    partial = path.with_name(f".{path.stem}.{os.getpid()}.partial.mp4")
    fourcc = cv2.VideoWriter.fourcc(*MP4_FOURCC)
    writer = cv2.VideoWriter(str(partial), fourcc, float(fps), (width, height))
    try:
        if not writer.isOpened():
            raise PhysverifyError(f"OpenCV cannot write {MP4_FOURCC} video to {partial}")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    except BaseException:
        writer.release()
        partial.unlink(missing_ok=True)
        raise
    writer.release()
    partial.replace(path)


def _selection(start: int, stop: int, step: int, n_frames: int) -> range:
    if step < 1 or start < 0 or stop > n_frames or start >= stop:
        raise ValueError(f"invalid frame range [{start}, {stop}) step {step} for {n_frames}")
    return range(start, stop, step)


def _stack(frames: Sequence[Frames], size: tuple[int, int]) -> Frames:
    if not frames:
        return np.zeros((0, size[1], size[0], 3), dtype=np.uint8)
    return np.stack(frames).astype(np.uint8, copy=False)


class Mp4VideoSource:
    """Container video read with OpenCV; ``real_fps`` overrides the container rate and
    ``max_frame_bytes`` is ``runtime.max_frame_bytes``."""

    def __init__(
        self,
        path: Path,
        max_frame_bytes: int,
        real_fps: float | None = None,
    ) -> None:
        if not path.is_file():
            raise MissingInputError(f"video not found: {path}")
        capture = cv2.VideoCapture(str(path))
        try:
            self._n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self._container_fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        self._path = path
        self._size = (width, height)
        self._real_fps = self._container_fps if real_fps is None else real_fps
        self._max_frame_bytes = max_frame_bytes

    @property
    def real_fps(self) -> float:
        return self._real_fps

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def native_size(self) -> tuple[int, int]:
        return self._size

    def frames(self, start: int, stop: int, step: int = 1, max_side: int | None = None) -> Frames:
        wanted = _selection(start, stop, step, self._n_frames)
        size = scaled_size(*self._size, max_side)
        check_frame_budget(len(wanted), size, self._max_frame_bytes)
        return _stack([resize_frame(f, size) for f in self._decode(wanted)], size)

    def _decode(self, wanted: range) -> Iterator[Frames]:
        capture = cv2.VideoCapture(str(self._path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, wanted.start)
            for index in range(wanted.start, wanted.stop):
                if (index - wanted.start) % wanted.step:
                    capture.grab()
                    continue
                ok, bgr = capture.read()
                if not ok:
                    raise MissingInputError(f"{self._path}: cannot decode frame {index}")
                yield np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        finally:
            capture.release()


class FrameDirVideoSource:
    """A directory of numbered PNG/JPEG frames (sorted by name) at a declared frame rate;
    ``max_frame_bytes`` is ``runtime.max_frame_bytes``."""

    def __init__(self, directory: Path, real_fps: float, max_frame_bytes: int) -> None:
        self._files = sorted(p for p in directory.iterdir() if p.suffix.lower() in FRAME_SUFFIXES)
        if not self._files:
            raise MissingInputError(f"no frames in {directory}")
        first = _read_rgb(self._files[0])
        self._size = (int(first.shape[1]), int(first.shape[0]))
        self._real_fps = real_fps
        self._max_frame_bytes = max_frame_bytes

    @property
    def real_fps(self) -> float:
        return self._real_fps

    @property
    def n_frames(self) -> int:
        return len(self._files)

    @property
    def native_size(self) -> tuple[int, int]:
        return self._size

    def frames(self, start: int, stop: int, step: int = 1, max_side: int | None = None) -> Frames:
        wanted = _selection(start, stop, step, len(self._files))
        size = scaled_size(*self._size, max_side)
        check_frame_budget(len(wanted), size, self._max_frame_bytes)
        return _stack([resize_frame(_read_rgb(self._files[i]), size) for i in wanted], size)


def _read_rgb(path: Path) -> Frames:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise MissingInputError(f"cannot read frame {path}")
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
