"""Generated frames to mp4 through OpenCV (a base dependency; ``core/video.py`` reads it)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from physverify.core.video import MP4_FOURCC
from physverify.timing.records import TimingError

VIDEO_SUFFIX: Final = ".mp4"
RGB_CHANNELS: Final = 3
VIDEO_NDIM: Final = 4  # (frames, height, width, channels)
UINT8_MAX: Final = 255.0


def to_uint8(frames: NDArray[np.floating] | NDArray[np.uint8]) -> NDArray[np.uint8]:
    """uint8 frames unchanged; float frames in [0, 1] become ``round(255 x)`` (clipped first)."""
    if frames.dtype == np.uint8:
        return frames.astype(np.uint8, copy=False)
    if not np.issubdtype(frames.dtype, np.floating):
        raise TimingError(f"frames must be uint8 or float in [0, 1], got {frames.dtype}")
    scaled = np.rint(np.clip(frames, 0.0, 1.0) * UINT8_MAX)
    return scaled.astype(np.uint8)


def _check_video(frames: NDArray[np.uint8]) -> tuple[int, int]:
    if frames.ndim != VIDEO_NDIM or frames.shape[-1] != RGB_CHANNELS or not len(frames):
        raise TimingError(f"expected RGB frames (T, H, W, 3) with T >= 1, got {frames.shape}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    return width, height


def write_mp4(frames: NDArray[np.uint8], path: Path, fps: float) -> None:
    """Write RGB frames ``(T, H, W, 3)`` at ``fps``; readers never see a partial file."""
    size = _check_video(frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The suffix stays .mp4 so OpenCV picks the container from the temporary name.
    partial = path.with_name(f".{path.stem}.{os.getpid()}.partial{VIDEO_SUFFIX}")
    writer = cv2.VideoWriter(str(partial), cv2.VideoWriter.fourcc(*MP4_FOURCC), float(fps), size)
    try:
        if not writer.isOpened():
            raise TimingError(f"OpenCV cannot write {MP4_FOURCC} video to {partial}")
        for frame in frames:
            writer.write(np.ascontiguousarray(frame[..., ::-1]))  # RGB -> BGR
    except BaseException:
        writer.release()
        partial.unlink(missing_ok=True)
        raise
    writer.release()
    os.replace(partial, path)
