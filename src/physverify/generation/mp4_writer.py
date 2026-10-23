"""Generated frames as the uint8 RGB video ``core.video.write_mp4`` encodes."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from physverify.core.errors import TimingError

VIDEO_SUFFIX: Final = ".mp4"
RGB_CHANNELS: Final = 3
UINT8_MAX: Final = 255.0


def to_uint8(frames: NDArray[np.floating] | NDArray[np.uint8]) -> NDArray[np.uint8]:
    """uint8 frames unchanged; float frames in [0, 1] become ``round(255 x)`` (clipped first)."""
    if frames.dtype == np.uint8:
        return frames.astype(np.uint8, copy=False)
    if not np.issubdtype(frames.dtype, np.floating):
        raise TimingError(f"frames must be uint8 or float in [0, 1], got {frames.dtype}")
    scaled = np.rint(np.clip(frames, 0.0, 1.0) * UINT8_MAX)
    return scaled.astype(np.uint8)
