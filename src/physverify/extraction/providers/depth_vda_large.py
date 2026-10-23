"""Video Depth Anything Large: ``depth/vda_small`` with the ViT-L encoder and its checkpoint.

Same vendored code, inference path and numeric precision (fp16 on a GPU, fp32 on the CPU); only the
encoder and the weights change. Its training data (TartanAir, VKITTI, PointOdyssey, IRS, wild stereo
video, and Depth Anything V2's sets) hold no Kubric or MOVi. The weights are CC-BY-NC-4.0:
``extract fetch-weights`` downloads them and they are never redistributed.
"""

from __future__ import annotations

from typing import ClassVar

from physverify.extraction._base import WeightKind, WeightSpec
from physverify.extraction.providers._vda import CODE, Encoder, VideoDepthAnythingSmall

LARGE_CHECKPOINT = WeightSpec(
    WeightKind.URL,
    "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/"
    "7aafbcb5c6af0bac741aad2b6471894fb4761afa/video_depth_anything_vitl.pth",
    "7aafbcb5c6af0bac741aad2b6471894fb4761afa",
    "43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa",
)
LARGE_ENCODER = Encoder("vitl", 256, (256, 512, 1024, 1024))


class VideoDepthAnythingLarge(VideoDepthAnythingSmall):
    name: ClassVar[str] = "vda_large"
    version: ClassVar[str] = "1"
    weights: ClassVar[tuple[WeightSpec, ...]] = (CODE, LARGE_CHECKPOINT)
    encoder: ClassVar[Encoder] = LARGE_ENCODER


PLUGINS = (VideoDepthAnythingLarge,)
