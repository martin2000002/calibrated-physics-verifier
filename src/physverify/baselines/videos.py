"""A clip as one video file a judge can read."""

from __future__ import annotations

from pathlib import Path

from physverify.core.video import VIDEO_SUFFIX, write_mp4
from physverify.datasets._base import DatasetSource
from physverify.datasets._layout import resolve_data_path
from physverify.schema.clip import ClipManifest
from physverify.synth.store import FRAMES_DIR, VIDEO


def judge_video(source: DatasetSource, clip: ClipManifest, data_dir: Path, encoded: Path) -> Path:
    """The clip's own video file; else the store's ``video.mp4`` beside its frames directory;
    else its frames encoded once under ``encoded``."""
    path = resolve_data_path(clip.video_path, data_dir)
    if path.is_file():
        return path
    beside = path.parent / VIDEO
    if path.name == FRAMES_DIR and beside.is_file():
        return beside
    target = encoded / clip.dataset / f"{clip.clip_id}{VIDEO_SUFFIX}"
    if not target.is_file():
        video = source.video(clip)
        write_mp4(video.frames(0, video.n_frames), target, video.real_fps)
    return target
