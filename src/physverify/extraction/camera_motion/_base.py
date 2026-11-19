"""Camera motion estimators: how much the background of a clip moves, one estimator per module.

A side stage keyed from TRACK, outside ``STAGE_ORDER``: it reads the frames at the track's
sampling and the track's masks, which mark what is not background."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from physverify.config.plugin_context import PluginKind
from physverify.core.errors import ConfigError
from physverify.core.registry import Registry, discover
from physverify.datasets._base import VideoSource
from physverify.schema.camera_motion import CameraMotion
from physverify.schema.perception import MaskTrack

CAMERA_MOTION_PACKAGE = "physverify.extraction.camera_motion"


class CameraMotionParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CameraMotionEstimator(Protocol):
    """Registered by ``name``; constructed as ``cls(params)`` with an instance of its ``Params``.
    ``version`` is bumped whenever the output changes for the same inputs and params."""

    name: ClassVar[str]
    version: ClassVar[str]
    Params: ClassVar[type[CameraMotionParams]]

    def __init__(self, params: CameraMotionParams) -> None: ...

    @property
    def params(self) -> CameraMotionParams: ...

    def estimate(self, video: VideoSource, track: MaskTrack) -> CameraMotion:
        """The motion between consecutive frames of ``track.sampling``; raises on bad input."""
        ...


CAMERA_MOTION_ESTIMATORS = PluginKind[type[CameraMotionEstimator]]("camera_motion_estimators")


def camera_motion_estimators() -> Registry[type[CameraMotionEstimator]]:
    return discover(CAMERA_MOTION_PACKAGE, (CameraMotionEstimator,), "name")


def build_estimator(
    name: str, params: Mapping[str, Any], registry: Mapping[str, type[CameraMotionEstimator]]
) -> CameraMotionEstimator:
    """The estimator ``name`` with its validated ``params``; a ConfigError names the known ones."""
    if name not in registry:
        known = ", ".join(sorted(registry))
        raise ConfigError(f"camera motion estimator {name!r} is unknown; known: {known}")
    estimator = registry[name]
    try:
        validated = estimator.Params.model_validate(dict(params))
    except ValidationError as error:
        raise ConfigError(f"camera motion estimator {name!r}: invalid params: {error}") from error
    return estimator(validated)
