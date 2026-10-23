"""Parameters of the back-projection lift (``lift/backproject``), public so the diagnosis
metrics that redo its geometry read the same defaults."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from physverify.extraction.backprojection import ShiftStrategy
from physverify.extraction.providers._common import ProviderParams
from physverify.physics.tracks import OCCLUDED_AT
from physverify.schema.states import PrimitiveKind

GravityMode = Literal["auto", "camera_pose", "level_camera", "none"]
# "none" is version 1 (in_frame = visible, occlusion unknown); "interpolate" infers both inside
# interior mask gaps (``providers/_visibility.py``).
GapMode = Literal["none", "interpolate"]
# "none" is version 2; "median" stabilises depth and image extent along each visible run
# (``providers/_temporal_smoothing.py``), off until a gate run adopts it. Run ends: "keep" (the
# first and last two rows stay unfiltered) is the only eligible form; "shifted" fails the survival
# test of the detections and exists only for a descriptive comparison run.
TemporalFilter = Literal["none", "median"]
EdgeMode = Literal["shifted", "keep"]


class LiftParams(ProviderParams):
    shift: ShiftStrategy = ShiftStrategy.ZERO_SHIFT
    gravity: GravityMode = "auto"
    gravity_by_dataset: dict[str, GravityMode] = {}
    level_sigma_deg: float = Field(default=10.0, ge=0.0)
    pose_sigma_deg: float = Field(default=2.0, ge=0.0)
    min_area_px: int = Field(default=9, ge=1)
    erode_px: int = Field(default=1, ge=0)
    extent_percentiles: tuple[float, float] = (2.0, 98.0)
    center_offset_frac: dict[PrimitiveKind, float] = {
        PrimitiveKind.SPHERE: 0.354,
        PrimitiveKind.CYLINDER: 0.354,
        PrimitiveKind.BOX: 0.5,
        PrimitiveKind.UNKNOWN: 0.4,
    }
    primitive_by_label: dict[str, PrimitiveKind] = {
        "ball": PrimitiveKind.SPHERE,
        "sphere": PrimitiveKind.SPHERE,
        "cube": PrimitiveKind.BOX,
        "box": PrimitiveKind.BOX,
        "cylinder": PrimitiveKind.CYLINDER,
    }
    sigma_mode: Literal["closed_form", "monte_carlo"] = "closed_form"
    mc_draws: int = Field(default=500, ge=2)
    gap_visibility: GapMode = "interpolate"
    # Must equal the checks' ``occluded_at`` (resolve_plan refuses otherwise); a gap estimate
    # below it and above 0 is unknown.
    gap_occluded_at: float = Field(default=OCCLUDED_AT, gt=0.0, le=1.0)
    temporal_filter: TemporalFilter = "none"
    temporal_window: int = Field(default=5, ge=3)
    temporal_edges: EdgeMode = "keep"

    @field_validator("temporal_window")
    @classmethod
    def _odd_window(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("temporal_window must be odd")
        return value
