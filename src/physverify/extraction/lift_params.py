"""Parameters of the back-projection lift (``lift/backproject``), public so the diagnosis
metrics that redo its geometry read the same defaults."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_validator

from physverify.extraction._base import ProviderParams
from physverify.extraction.backprojection import ShiftStrategy
from physverify.physics.running_median import EdgeMode
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
KUBRIC_PRIMITIVES: Mapping[str, PrimitiveKind] = MappingProxyType(
    {
        "ball": PrimitiveKind.SPHERE,
        "sphere": PrimitiveKind.SPHERE,
        "cube": PrimitiveKind.BOX,
        "box": PrimitiveKind.BOX,
        "cylinder": PrimitiveKind.CYLINDER,
    }
)
"""The labels of the synthetic scenes (``lookup_label``: whole phrase, then the last known word)."""
REAL_PRIMITIVES: Mapping[str, PrimitiveKind] = MappingProxyType(
    {
        **KUBRIC_PRIMITIVES,
        **dict.fromkeys(
            ("marble", "globe", "apple", "volleyball", "basketball"),
            PrimitiveKind.SPHERE,
        ),
        **dict.fromkeys(
            ("block", "blocks", "brick", "domino", "dominoes", "book", "toolbox", "die", "dice"),
            PrimitiveKind.BOX,
        ),
        **dict.fromkeys(
            ("bottle", "cup", "mug", "candle", "coin", "tire", "wheel", "pencil"),
            PrimitiveKind.CYLINDER,
        ),
    }
)
"""Head nouns of the real-video queries (Physics-IQ, PISA, VideoPhy) whose shape is one
primitive; words that are also a material or a colour ("glass", "orange") are left out, and
"oranges" (a bag of them) and "duck" stay unknown."""
NESTED_OVERLAP = 0.5
"""A mask with more than half of its area inside another track's mask is a part or a duplicate
of that object."""


class LiftParams(ProviderParams):
    shift: ShiftStrategy = ShiftStrategy.ZERO_SHIFT
    gravity: GravityMode = "auto"
    gravity_by_dataset: dict[str, GravityMode] = {}
    level_sigma_deg: float = Field(default=10.0, ge=0.0)
    # Gravity sigma for a camera pose that carries no estimate of its own.
    pose_sigma_deg: float = Field(default=2.0, ge=0.0)
    min_area_px: int = Field(default=9, ge=1)
    erode_px: int = Field(default=1, ge=0)
    border_margin_px: int = Field(default=2, ge=1)
    extent_percentiles: tuple[float, float] = (2.0, 98.0)
    center_offset_frac: dict[PrimitiveKind, float] = {
        PrimitiveKind.SPHERE: 0.354,
        PrimitiveKind.CYLINDER: 0.354,
        PrimitiveKind.BOX: 0.5,
        PrimitiveKind.UNKNOWN: 0.4,
    }
    primitive_by_label: dict[str, PrimitiveKind] = Field(
        default_factory=lambda: dict(KUBRIC_PRIMITIVES)
    )
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


class UnnestedLiftParams(LiftParams):
    """``lift/backproject_unnested``: the real-video primitives, and tracks whose mask lies more
    than ``nested_overlap`` inside another track's mask left out."""

    primitive_by_label: dict[str, PrimitiveKind] = Field(
        default_factory=lambda: dict(REAL_PRIMITIVES)
    )
    nested_overlap: float = Field(default=NESTED_OVERLAP, gt=0.0, le=1.0)
