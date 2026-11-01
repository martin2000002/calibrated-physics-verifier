"""Parameters of the back-projection lift (``lift/backproject``), public so the diagnosis
metrics that redo its geometry read the same defaults."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_validator

from physverify.extraction._base import ProviderParams
from physverify.extraction.backprojection import ShiftStrategy, SummaryOptions
from physverify.extraction.ground_anchor import AnchorMode, GroundAnchorParams
from physverify.extraction.size_constancy import SizeConstancyParams
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
    # Fallback only: the assumed (not measured) gravity sigma of an estimated pose whose camera
    # gives no sigma of its own; GeoCalib-based cameras give theirs.
    pose_sigma_deg: float = Field(default=2.0, ge=0.0)
    min_area_px: int = Field(default=9, ge=1)
    erode_px: int = Field(default=1, ge=0)
    border_margin_px: int = Field(default=2, ge=1)
    extent_percentiles: tuple[float, float] = (0.0, 100.0)
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
    # Version 5: the footprint behind a box's yaw drops surface points farther from its median
    # than this many robust sigma (``lift_geometry.footprint_yaw``).
    footprint_trim_mads: float = Field(default=3.0, gt=0.0)
    # Version 5: extents ignore mask parts smaller than this fraction of the largest one.
    min_part_frac: float = Field(default=0.2, ge=0.0, le=1.0)

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


class GroundAnchoredLiftParams(LiftParams):
    """``lift/ground_anchored``: the anchoring settings, fixed a priori (never fitted on calib).

    They live here rather than in a precise-profile table, which would change that profile's
    section and run-config hashes while the provider is still outside the plan."""

    normal_source: Literal["gravity", "depth_plane"] = "gravity"
    envelope_quantile: float = Field(default=0.9, gt=0.5, lt=1.0)
    anchor_z: float = Field(default=2.0, gt=0.0)
    min_anchor_rows: int = Field(default=5, ge=2)
    # Above it the whole clip keeps the network depth: at a typical elevation of 25 deg the
    # anchor's relative error would be about 0.35.
    max_normal_sigma_deg: float = Field(default=10.0, gt=0.0)
    steady_window: int = Field(default=7, ge=5)
    # A ray this flat amplifies a normal error by cot(beta) = 5.7: 2 deg of tilt already give 0.2
    # relative depth, the network's own error, so flatter rows keep the network depth.
    min_elevation_deg: float = Field(default=10.0, ge=0.0, lt=90.0)
    # Pixels a body's lowest mask pixels are probed below it for another object's mask.
    stack_reach_px: int = Field(default=4, ge=1)
    # per_row matched per_row_smoothed with true floor rows and was within 0.01 L with perceived
    # ones on calib, so the simpler mode is the default.
    anchor_mode: AnchorMode = AnchorMode.PER_ROW

    @field_validator("steady_window")
    @classmethod
    def _odd_steady_window(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("steady_window must be odd")
        return value


class SizeConstancyLiftParams(LiftParams):
    """``lift/backproject_size_constancy``: the along-ray sigma settings, fixed a priori from
    pixel arithmetic and never swept on calib. Like the ground-anchoring settings they
    live here, outside the profile tables, so no section or run-config hash moves."""

    # A box side is off by up to a pixel at each end (sqrt(2) px together); at 12 px that is
    # already 0.12 of the diameter, so smaller boxes say little about depth.
    min_diameter_px: float = Field(default=12.0, gt=0.0)
    # Two independent one-pixel quantisation errors, one per end of the box side.
    diameter_noise_px: float = Field(default=math.sqrt(2.0), ge=0.0)
    # Mask outline noise (segmentation edge, antialiasing) assumed at 5 % of the diameter.
    shape_sigma_floor: float = Field(default=0.05, ge=0.0)
    # The smallest count whose median is one of the rows rather than the mean of two.
    min_clean_rows: int = Field(default=3, ge=1)
    # A mask with a pixel in the outermost row or column is cut by the frame.
    constancy_border_px: int = Field(default=1, ge=1)


def ground_anchor_params(params: GroundAnchoredLiftParams) -> GroundAnchorParams:
    return GroundAnchorParams(
        envelope_quantile=params.envelope_quantile,
        anchor_z=params.anchor_z,
        min_anchor_rows=params.min_anchor_rows,
        max_normal_sigma_rad=math.radians(params.max_normal_sigma_deg),
        steady_window=params.steady_window,
        min_elevation_rad=math.radians(params.min_elevation_deg),
        anchor_mode=params.anchor_mode,
    )


def size_constancy_params(params: SizeConstancyLiftParams) -> SizeConstancyParams:
    return SizeConstancyParams(
        min_diameter_px=params.min_diameter_px,
        diameter_noise_px=params.diameter_noise_px,
        shape_sigma_floor=params.shape_sigma_floor,
        min_clean_rows=params.min_clean_rows,
        border_margin_px=params.constancy_border_px,
    )


def summary_options(params: LiftParams) -> SummaryOptions:
    """Per-object summary settings, shared by LIFT and the depth metrics that mirror it."""
    return SummaryOptions(
        erode_px=params.erode_px,
        extent_percentiles=params.extent_percentiles,
        min_area_px=params.min_area_px,
        footprint_trim_mads=params.footprint_trim_mads,
        min_part_frac=params.min_part_frac,
    )
