"""Pipeline stages, their order and the metadata stored next to every cached stage output."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from physverify.schema._validate import (
    check_bool,
    check_enum,
    check_number,
    check_optional_str,
    check_str,
)
from physverify.schema.errors import SchemaError
from physverify.schema.io import SCHEMA_VERSION, Record

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Self


class Stage(str, Enum):
    """Pipeline stage; declaration order is the execution order (STAGE_ORDER)."""

    DETECT = "detect"
    TRACK = "track"
    DEPTH = "depth"
    CAMERA = "camera"
    LIFT = "lift"
    SCALE = "scale"
    KINEMATICS = "kinematics"
    CONTACTS = "contacts"


STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)


def stage_index(stage: Stage) -> int:
    return STAGE_ORDER.index(stage)


def parse_stages(values: Iterable[str]) -> tuple[Stage, ...]:
    """Parse stage names into a tuple sorted by STAGE_ORDER, without duplicates."""
    try:
        stages = {Stage(value) for value in values}
    except ValueError as exc:
        raise SchemaError(f"unknown stage: {exc}") from exc
    return tuple(sorted(stages, key=stage_index))


def check_upstream_order(stage: Stage, upstream: Iterable[Stage]) -> None:
    """A stage may only depend on stages that precede it."""
    late = [s.value for s in upstream if stage_index(s) >= stage_index(stage)]
    if late:
        raise SchemaError(f"{stage.value} cannot depend on later or equal stages {late}")


@dataclass(frozen=True, eq=False)
class StageArtifactMeta(Record):
    """Provenance of one cached stage output (``meta.json`` next to ``output.{npz,json}``)."""

    stage: Stage
    provider: str
    provider_version: str
    params_hash: str
    sampling_hash: str
    upstream: dict[Stage, str]
    error_profile_hash: str
    gt_version: str | None
    device: str | None
    cache_key: str
    clip_id: str
    is_ground_truth: bool
    wall_s: float
    peak_rss_mb: float
    created_utc: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        super().validate()
        check_enum("stage", self.stage, Stage)
        for label in ("provider", "provider_version", "params_hash", "sampling_hash"):
            check_str(label, getattr(self, label))
        for label in ("cache_key", "clip_id", "created_utc", "schema_version"):
            check_str(label, getattr(self, label))
        check_str("error_profile_hash", self.error_profile_hash)
        check_optional_str("gt_version", self.gt_version)
        check_optional_str("device", self.device)
        check_bool("is_ground_truth", self.is_ground_truth)
        check_number("wall_s", self.wall_s)
        check_number("peak_rss_mb", self.peak_rss_mb)
        self._validate_upstream()

    def _validate_upstream(self) -> None:
        if not isinstance(self.upstream, dict):
            raise SchemaError("upstream must be a dict of Stage -> cache key")
        for stage, key in self.upstream.items():
            check_enum("upstream stage", stage, Stage)
            check_str(f"upstream[{stage}]", key)
        check_upstream_order(self.stage, self.upstream)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            stage=Stage(data["stage"]),
            provider=data["provider"],
            provider_version=data["provider_version"],
            params_hash=data["params_hash"],
            sampling_hash=data["sampling_hash"],
            upstream={Stage(k): v for k, v in data["upstream"].items()},
            error_profile_hash=data["error_profile_hash"],
            gt_version=data["gt_version"],
            device=data["device"],
            cache_key=data["cache_key"],
            clip_id=data["clip_id"],
            is_ground_truth=data["is_ground_truth"],
            wall_s=data["wall_s"],
            peak_rss_mb=data["peak_rss_mb"],
            created_utc=data["created_utc"],
            schema_version=data["schema_version"],
        )
