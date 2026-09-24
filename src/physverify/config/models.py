"""Typed settings shared by every subsystem; each subsystem validates its own section."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, model_validator

from physverify.core.errors import ConfigError
from physverify.schema.io import stable_hash

M = TypeVar("M", bound=BaseModel)

DEVICE_PATTERN = re.compile(r"^(cpu|cuda(:\d+)?)$")

_DERIVED_DIRS = {
    "raw_dir": "raw",
    "synthetic_dir": "synthetic",
    "cache_dir": "cache",
    "runs_dir": "runs",
    "calibration_dir": "calibration",
    "weights_dir": "weights",
    "vendor_dir": "vendor",
}
_CALIBRATION_FILES = {
    "error_profile": "stage_errors.json",
    "thresholds": "thresholds.json",
}


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsSettings(FrozenModel):
    """Data locations. Unset directories derive from ``data_dir``; the loader anchors relative
    paths at the repository root and ``~`` at the injected HOME."""

    data_dir: Path = Path("data")
    raw_dir: Path
    synthetic_dir: Path
    cache_dir: Path
    runs_dir: Path
    calibration_dir: Path
    error_profile: Path
    thresholds: Path
    weights_dir: Path
    vendor_dir: Path
    annotations_dir: Path = Path("annotations")
    hf_home: Path = Path("~/.cache/huggingface")

    @model_validator(mode="before")
    @classmethod
    def _derive(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        filled = dict(data)
        root = Path(filled.get("data_dir", "data"))
        for name, sub in _DERIVED_DIRS.items():
            filled.setdefault(name, root / sub)
        calibration = Path(filled["calibration_dir"])
        for name, file_name in _CALIBRATION_FILES.items():
            filled.setdefault(name, calibration / file_name)
        return filled

    def anchored(self, root: Path, home: Path) -> PathsSettings:
        """Copy with ``~`` expanded to ``home`` and relative paths made absolute under ``root``."""
        values = {name: _anchor(value, root, home) for name, value in self}
        return PathsSettings(**values)


def _anchor(path: Path, root: Path, home: Path) -> Path:
    parts = path.parts
    if parts and parts[0] == "~":
        return home.joinpath(*parts[1:])
    return path if path.is_absolute() else root / path


class RuntimeSettings(FrozenModel):
    threads: int = Field(default=4, ge=1)
    device: str = "cpu"
    heavy_lock: str = ""
    lock_timeout_s: float | None = Field(default=None, ge=0)
    default_profile: str = "fast"
    max_frame_bytes: int = Field(default=1_500_000_000, gt=0)
    docker_image: str = ""

    @model_validator(mode="after")
    def _check_device(self) -> RuntimeSettings:
        if not DEVICE_PATTERN.match(self.device):
            raise ValueError(f"device must be cpu, cuda or cuda:N, got {self.device!r}")
        return self


class SizePrior(FrozenModel):
    """Seed prior on the characteristic length of a labelled object (an estimate, not a
    measurement): mean in meters and relative spread (one sigma)."""

    mean_m: float = Field(gt=0)
    rel_spread: float = Field(ge=0)


class SizePriorTable(RootModel[dict[str, SizePrior]]):
    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True)
class Settings:
    """Immutable effective configuration of one invocation."""

    paths: PathsSettings
    runtime: RuntimeSettings
    profile: str
    config_dir: Path
    repo_root: Path
    sections: Mapping[str, Any]
    sources: tuple[str, ...]

    def section(self, name: str, model: type[M]) -> M:
        """Validate one configuration section with the subsystem's own model."""
        try:
            return model.model_validate(self.sections.get(name, {}))
        except ValidationError as exc:
            raise ConfigError(f"[{name}] is invalid:\n{exc}") from exc

    def config_hash(self) -> str:
        return stable_hash(
            {
                "paths": self.paths.model_dump(mode="json"),
                "runtime": self.runtime.model_dump(mode="json"),
                "profile": self.profile,
                "sections": dict(self.sections),
            }
        )
