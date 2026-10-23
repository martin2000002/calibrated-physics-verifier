"""Typed settings shared by every subsystem; each subsystem validates its own section."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
    ValidationInfo,
    model_validator,
)

from physverify.config.errors import describe_errors
from physverify.core.errors import ConfigError
from physverify.core.models import FrozenModel
from physverify.schema.io import stable_hash

DEVICE_PATTERN = re.compile(r"^(cpu|cuda(:\d+)?)$")


def is_device(value: str) -> bool:
    return DEVICE_PATTERN.match(value) is not None


_DERIVED_DIRS = {
    "raw_dir": "raw",
    "synthetic_dir": "synthetic",
    "cache_dir": "cache",
    "runs_dir": "runs",
    "calibration_dir": "calibration",
    "weights_dir": "weights",
    "vendor_dir": "vendor",
}
_CALIBRATION_FILES = {"thresholds": "thresholds.json"}
ERROR_PROFILE_FILE = "stage_errors.{profile}.json"
UNPROFILED_ERROR_PROFILE_FILE = "stage_errors.json"
"""Error-profile file name when no extraction profile is known (direct validation only)."""
PROFILE_CONTEXT = "profile"
"""Validation-context key of the active extraction profile, which names its error profile."""


def validated[T: BaseModel](
    model: type[T], data: object, section: str, context: Mapping[str, object] | None = None
) -> T:
    """``model`` validated from ``data``; a failure names the configuration ``section``."""
    try:
        return model.model_validate(data, context=context)
    except ValidationError as exc:
        raise ConfigError(f"[{section}] is invalid:\n{describe_errors(section, exc)}") from exc


class PathsSettings(FrozenModel):
    """Data locations. Unset directories derive from ``data_dir``; the loader anchors relative
    paths at the repository root and ``~`` at the injected HOME.

    The error profile defaults to ``stage_errors.<profile>.json``: entries of different
    extraction profiles are measured on different data and never share a file. A missing file
    has no fallback, so reads and ``--write-profile`` always use the same per-profile file."""

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
    def _derive(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, Mapping):
            return data
        filled = dict(data)
        root = Path(filled.get("data_dir", "data"))
        for name, sub in _DERIVED_DIRS.items():
            filled.setdefault(name, root / sub)
        calibration = Path(filled["calibration_dir"])
        profile = (info.context or {}).get(PROFILE_CONTEXT)
        filled.setdefault("error_profile", calibration / error_profile_file(profile))
        for name, file_name in _CALIBRATION_FILES.items():
            filled.setdefault(name, calibration / file_name)
        return filled

    def anchored(self, root: Path, home: Path) -> PathsSettings:
        """Copy with ``~`` expanded to ``home`` and relative paths made absolute under ``root``."""
        values = {name: anchor(value, root, home) for name, value in self}
        return PathsSettings(**values)


def error_profile_file(profile: str | None) -> str:
    """Default error-profile file name of an extraction profile."""
    if profile is None:
        return UNPROFILED_ERROR_PROFILE_FILE
    return ERROR_PROFILE_FILE.format(profile=profile)


def anchor(path: Path, root: Path, home: Path) -> Path:
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
        if not is_device(self.device):
            raise ValueError(f"device must be cpu, cuda or cuda:N, got {self.device!r}")
        return self


class SizePrior(FrozenModel):
    """Seed prior on the characteristic length of a labelled object (an estimate, not a
    measurement): mean in meters and relative spread (one sigma)."""

    mean_m: float = Field(gt=0)
    rel_spread: float = Field(ge=0)


SIZE_PRIORS_SECTION = "size_priors"


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

    def section[M: BaseModel](self, name: str, model: type[M]) -> M:
        """Validate one configuration section with the subsystem's own model."""
        return validated(model, self.sections.get(name, {}), name)

    def config_hash(self) -> str:
        return stable_hash(
            {
                "paths": self.paths.model_dump(mode="json"),
                "runtime": self.runtime.model_dump(mode="json"),
                "profile": self.profile,
                "sections": dict(self.sections),
            }
        )
