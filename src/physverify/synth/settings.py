"""Typed ``[dataset]`` section (configs/dataset.toml) and the parameters shared by every family."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from physverify.config.models import FrozenModel, Settings, SizePrior, SizePriorTable
from physverify.core.errors import ConfigError
from physverify.core.hashing import params_hash
from physverify.schema.scene import AssetSource
from physverify.synth.families._base import SceneFamily
from physverify.synth.perturbations._base import ScenePerturbation, TrajectoryPerturbation

SECTION = "dataset"
PRIORS_SECTION = "size_priors"
MAX_LEVEL = 4
NON_CONTENT_KEYS = ("render", "subsets", "version")

type FamilyClass = type[SceneFamily]
type PerturbationClass = type[ScenePerturbation] | type[TrajectoryPerturbation]


class FloorSettings(FrozenModel):
    """The ground plane (environment). Bullet multiplies both coefficients by the object's."""

    friction: float = Field(default=0.6, ge=0.0)
    restitution: float = Field(default=1.0, ge=0.0)


class ContactSettings(FrozenModel):
    tol_m: float = Field(default=0.002, gt=0.0)
    gap_frames: float = Field(default=0.25, gt=0.0)
    impact_max_frames: float = Field(default=0.5, gt=0.0)
    impact_min_speed: float = Field(default=0.05, ge=0.0)
    normal_split_deg: float = Field(default=10.0, gt=0.0)
    area_split_rel: float = Field(default=0.25, gt=0.0)
    drift_split_m: float = Field(default=0.01, gt=0.0)


class RenderSettings(FrozenModel):
    jobs_per_process: int = Field(default=20, ge=1)
    workers: int = Field(default=1, ge=1)
    far_z_m: float = Field(default=1e4, gt=0.0)


class SubsetSettings(FrozenModel):
    scenes_per_family: int = Field(ge=1)
    levels: tuple[int, ...]
    perturbations: dict[str, tuple[str, ...]]
    level_mode: Literal["all", "cycle"] = "all"

    @model_validator(mode="after")
    def _levels_in_range(self) -> SubsetSettings:
        if not self.levels or any(not 1 <= level <= MAX_LEVEL for level in self.levels):
            raise ValueError(f"levels must be a non-empty subset of 1..{MAX_LEVEL}")
        return self


class DatasetSettings(FrozenModel):
    version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    n_frames: int = Field(ge=8)
    fps: int = Field(gt=0)
    step_rate: int = 240
    resolution: int = Field(ge=16)
    samples_per_pixel: int = Field(ge=1)
    background: Literal["plain", "hdri"] = "plain"
    assets: Literal["kubasic"] = "kubasic"
    calib_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    floor: FloorSettings = FloorSettings()
    contacts: ContactSettings = ContactSettings()
    render: RenderSettings = RenderSettings()
    subsets: dict[str, SubsetSettings] = {}
    families: dict[str, dict[str, Any]] = {}
    perturbations: dict[str, dict[str, Any]] = {}

    @model_validator(mode="after")
    def _step_rate_divides(self) -> DatasetSettings:
        if self.step_rate % self.fps:
            raise ValueError(
                "step_rate must be a multiple of fps (PyBullet always steps at 240 Hz)"
            )
        return self

    def content_hash(self) -> str:
        """Hash of everything that changes a clip's content (not paths, workers or subsets)."""
        dumped = self.model_dump(mode="json")
        return params_hash({k: v for k, v in dumped.items() if k not in NON_CONTENT_KEYS})


class SceneCommon(FrozenModel):
    """Parameters every family receives; family ``Params`` subclass it."""

    n_frames: int
    fps: int
    step_rate: int
    resolution: tuple[int, int]
    spp: int
    background: str
    asset_source: AssetSource = AssetSource.KUBASIC
    size_priors: dict[str, SizePrior]
    floor: FloorSettings


class PerturbationCommon(FrozenModel):
    """Parameters every perturbation receives; ``levels[k - 1]`` is the target of level k."""

    levels: tuple[float, float, float, float]
    max_attempts: int = Field(default=8, ge=1)
    contact_tol_m: float = Field(default=0.002, gt=0.0)


def load_dataset_settings(settings: Settings) -> DatasetSettings:
    return settings.section(SECTION, DatasetSettings)


def dataset_root(settings: Settings, ds: DatasetSettings) -> Path:
    """``<paths.synthetic_dir>/<version>``: one directory per dataset version."""
    return settings.paths.synthetic_dir / ds.version


def load_size_priors(settings: Settings) -> dict[str, SizePrior]:
    return dict(settings.section(PRIORS_SECTION, SizePriorTable).root)


def scene_common(ds: DatasetSettings, priors: dict[str, SizePrior]) -> dict[str, Any]:
    return {
        "n_frames": ds.n_frames,
        "fps": ds.fps,
        "step_rate": ds.step_rate,
        "resolution": (ds.resolution, ds.resolution),
        "spp": ds.samples_per_pixel,
        "background": ds.background,
        "asset_source": AssetSource.KUBASIC,
        "size_priors": priors,
        "floor": ds.floor,
    }


def family_params(
    family: FamilyClass, ds: DatasetSettings, priors: dict[str, SizePrior]
) -> SceneCommon:
    """``family.Params`` from the shared values plus ``[dataset.families.<name>]`` overrides."""
    name = family.name
    raw = {**scene_common(ds, priors), **ds.families.get(name, {})}
    params: SceneCommon = _validated(family, f"{SECTION}.families.{name}", raw)
    return params


def perturbation_params(plugin: PerturbationClass, ds: DatasetSettings) -> Any:
    """``plugin.Params`` with ``[dataset.perturbations.<name>]`` overrides; the contact tolerance
    always comes from ``[dataset.contacts]`` so there is a single source for it."""
    name = plugin.name
    raw = {**ds.perturbations.get(name, {}), "contact_tol_m": ds.contacts.tol_m}
    return _validated(plugin, f"{SECTION}.perturbations.{name}", raw)


def _validated(plugin: FamilyClass | PerturbationClass, where: str, raw: dict[str, Any]) -> Any:
    try:
        return plugin.Params.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"[{where}] is invalid:\n{exc}") from exc
