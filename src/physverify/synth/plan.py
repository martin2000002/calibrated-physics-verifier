"""The dataset plan: scenes, clip variants and the clip ids of every generated subset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from physverify.core.errors import ConfigError, UsageError
from physverify.schema.clip import Split

NONE = "none"
# Fixes of the APPLY ground truth with no alternative convention, in order: the migration at index
# i brings a stored plan from revision i up to this checkout's. ``gt_revision`` lives outside
# ``content_hash``; adding a fix is adding its migration name here.
GT_MIGRATIONS = ("floor-crossing",)
GT_REVISION = len(GT_MIGRATIONS)


def scene_id_of(family: str, index: int) -> str:
    return f"{family}-s{index:03d}"


def clip_id_of(scene_id: str, perturbation: str, level: int) -> str:
    return f"{scene_id}__{NONE}" if perturbation == NONE else f"{scene_id}__{perturbation}_L{level}"


@dataclass(frozen=True)
class ScenePlan:
    scene_id: str
    family: str
    index: int
    split: Split


@dataclass(frozen=True)
class VariantPlan:
    """One clip. ``sim_id`` is the scene for trajectory edits and the clip itself for scene
    perturbations, which need their own simulation."""

    clip_id: str
    scene_id: str
    sim_id: str
    family: str
    index: int
    perturbation: str
    level: int
    target_magnitude: float
    split: Split

    @property
    def scene_perturbation(self) -> bool:
        return self.sim_id != self.scene_id


@dataclass(frozen=True)
class DatasetPlan:
    version: str
    config_hash: str
    scenes: tuple[ScenePlan, ...]
    variants: tuple[VariantPlan, ...]
    subsets: dict[str, tuple[str, ...]]
    gt_revision: int = GT_REVISION

    def pending_migration(self) -> str | None:
        """The first ground-truth migration this plan has not had (None when up to date)."""
        return GT_MIGRATIONS[self.gt_revision] if self.gt_revision < GT_REVISION else None

    def check_gt_revision(self) -> None:
        """A plan built before the current ground-truth revision must be migrated first."""
        pending = self.pending_migration()
        if pending is not None:
            raise ConfigError(
                f"dataset {self.version} has ground-truth revision {self.gt_revision}, this "
                f"checkout {GT_REVISION}: run `physverify dataset migrate --name {pending}` "
                "before building on it"
            )
        if self.gt_revision > GT_REVISION:
            raise ConfigError(
                f"dataset {self.version} has ground-truth revision {self.gt_revision}, newer than "
                f"this checkout's {GT_REVISION}: update the checkout"
            )

    def subset(self, name: str | None) -> DatasetPlan:
        """The part of the plan that ``name`` covers (``None`` = everything planned)."""
        if name is None:
            return self
        if name not in self.subsets:
            raise UsageError(
                f"subset {name!r} is not planned yet: "
                f"run `physverify dataset generate --subset {name}`"
            )
        wanted = set(self.subsets[name])
        variants = tuple(v for v in self.variants if v.clip_id in wanted)
        scene_ids = {v.scene_id for v in variants}
        scenes = tuple(s for s in self.scenes if s.scene_id in scene_ids)
        return replace(self, scenes=scenes, variants=variants, subsets={name: self.subsets[name]})

    def sim_ids(self) -> tuple[str, ...]:
        return tuple(sorted({v.sim_id for v in self.variants}))

    def variant(self, clip_id: str) -> VariantPlan:
        return next(v for v in self.variants if v.clip_id == clip_id)

    def merged(self, other: DatasetPlan) -> DatasetPlan:
        """Union of two plans of the same content and ground-truth revision; the same id must
        mean the same item."""
        if (self.version, self.config_hash) != (other.version, other.config_hash):
            raise ConfigError(
                f"dataset {self.version} was planned with other settings: bump [dataset] version "
                "before changing anything that alters clip content (a plan made before "
                "velocity_convention existed is migrated by "
                "`physverify dataset migrate --name velocity-<convention>`)"
            )
        self.check_gt_revision()
        other.check_gt_revision()
        scenes = _union(self.scenes, other.scenes, "scene_id")
        variants = _union(self.variants, other.variants, "clip_id")
        return replace(
            self, scenes=scenes, variants=variants, subsets={**self.subsets, **other.subsets}
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "config_hash": self.config_hash,
            "gt_revision": self.gt_revision,
            "scenes": [asdict(s) | {"split": s.split.value} for s in self.scenes],
            "variants": [asdict(v) | {"split": v.split.value} for v in self.variants],
            "subsets": {k: list(v) for k, v in sorted(self.subsets.items())},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DatasetPlan:
        return cls(
            version=data["version"],
            config_hash=data["config_hash"],
            scenes=tuple(ScenePlan(**(s | {"split": Split(s["split"])})) for s in data["scenes"]),
            variants=tuple(
                VariantPlan(**(v | {"split": Split(v["split"])})) for v in data["variants"]
            ),
            subsets={k: tuple(v) for k, v in data["subsets"].items()},
            gt_revision=int(data.get("gt_revision", 0)),
        )


def _union(first: tuple[Any, ...], second: tuple[Any, ...], key: str) -> tuple[Any, ...]:
    items = {getattr(item, key): item for item in first}
    for item in second:
        known = items.get(getattr(item, key))
        if known is not None and known != item:
            raise ConfigError(f"plan item {getattr(item, key)} changed: bump [dataset] version")
        items[getattr(item, key)] = item
    return tuple(items[k] for k in sorted(items))
