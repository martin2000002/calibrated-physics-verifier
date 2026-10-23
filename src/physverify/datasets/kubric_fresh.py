"""``kubric_fresh*``: the sealed Kubric renders of the freeze.

Same generator, store layout and loader as ``kubric_synthetic``, rendered under another dataset
version with a new seed, so only the root and the name differ: the name keys the stage cache and
the sealed sets. ``kubric_fresh_blind`` is ``kubric_fresh`` restricted to ``blind_scenes`` with
the expected object counts withheld from the detector."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from physverify.core.models import FrozenModel
from physverify.datasets._kubric_synthetic import KubricSynthetic
from physverify.schema.clip import ClipManifest, ObjectQuery
from physverify.synth.scene_scope import expand, in_scope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from physverify.config.models import Settings

SECTION = "kubric_fresh"
FRESH_SUBSET = "full"
VERSION_ENV = "PHYSVERIFY__DATASET__VERSION"


class KubricFreshSettings(FrozenModel):
    version: str
    ood_version: str
    blind_scenes: tuple[str, ...]
    scenes_per_family: int = Field(ge=1)
    ood_scenes_per_family: int = Field(ge=1)
    ood_hdri_pool: str


def fresh_settings(settings: Settings) -> KubricFreshSettings:
    return settings.section(SECTION, KubricFreshSettings)


class KubricFresh(KubricSynthetic):
    name: ClassVar[str] = "kubric_fresh"

    def __init__(self, settings: Settings) -> None:
        version = self.version_of(fresh_settings(settings))
        super().__init__(settings, root=settings.paths.synthetic_dir / version)

    @classmethod
    def render_overrides(cls, fresh: KubricFreshSettings, seed: int) -> dict[str, str]:
        """The dataset section a fresh render reads: its version, seed and scenes, with no calib
        clip."""
        return {
            VERSION_ENV: cls.version_of(fresh),
            "PHYSVERIFY__DATASET__SEED": str(seed),
            "PHYSVERIFY__DATASET__CALIB_FRACTION": "0.0",
            f"PHYSVERIFY__DATASET__SUBSETS__{FRESH_SUBSET.upper()}__SCENES_PER_FAMILY": str(
                cls.scenes_of(fresh)
            ),
            **cls.render_extra(fresh),
        }

    @staticmethod
    def version_of(fresh: KubricFreshSettings) -> str:
        return fresh.version

    @staticmethod
    def scenes_of(fresh: KubricFreshSettings) -> int:
        return fresh.scenes_per_family

    @staticmethod
    def render_extra(_fresh: KubricFreshSettings) -> dict[str, str]:
        return {}


class KubricFreshOod(KubricFresh):
    name: ClassVar[str] = "kubric_fresh_ood"

    @staticmethod
    def version_of(fresh: KubricFreshSettings) -> str:
        return fresh.ood_version

    @staticmethod
    def scenes_of(fresh: KubricFreshSettings) -> int:
        return fresh.ood_scenes_per_family

    @staticmethod
    def render_extra(fresh: KubricFreshSettings) -> dict[str, str]:
        return {"PHYSVERIFY__DATASET__HDRI_POOL": fresh.ood_hdri_pool}


class KubricFreshBlind(KubricFresh):
    name: ClassVar[str] = "kubric_fresh_blind"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._scenes = expand(fresh_settings(settings).blind_scenes, SECTION)

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return [
            blind(c)
            for c in super().clips(subset)
            if c.scene_id is not None and in_scope(c.scene_id, self._scenes)
        ]


def blind(clip: ClipManifest) -> ClipManifest:
    queries = tuple(ObjectQuery(q.phrase, expected_count=None) for q in clip.object_queries)
    return replace(clip, object_queries=queries)


PLUGINS = (KubricFresh, KubricFreshOod, KubricFreshBlind)
