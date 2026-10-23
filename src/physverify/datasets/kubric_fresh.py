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
from physverify.datasets.kubric_synthetic import KubricSynthetic
from physverify.schema.clip import ClipManifest, ObjectQuery
from physverify.synth.scene_scope import expand, in_scope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from physverify.config.models import Settings

SECTION = "kubric_fresh"


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

    @staticmethod
    def version_of(fresh: KubricFreshSettings) -> str:
        return fresh.version


class KubricFreshOod(KubricFresh):
    name: ClassVar[str] = "kubric_fresh_ood"

    @staticmethod
    def version_of(fresh: KubricFreshSettings) -> str:
        return fresh.ood_version


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
