"""Calibration extensions (``checks_only`` subsets): whole calib scenes added for some checks.

An extension scene keeps only the rows, results and sigma pairs of its checks, and its clips never
enter the aggregate score; ``dev_score`` folds the base scenes exactly as without it. Every other
check, the aggregate's population and the base partition are therefore those of the base calib.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace

from physverify.calibration.fit import LevelData
from physverify.core.errors import ConfigError
from physverify.datasets._base import DatasetSource, ScopedSubset, ScopedSubsetSource
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.clip import ClipManifest, Split
from physverify.synth.scene_scope import scene_of
from physverify.verification.runner import VerificationReport


@dataclass(frozen=True)
class ExtensionScope:
    """scene -> the checks it counts for; a scene not listed counts for every check."""

    checks_by_scene: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @property
    def scenes(self) -> frozenset[str]:
        return frozenset(self.checks_by_scene)

    def allows(self, scene_id: str, check_id: str) -> bool:
        checks = self.checks_by_scene.get(scene_id)
        return checks is None or check_id in checks


NO_EXTENSION = ExtensionScope()


@dataclass(frozen=True)
class Extension:
    """The calib clips the extensions add to a base calib, and the scope they count in."""

    subsets: tuple[ScopedSubset, ...] = ()

    @property
    def clips(self) -> tuple[ClipManifest, ...]:
        return tuple(c for subset in self.subsets for c in subset.clips)

    @property
    def scope(self) -> ExtensionScope:
        found: dict[str, frozenset[str]] = {}
        for subset in self.subsets:
            for clip in subset.clips:
                scene = scene_of(clip)
                found[scene] = found.get(scene, frozenset()) | frozenset(subset.checks)
        return ExtensionScope(found)

    def label(self, base: str, clips: Sequence[ClipManifest]) -> str:
        """``base`` joined with every extension that has a clip among ``clips``."""
        ids = {c.clip_id for c in clips}
        used = [s.name for s in self.subsets if any(c.clip_id in ids for c in s.clips)]
        return "+".join([base, *used])


def declared(source: DatasetSource) -> tuple[ScopedSubset, ...]:
    """The dataset's check-scoped subsets that have a calib clip."""
    if not isinstance(source, ScopedSubsetSource):
        return ()
    return tuple(
        replace(s, clips=calib)
        for s in source.scoped_subsets()
        if (calib := tuple(c for c in s.clips if c.split is Split.CALIB))
    )


def extension_of(
    subsets: Sequence[ScopedSubset], base: Sequence[ClipManifest], check_ids: Collection[str]
) -> Extension:
    """The clips of ``subsets`` in scenes ``base`` lacks (a base scene stays the base's alone),
    scoped to active checks."""
    base_scenes = {scene_of(c) for c in base}
    seen: set[str] = set()
    found: list[ScopedSubset] = []
    for subset in subsets:
        unknown = sorted(set(subset.checks) - set(check_ids))
        if unknown:
            raise ConfigError(
                f"subset {subset.name!r}: checks_only names inactive checks {unknown} "
                f"(disable it with --set dataset.subsets.{subset.name}.checks_only=[])"
            )
        added = tuple(
            c for c in subset.clips if scene_of(c) not in base_scenes and c.clip_id not in seen
        )
        seen |= {c.clip_id for c in added}
        found.append(replace(subset, clips=added))
    return Extension(tuple(s for s in found if s.clips))


def unloaded(report: VerificationReport, clip_ids: Collection[str]) -> dict[str, str]:
    """clip -> error, for the ``clip_ids`` whose every result abstains ``MISSING_INPUT``: no input
    reached any of their checks (e.g. a heavy stage not in the cache)."""
    by_clip: dict[str, list[CheckResult]] = {}
    for result in report.results:
        if result.clip_id in clip_ids:
            by_clip.setdefault(result.clip_id, []).append(result)
    return {
        clip: next((str(e) for r in results if (e := r.details.get("error"))), "missing input")
        for clip, results in sorted(by_clip.items())
        if all(r.abstain_reason is AbstainReason.MISSING_INPUT for r in results)
    }


def restrict(data: LevelData, scope: ExtensionScope, scene_by_clip: Mapping[str, str]) -> LevelData:
    """``data`` without what an extension scene contributes to checks outside its scope."""
    if not scope.checks_by_scene:
        return data
    pairs = data.sigma_pairs
    return replace(
        data,
        rows=tuple(r for r in data.rows if scope.allows(r.scene_id, r.check_id)),
        results=tuple(
            r
            for r in data.results
            if r.clip_id not in scene_by_clip or scope.allows(scene_by_clip[r.clip_id], r.check_id)
        ),
        sigma_pairs=None
        if pairs is None
        else tuple(p for p in pairs if scope.allows(p.scene_id, p.check_id)),
        extension_scenes=data.extension_scenes | scope.scenes,
    )
