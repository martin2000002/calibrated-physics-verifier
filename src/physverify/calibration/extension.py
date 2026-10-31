"""Calibration extensions (``checks_only`` subsets): whole calib scenes added for some checks.

An extension scene keeps only the rows, results and sigma pairs of its checks, and its clips never
enter the aggregate score; ``dev_score`` folds the base scenes exactly as without it. Every other
check, the aggregate's population and the base partition are therefore those of the base calib.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from physverify.calibration.fit import LevelData


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
