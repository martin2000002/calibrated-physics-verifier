"""The only reader of comparison labels: a Kubric clip is perturbed when its manifest carries a
perturbation other than the identity, and belongs to its scene; a real clip is clean and its own
unit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from physverify.baselines.settings import Part
from physverify.calibration.table import is_perturbed
from physverify.schema.clip import ClipManifest
from physverify.synth.scene_scope import scene_of


@dataclass(frozen=True)
class ClipLabel:
    clip_id: str
    scene: str
    perturbed: bool


def labels_of(part: Part, clips: Sequence[ClipManifest]) -> tuple[ClipLabel, ...]:
    if part == "kubric":
        return tuple(ClipLabel(c.clip_id, scene_of(c), is_perturbed(c.perturbation)) for c in clips)
    return tuple(ClipLabel(c.clip_id, c.clip_id, False) for c in clips)
