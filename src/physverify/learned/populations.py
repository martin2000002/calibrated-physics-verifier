"""Who the gates and controls of a learned variant read: the internal validation clips of the
training set, by role, with what each control needs to know about them.

- Decoys: every clip ``clip_role`` calls a decoy, its own check's or another's (a decoy is a
  physically valid clip, so a legitimate negative for every check). Its origin is ``own`` when the
  perturbation it mirrors (the one whose seed stream it shares) is aimed at the check, ``cross``
  when it is aimed elsewhere.
- Broken pairs: a scene whose positive and decoy accepted different perturbation seeds; its decoy
  is left out of the decoy control only.
- Positives: invisible by construction or not (``learned.visibility``); a positive whose evidence
  is missing is not marked invisible, so it counts as visible.
- Backgrounds: a validation clip's background is unseen when no scene of the training part of the
  selection (outside the validation split) was rendered with it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from physverify.learned.samples import CLEAN_NAME, Role, clip_role
from physverify.learned.train import is_validation
from physverify.learned.visibility import PositiveEvidence, invisible
from physverify.schema.clip import ClipManifest
from physverify.schema.scene import PerturbationRecord


class Origin(StrEnum):
    OWN = "own"
    CROSS = "cross"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClipFacts:
    """``invisible`` is set for positives only (None: undetermined); ``origin`` and ``broken``
    for decoys only."""

    clip_id: str
    scene_id: str
    role: Role
    invisible: bool | None = None
    origin: Origin | None = None
    broken: bool = False
    unseen_background: bool = False

    @property
    def visible(self) -> bool:
        return self.role is Role.POSITIVE and self.invisible is not True


def aimed_checks(
    target_checks: Mapping[str, tuple[str, ...]], seed_streams: Mapping[str, str]
) -> dict[str, tuple[str, ...]]:
    """Each perturbation's checks; a decoy's are those of the perturbation whose seed stream it
    shares (``seed_streams``: name -> stream name)."""
    return {
        name: checks or target_checks.get(seed_streams.get(name, name), ())
        for name, checks in target_checks.items()
    }


def decoy_origin(
    record: PerturbationRecord, check_id: str, aimed: Mapping[str, tuple[str, ...]]
) -> Origin:
    checks = aimed.get(record.name)
    if not checks:
        return Origin.UNKNOWN
    return Origin.OWN if check_id in checks else Origin.CROSS


def _scene(clip: ClipManifest) -> str:
    return clip.scene_id or clip.clip_id


def broken_scenes(clips: Iterable[ClipManifest]) -> frozenset[str]:
    """Scenes whose positive and decoy perturbation seeds differ (whatever check they aim at)."""
    seeds: dict[str, dict[bool, set[int]]] = {}
    for clip in clips:
        record = clip.perturbation
        if record is None or record.name == CLEAN_NAME:
            continue
        by_kind = seeds.setdefault(_scene(clip), {True: set(), False: set()})
        by_kind[bool(record.target_checks)].add(int(record.seed))
    return frozenset(
        scene
        for scene, kind in seeds.items()
        if kind[True] and kind[False] and kind[True] != kind[False]
    )


def scene_backgrounds(
    clips: Sequence[ClipManifest], background: Callable[[ClipManifest], str]
) -> dict[str, str]:
    """Each scene's background, asked once per scene (its clips share one render setup)."""
    found: dict[str, str] = {}
    for clip in clips:
        scene = _scene(clip)
        if scene not in found:
            found[scene] = background(clip)
    return found


def validation_facts(
    clips: Sequence[ClipManifest],
    check_id: str,
    share: float,
    aimed: Mapping[str, tuple[str, ...]],
    readers: tuple[
        Callable[[ClipManifest, PerturbationRecord], PositiveEvidence],
        Callable[[ClipManifest], str],
    ],
) -> list[ClipFacts]:
    """The validation clips of ``clips`` with a role for ``check_id``. ``readers``: the evidence
    of a positive (asked for positives only) and the background of a scene."""
    evidence, background = readers
    broken = broken_scenes(clips)
    backgrounds = scene_backgrounds(clips, background)
    trained = {b for scene, b in backgrounds.items() if not is_validation(scene, share)}
    facts = []
    for clip in clips:
        scene = _scene(clip)
        role = clip_role(clip.perturbation, check_id)
        if role is None or not is_validation(scene, share):
            continue
        found = _facts(clip, scene, role, check_id, (aimed, broken), evidence)
        facts.append(replace(found, unseen_background=backgrounds[scene] not in trained))
    return facts


def _facts(
    clip: ClipManifest,
    scene: str,
    role: Role,
    check_id: str,
    known: tuple[Mapping[str, tuple[str, ...]], frozenset[str]],
    evidence: Callable[[ClipManifest, PerturbationRecord], PositiveEvidence],
) -> ClipFacts:
    aimed, broken = known
    record = clip.perturbation
    if role is Role.POSITIVE and record is not None:
        seen = invisible(check_id, evidence(clip, record))
        return ClipFacts(clip.clip_id, scene, role, invisible=seen)
    if role is Role.DECOY and record is not None:
        origin = decoy_origin(record, check_id, aimed)
        return ClipFacts(clip.clip_id, scene, role, origin=origin, broken=scene in broken)
    return ClipFacts(clip.clip_id, scene, role)


def populations(facts: Sequence[ClipFacts]) -> dict[str, dict[str, int]]:
    """How many clips each gate and control reads, before any score."""
    roles = {r.value: sum(f.role is r for f in facts) for r in Role}
    positives = [f for f in facts if f.role is Role.POSITIVE]
    decoys = [f for f in facts if f.role is Role.DECOY]
    kept = [f for f in decoys if not f.broken]
    visible = {
        "visible": sum(f.visible for f in positives),
        "decoy": len(decoys),
        "visibility_undetermined": sum(f.invisible is None for f in positives),
    }
    by_origin = {f"decoy_{o.value}": sum(f.origin is o for f in kept) for o in Origin}
    return {
        "validation": roles,
        "G-L0": visible,
        "G-L1": visible,
        "C-dec": {
            "clean": roles[Role.CLEAN.value],
            "decoy": len(kept),
            "broken_excluded": len(decoys) - len(kept),
            **by_origin,
        },
        "C-inv": {
            "invisible": sum(f.invisible is True for f in positives),
            "clean": roles[Role.CLEAN.value],
        },
        "C-scr": {"positive": len(positives), "clean": roles[Role.CLEAN.value]},
        "C-hdri": {
            f"{role.value}_{name}": sum(
                f.role is role and f.unseen_background is unseen for f in facts
            )
            for name, unseen in (("seen", False), ("unseen", True))
            for role in (Role.POSITIVE, Role.CLEAN)
        },
    }
