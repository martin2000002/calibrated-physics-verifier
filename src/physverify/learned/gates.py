"""The gates and controls of a learned variant in their fixed order, on clip scores of the
internal validation clips (``learned.populations``).

G-L0 reads the ground-truth arm. When it does not pass, nothing else is computed: the check stops
there. Otherwise G-L1 and every control read the perceived arm, C-scr pairs it with the
single-frame rows and C-hdri compares it with the unseen-background set. G-L1 passes with its
controls when its bound passes and C-dec, C-scr and C-inv (or C-inv not evaluable) pass; C-hdri
only flags. C-dec is also reported per decoy origin, as a description outside any verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from physverify.learned.controls import (
    ClipScore,
    ControlResult,
    ControlSettings,
    Status,
    decoy_control,
    hdri_control,
    invisible_control,
    oracle_gate,
    perceived_gate,
    role_auroc,
    single_frame_control,
)
from physverify.learned.populations import ClipFacts, Origin
from physverify.learned.samples import Role

ORDER = ("G-L0", "G-L1", "C-dec", "C-inv", "C-scr", "C-hdri")
PERCEIVED_STEPS = ORDER[1:]


@dataclass(frozen=True)
class ArmScores:
    """Clip scores by clip id under each arm; ``hdri``: every scored clip of the unseen-background
    set (perceived), None when no such set was given."""

    oracle: Mapping[str, float]
    perceived: Mapping[str, float]
    single_frame: Mapping[str, float]
    hdri: Sequence[ClipScore] | None


@dataclass(frozen=True)
class GateReport:
    results: dict[str, ControlResult]
    decoys_by_origin: dict[str, ControlResult]
    oracle_passed: bool
    perceived_passed: bool


def scored(
    facts: Sequence[ClipFacts], scores: Mapping[str, float], keep: Callable[[ClipFacts], bool]
) -> list[ClipScore]:
    """The kept clips that have a score; a clip without one is left out (counted by the caller)."""
    return [
        ClipScore(f.clip_id, f.scene_id, f.role, scores[f.clip_id])
        for f in facts
        if keep(f) and f.clip_id in scores
    ]


def _role(role: Role) -> Callable[[ClipFacts], bool]:
    return lambda f: f.role is role


def _kept_decoy(origin: Origin | None = None) -> Callable[[ClipFacts], bool]:
    """Decoys outside broken pairs; of ``origin`` only when given."""
    return lambda f: (
        f.role is Role.DECOY and not f.broken and (origin is None or f.origin is origin)
    )


def _not_run(name: str, why: str) -> ControlResult:
    return ControlResult(name, Status.NOT_RUN, float("nan"), None, why)


def run_gates(
    facts: Sequence[ClipFacts], scores: ArmScores, settings: ControlSettings
) -> GateReport:
    decoys = _role(Role.DECOY)
    oracle = oracle_gate(
        scored(facts, scores.oracle, lambda f: f.visible),
        scored(facts, scores.oracle, decoys),
        settings,
    )
    if not oracle.passed:
        stopped = {name: _not_run(name, "G-L0 did not pass") for name in PERCEIVED_STEPS}
        return GateReport({"G-L0": oracle, **stopped}, {}, False, False)
    results = {"G-L0": oracle, **_perceived(facts, scores, settings)}
    passed = (
        all(results[name].passed for name in ("G-L1", "C-dec", "C-scr"))
        and results["C-inv"].status in (Status.PASS, Status.NOT_EVALUABLE)
    )  # fmt: skip
    return GateReport(results, _by_origin(facts, scores.perceived, settings), True, passed)


def _perceived(
    facts: Sequence[ClipFacts], scores: ArmScores, settings: ControlSettings
) -> dict[str, ControlResult]:
    perceived = scores.perceived
    clean = scored(facts, perceived, _role(Role.CLEAN))
    paired = scored(facts, perceived, lambda f: f.role in (Role.POSITIVE, Role.CLEAN))
    kept_decoys = scored(facts, perceived, _kept_decoy())
    return {
        "G-L1": perceived_gate(
            scored(facts, perceived, lambda f: f.visible),
            scored(facts, perceived, _role(Role.DECOY)),
            settings,
        ),
        "C-dec": decoy_control(clean + kept_decoys, settings),
        "C-inv": invisible_control(
            scored(facts, perceived, lambda f: f.invisible is True), clean, settings
        ),
        "C-scr": single_frame_control(
            paired,
            scored(facts, scores.single_frame, lambda f: f.role in (Role.POSITIVE, Role.CLEAN)),
            settings,
        ),
        "C-hdri": _hdri(paired, scores.hdri, settings),
    }


def _hdri(
    reference: Sequence[ClipScore], unseen: Sequence[ClipScore] | None, settings: ControlSettings
) -> ControlResult:
    if unseen is None:
        return _not_run("C-hdri", "no unseen-background set given")
    return hdri_control(role_auroc(reference), role_auroc(unseen), settings)


def _by_origin(
    facts: Sequence[ClipFacts], scores: Mapping[str, float], settings: ControlSettings
) -> dict[str, ControlResult]:
    clean = scored(facts, scores, _role(Role.CLEAN))
    found = {f.origin for f in facts if f.role is Role.DECOY and not f.broken}
    return {
        origin.value: decoy_control(
            clean + scored(facts, scores, _kept_decoy(origin)), settings, f"C-dec/{origin.value}"
        )
        for origin in Origin
        if origin in found
    }
