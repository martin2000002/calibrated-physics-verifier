"""The gates and controls of a learned variant in their fixed order, on clip scores of the
internal validation clips (``learned.populations``).

G-L0 reads the ground-truth arm. When it does not pass, nothing else is computed: the check stops
there. Otherwise G-L1 and every control read the perceived arm, C-scr pairs it with the
single-frame rows and C-hdri splits it by whether training saw the scene's background. G-L1
passes with its controls when its bound passes and C-dec, C-scr and C-inv (or C-inv not
evaluable) pass; C-hdri only flags. C-dec is also reported per decoy origin, as a description
outside any verdict.

Each gate compares a positive side with a negative side (C-dec: its decoys with the clean clips;
C-hdri: positives with clean clips on each background side). Before any interval, a gate fails
when a side has fewer than ``min_scored`` scored clips (undefined) or when the unscored rates of
its two sides differ by more than ``coverage_gap`` (biased coverage).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

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
Keep = Callable[[ClipFacts], bool]


@dataclass(frozen=True)
class ArmScores:
    """Clip scores by clip id under each arm."""

    oracle: Mapping[str, float]
    perceived: Mapping[str, float]
    single_frame: Mapping[str, float]


@dataclass(frozen=True)
class Side:
    """The clips of one role a gate reads, and those of them scored under every reading."""

    label: str
    clips: int
    scored: tuple[ClipScore, ...]

    @property
    def unscored(self) -> int:
        return self.clips - len(self.scored)

    @property
    def unscored_rate(self) -> float:
        return self.unscored / self.clips if self.clips else float("nan")

    @property
    def exact_unscored_rate(self) -> Fraction:
        """For the coverage rule, which compares rates exactly; only called with clips > 0."""
        return Fraction(self.unscored, self.clips)


Pair = tuple[Side, Side]
"""(positive side, negative side)."""


@dataclass(frozen=True)
class GateReport:
    """``unscored``: per gate, per side, its clips and how many of them have no score."""

    results: dict[str, ControlResult]
    decoys_by_origin: dict[str, ControlResult]
    oracle_passed: bool
    perceived_passed: bool
    unscored: dict[str, dict[str, dict[str, int]]]


def scored(facts: Sequence[ClipFacts], scores: Mapping[str, float], keep: Keep) -> list[ClipScore]:
    """The kept clips that have a score; a clip without one is left out (counted by ``side``)."""
    return [
        ClipScore(f.clip_id, f.scene_id, f.role, scores[f.clip_id])
        for f in facts
        if keep(f) and f.clip_id in scores
    ]


def side(
    label: str, facts: Sequence[ClipFacts], keep: Keep, *readings: Mapping[str, float]
) -> Side:
    """Scores from the first reading, of the kept clips that every reading scores."""
    kept = [f for f in facts if keep(f)]
    first, *rest = readings
    found = scored(kept, first, lambda f: all(f.clip_id in r for r in rest))
    return Side(label, len(kept), tuple(found))


def _role(role: Role) -> Keep:
    return lambda f: f.role is role


def _kept_decoy(origin: Origin | None = None) -> Keep:
    """Decoys outside broken pairs; of ``origin`` only when given."""
    return lambda f: (
        f.role is Role.DECOY and not f.broken and (origin is None or f.origin is origin)
    )


def _visible(f: ClipFacts) -> bool:
    return f.visible


def _invisible(f: ClipFacts) -> bool:
    return f.invisible is True


def _paired(f: ClipFacts) -> bool:
    return f.role in (Role.POSITIVE, Role.CLEAN)


def _background(unseen: bool, role: Role) -> Keep:
    return lambda f: f.role is role and f.unseen_background is unseen


def _two(
    labels: tuple[str, str],
    facts: Sequence[ClipFacts],
    keeps: tuple[Keep, Keep],
    *readings: Mapping[str, float],
) -> Pair:
    return (
        side(labels[0], facts, keeps[0], *readings),
        side(labels[1], facts, keeps[1], *readings),
    )


def sides(facts: Sequence[ClipFacts], scores: ArmScores) -> dict[str, tuple[Pair, ...]]:
    """The sides of every gate and control, in ``ORDER``."""
    oracle, perceived, single = scores.oracle, scores.perceived, scores.single_frame
    clean, positive, decoy = _role(Role.CLEAN), _role(Role.POSITIVE), _role(Role.DECOY)
    return {
        "G-L0": (_two(("visible", "decoy"), facts, (_visible, decoy), oracle),),
        "G-L1": (_two(("visible", "decoy"), facts, (_visible, decoy), perceived),),
        "C-dec": (_two(("decoy", "clean"), facts, (_kept_decoy(), clean), perceived),),
        "C-inv": (_two(("invisible", "clean"), facts, (_invisible, clean), perceived),),
        "C-scr": (_two(("positive", "clean"), facts, (positive, clean), perceived, single),),
        "C-hdri": tuple(
            _two(
                (f"positive_{name}", f"clean_{name}"),
                facts,
                (_background(unseen, Role.POSITIVE), _background(unseen, Role.CLEAN)),
                perceived,
            )
            for name, unseen in (("seen", False), ("unseen", True))
        ),
    }


def coverage_failure(
    name: str, pairs: Sequence[Pair], settings: ControlSettings
) -> ControlResult | None:
    """The failure of a gate whose sides are too small or unevenly scored; None when neither."""
    for positive, negative in pairs:
        few = [s.label for s in (positive, negative) if len(s.scored) < settings.min_scored]
        if few:
            why = f"undefined: fewer than {settings.min_scored} scored clips in {few}"
            return ControlResult(name, Status.FAIL, float("nan"), None, why)
        gap = abs(positive.exact_unscored_rate - negative.exact_unscored_rate)
        if gap > Fraction(str(settings.coverage_gap)):  # the configured decimal, exactly
            why = (
                f"biased coverage: unscored {positive.label} {positive.unscored_rate:.4f} "
                f"vs {negative.label} {negative.unscored_rate:.4f}"
            )
            return ControlResult(name, Status.FAIL, float(gap), None, why)
    return None


def _checked(
    name: str,
    pairs: Sequence[Pair],
    settings: ControlSettings,
    compute: Callable[[], ControlResult],
) -> ControlResult:
    failure = coverage_failure(name, pairs, settings)
    return compute() if failure is None else failure


def _not_run(name: str, why: str) -> ControlResult:
    return ControlResult(name, Status.NOT_RUN, float("nan"), None, why)


def unscored_counts(
    found: Mapping[str, tuple[Pair, ...]],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        name: {s.label: {"clips": s.clips, "unscored": s.unscored} for pair in pairs for s in pair}
        for name, pairs in found.items()
    }


def run_gates(
    facts: Sequence[ClipFacts], scores: ArmScores, settings: ControlSettings
) -> GateReport:
    found = sides(facts, scores)
    unscored = unscored_counts(found)
    ((visible, decoys),) = found["G-L0"]
    oracle = _checked(
        "G-L0",
        found["G-L0"],
        settings,
        lambda: oracle_gate(visible.scored, decoys.scored, settings),
    )
    if not oracle.passed:
        stopped = {name: _not_run(name, "G-L0 did not pass") for name in PERCEIVED_STEPS}
        return GateReport({"G-L0": oracle, **stopped}, {}, False, False, unscored)
    results = {"G-L0": oracle, **_perceived(facts, scores, found, settings)}
    passed = (
        all(results[name].passed for name in ("G-L1", "C-dec", "C-scr"))
        and results["C-inv"].status in (Status.PASS, Status.NOT_EVALUABLE)
    )  # fmt: skip
    by_origin = _by_origin(facts, scores.perceived, settings)
    return GateReport(results, by_origin, True, passed, unscored)


def _perceived(
    facts: Sequence[ClipFacts],
    scores: ArmScores,
    found: Mapping[str, tuple[Pair, ...]],
    settings: ControlSettings,
) -> dict[str, ControlResult]:
    ((visible, decoys),) = found["G-L1"]
    ((kept_decoys, clean),) = found["C-dec"]
    paired = scored(facts, scores.perceived, _paired)
    single = scored(facts, scores.single_frame, _paired)
    seen, unseen = ([*p.scored, *n.scored] for p, n in found["C-hdri"])
    return {
        "G-L1": _checked(
            "G-L1",
            found["G-L1"],
            settings,
            lambda: perceived_gate(visible.scored, decoys.scored, settings),
        ),
        "C-dec": _checked(
            "C-dec",
            found["C-dec"],
            settings,
            lambda: decoy_control([*clean.scored, *kept_decoys.scored], settings),
        ),
        "C-inv": _invisible_control(found["C-inv"], settings),
        "C-scr": _checked(
            "C-scr",
            found["C-scr"],
            settings,
            lambda: single_frame_control(paired, single, settings),
        ),
        "C-hdri": _checked(
            "C-hdri",
            found["C-hdri"],
            settings,
            lambda: hdri_control(role_auroc(seen), role_auroc(unseen), settings),
        ),
    }


def _invisible_control(pairs: Sequence[Pair], settings: ControlSettings) -> ControlResult:
    """Fewer than ``min_invisible`` invisible positives in the population: not evaluable, before
    the coverage rule."""
    ((invisible, clean),) = pairs
    if invisible.clips < settings.min_invisible:
        detail = f"n_invisible={invisible.clips}"
        return ControlResult("C-inv", Status.NOT_EVALUABLE, float("nan"), None, detail)
    return _checked(
        "C-inv",
        pairs,
        settings,
        lambda: invisible_control(invisible.scored, clean.scored, settings),
    )


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
