"""The gates and controls in their fixed order: each one's pass, fail and undefined path, the
populations they read (cross decoys counted, broken pairs out of the decoy control only) and the
per-origin decoy description outside every verdict."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from physverify.learned.controls import ClipScore, ControlSettings, Status
from physverify.learned.gates import ArmScores, GateReport, run_gates
from physverify.learned.populations import ClipFacts, Origin, aimed_checks, populations
from physverify.learned.samples import Role

FAST = ControlSettings(n_boot=200)
N = 40
LEVELS = list(np.linspace(0.0, 1.0, N))


def fact(role: Role, i: int, **kw: object) -> ClipFacts:
    return ClipFacts(f"{role.value}{i}", f"scene{i}", role, **kw)  # type: ignore[arg-type]


def facts(n_invisible: int = 6) -> list[ClipFacts]:
    clean = [fact(Role.CLEAN, i) for i in range(N)]
    own = [fact(Role.DECOY, i, origin=Origin.OWN) for i in range(N // 2)]
    cross = [fact(Role.DECOY, i, origin=Origin.CROSS) for i in range(N // 2, N)]
    visible = [fact(Role.POSITIVE, i, invisible=False) for i in range(N)]
    step = N // n_invisible
    spread = (N + i * step + step // 2 for i in range(n_invisible))
    hidden = [fact(Role.POSITIVE, i, invisible=True) for i in spread]
    return clean + own + cross + visible + hidden


def separating(found: list[ClipFacts]) -> dict[str, float]:
    """Visible positives far above; clean, decoys and invisible positives on the same levels."""
    scores: dict[str, float] = {}
    for f in found:
        index = int(f.scene_id.removeprefix("scene")) % N
        scores[f.clip_id] = 10.0 + LEVELS[index] if f.visible else LEVELS[index]
    return scores


def arms(found: list[ClipFacts], **kw: object) -> ArmScores:
    base = separating(found)
    defaults: dict[str, object] = {
        "oracle": base,
        "perceived": base,
        "single_frame": base,
        "hdri": [ClipScore(k, k, Role.CLEAN, v) for k, v in base.items() if "clean" in k]
        + [ClipScore(k, k, Role.POSITIVE, v) for k, v in base.items() if "positive" in k],
    }
    return ArmScores(**{**defaults, **kw})  # type: ignore[arg-type]


def gates(found: list[ClipFacts], **kw: object) -> GateReport:
    return run_gates(found, arms(found, **kw), FAST)


def statuses(report: GateReport) -> dict[str, Status]:
    return {name: r.status for name, r in report.results.items()}


def test_a_clean_separation_passes_every_gate_and_control() -> None:
    report = gates(facts())
    assert all(s is Status.PASS for s in statuses(report).values()), statuses(report)
    assert report.oracle_passed and report.perceived_passed


def test_a_failed_oracle_gate_stops_the_check_before_the_perceived_arm() -> None:
    found = facts()
    flat = dict.fromkeys(separating(found), 0.0)
    report = gates(found, oracle=flat)
    assert report.results["G-L0"].status is Status.FAIL
    assert all(report.results[n].status is Status.NOT_RUN for n in ("G-L1", "C-dec", "C-hdri"))
    assert not report.oracle_passed and not report.perceived_passed
    assert report.decoys_by_origin == {}


def test_an_undefined_oracle_gate_fails() -> None:
    no_decoys = [f for f in facts() if f.role is not Role.DECOY]
    assert gates(no_decoys).results["G-L0"].status is Status.FAIL


def test_the_perceived_gate_fails_without_separation_and_when_undefined() -> None:
    found = facts()
    flat = dict.fromkeys(separating(found), 0.0)
    assert gates(found, perceived=flat).results["G-L1"].status is Status.FAIL
    empty = gates(found, perceived={})
    assert empty.results["G-L1"].status is Status.FAIL
    assert not empty.perceived_passed


def loud_decoys(found: list[ClipFacts], keep: Callable[[ClipFacts], bool]) -> dict[str, float]:
    scores = separating(found)
    return {k: 5.0 if keep(f) else scores[k] for f in found for k in (f.clip_id,)}


def test_cross_decoys_count_in_the_decoy_control() -> None:
    found = facts()
    cross = loud_decoys(found, lambda f: f.origin is Origin.CROSS)
    report = gates(found, perceived=cross)
    assert report.results["C-dec"].status is Status.FAIL
    assert not report.perceived_passed
    assert report.decoys_by_origin["own"].passed
    assert not report.decoys_by_origin["cross"].passed


def test_the_decoy_description_by_origin_stays_outside_the_verdict() -> None:
    report = gates(facts())
    assert set(report.decoys_by_origin) == {"own", "cross"}
    assert {r.name for r in report.decoys_by_origin.values()} == {"C-dec/own", "C-dec/cross"}
    assert set(report.results) == {"G-L0", "G-L1", "C-dec", "C-inv", "C-scr", "C-hdri"}


def test_a_broken_pair_leaves_the_decoy_control_only() -> None:
    found = [replace(f, broken=f.role is Role.DECOY and f.origin is Origin.CROSS) for f in facts()]
    report = gates(found, perceived=loud_decoys(found, lambda f: f.broken))
    assert report.results["C-dec"].passed
    assert "cross" not in report.decoys_by_origin
    assert "n_decoy=40" in report.results["G-L1"].detail


@pytest.mark.parametrize(
    ("drop", "why"),
    [
        (lambda f: f.role is Role.CLEAN, "no clean tau"),
        (lambda f: f.role is Role.DECOY, "no decoys"),
    ],
)
def test_an_undefined_decoy_control_fails(drop: Callable[[ClipFacts], bool], why: str) -> None:
    found = facts()
    scores = {
        k: v for f in found if not drop(f) for k, v in [(f.clip_id, separating(found)[f.clip_id])]
    }
    report = gates(found, perceived=scores)
    assert report.results["C-dec"].status is Status.FAIL
    assert report.results["C-dec"].detail == why


def test_the_invisible_control_passes_fails_or_is_not_evaluable() -> None:
    found = facts()
    shortcut = {**separating(found), **{f.clip_id: 10.0 for f in found if f.invisible}}
    assert gates(found).results["C-inv"].status is Status.PASS
    assert gates(found, perceived=shortcut).results["C-inv"].status is Status.FAIL
    few = gates(facts(n_invisible=4))
    assert few.results["C-inv"].status is Status.NOT_EVALUABLE
    assert few.perceived_passed


def test_an_undefined_invisible_control_fails() -> None:
    found = facts()
    no_clean = {k: v for k, v in separating(found).items() if "clean" not in k}
    assert gates(found, perceived=no_clean).results["C-inv"].status is Status.FAIL


def test_the_single_frame_control_fails_on_temporal_signal_and_when_undefined() -> None:
    found = facts()
    blind = dict.fromkeys(separating(found), 0.0)
    assert gates(found, single_frame=blind).results["C-scr"].status is Status.FAIL
    undefined = gates(found, single_frame={})
    assert undefined.results["C-scr"].status is Status.FAIL
    assert not undefined.perceived_passed


def test_the_unseen_background_control_flags_a_drop_or_an_undefined_value_without_stopping() -> (
    None
):
    found = facts()
    dropped = [
        ClipScore(f"h{i}", f"h{i}", role, 0.0)
        for i, role in enumerate([Role.CLEAN, Role.POSITIVE] * 5)
    ]
    flagged = gates(found, hdri=dropped)
    assert flagged.results["C-hdri"].status is Status.FAIL
    assert flagged.perceived_passed
    assert gates(found, hdri=[]).results["C-hdri"].status is Status.FAIL
    assert gates(found, hdri=None).results["C-hdri"].status is Status.NOT_RUN


def test_populations_count_cross_decoys_broken_pairs_and_undetermined_visibility() -> None:
    found = [
        *facts(),
        fact(Role.POSITIVE, 99),
        fact(Role.DECOY, 98, origin=Origin.OWN, broken=True),
    ]
    counts = populations(found)
    assert counts["G-L0"] == {"visible": N + 1, "decoy": N + 1, "visibility_undetermined": 1}
    assert counts["C-dec"] == {
        "clean": N,
        "decoy": N,
        "broken_excluded": 1,
        "decoy_own": N // 2,
        "decoy_cross": N // 2,
        "decoy_unknown": 0,
    }
    assert counts["C-inv"] == {"invisible": 6, "clean": N}


def test_a_decoy_is_aimed_where_the_perturbation_it_mirrors_is() -> None:
    aimed = aimed_checks(
        {"r1": ("R1",), "r1_decoy": (), "r2": ("R2",), "orphan_decoy": ()},
        {"r1": "r1", "r1_decoy": "r1", "r2": "r2", "orphan_decoy": "orphan_decoy"},
    )
    assert aimed == {"r1": ("R1",), "r1_decoy": ("R1",), "r2": ("R2",), "orphan_decoy": ()}
