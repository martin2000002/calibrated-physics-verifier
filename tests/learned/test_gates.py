"""The gates and controls in their fixed order: each one's pass, fail and undefined path, the
coverage rule (too few scored clips, unevenly scored sides), the populations they read (cross
decoys counted, broken pairs out of the decoy control only, internal validation clips only) and
the per-origin decoy description outside every verdict."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from records import manifest

from physverify.learned.controls import ControlSettings, Status
from physverify.learned.gates import ArmScores, GateReport, run_gates
from physverify.learned.populations import (
    ClipFacts,
    Origin,
    aimed_checks,
    populations,
    validation_facts,
)
from physverify.learned.samples import Role
from physverify.learned.train import is_validation
from physverify.learned.visibility import PositiveEvidence
from physverify.schema.clip import ClipManifest
from physverify.schema.scene import PerturbationRecord

FAST = ControlSettings(n_boot=200)
N = 40
UNSEEN = 8
"""Scenes below this index were rendered with a background training did not see."""
LEVELS = list(np.linspace(0.0, 1.0, N))


def index(f: ClipFacts) -> int:
    return int(f.scene_id.removeprefix("scene")) % N


def fact(role: Role, i: int, **kw: object) -> ClipFacts:
    found = ClipFacts(f"{role.value}{i}", f"scene{i}", role, **kw)  # type: ignore[arg-type]
    return replace(found, unseen_background=index(found) < UNSEEN)


def facts(n_invisible: int = 6, n_decoy: int = N) -> list[ClipFacts]:
    clean = [fact(Role.CLEAN, i) for i in range(N)]
    own = [fact(Role.DECOY, i, origin=Origin.OWN) for i in range(n_decoy // 2)]
    cross = [fact(Role.DECOY, i, origin=Origin.CROSS) for i in range(n_decoy // 2, n_decoy)]
    visible = [fact(Role.POSITIVE, i, invisible=False) for i in range(N)]
    step = N // n_invisible
    spread = (N + i * step + step // 2 for i in range(n_invisible))
    hidden = [fact(Role.POSITIVE, i, invisible=True) for i in spread]
    return clean + own + cross + visible + hidden


def separating(found: list[ClipFacts]) -> dict[str, float]:
    """Visible positives far above; clean, decoys and invisible positives on the same levels."""
    return {f.clip_id: 10.0 + LEVELS[index(f)] if f.visible else LEVELS[index(f)] for f in found}


def arms(found: list[ClipFacts], **kw: object) -> ArmScores:
    base = separating(found)
    defaults = {"oracle": base, "perceived": base, "single_frame": base}
    return ArmScores(**{**defaults, **kw})  # type: ignore[arg-type]


def gates(found: list[ClipFacts], **kw: object) -> GateReport:
    return run_gates(found, arms(found, **kw), FAST)


def statuses(report: GateReport) -> dict[str, Status]:
    return {name: r.status for name, r in report.results.items()}


def without(scores: dict[str, float], drop: Callable[[str], bool]) -> dict[str, float]:
    return {k: v for k, v in scores.items() if not drop(k)}


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


@pytest.mark.parametrize(("n_decoy", "undefined"), [(4, True), (6, False)])
def test_a_side_with_fewer_scored_clips_than_the_minimum_is_undefined(
    n_decoy: int, undefined: bool
) -> None:
    oracle = gates(facts(n_decoy=n_decoy)).results["G-L0"]
    assert oracle.detail.startswith("undefined") is undefined
    assert oracle.passed is not undefined


@pytest.mark.parametrize(("n_unscored", "biased"), [(4, False), (8, True)])
def test_sides_scored_unevenly_fail_on_biased_coverage(n_unscored: int, biased: bool) -> None:
    found = facts()
    missing = {f"decoy{i}" for i in range(n_unscored)}
    report = gates(found, perceived=without(separating(found), missing.__contains__))
    perceived = report.results["G-L1"]
    assert perceived.detail.startswith("biased coverage") is biased
    assert perceived.passed is not biased
    assert report.unscored["G-L1"]["decoy"] == {"clips": N, "unscored": n_unscored}
    assert report.unscored["G-L1"]["visible"] == {"clips": N, "unscored": 0}


def test_unscored_clips_are_counted_per_gate_and_side() -> None:
    found = facts()
    no_clean = without(separating(found), lambda k: k in {"clean0", "clean1"})
    single = without(no_clean, {"positive3"}.__contains__)
    report = gates(found, perceived=no_clean, single_frame=single)
    assert set(report.unscored) == {"G-L0", "G-L1", "C-dec", "C-inv", "C-scr", "C-hdri"}
    assert report.unscored["G-L0"]["decoy"]["unscored"] == 0
    assert report.unscored["C-dec"]["clean"] == {"clips": N, "unscored": 2}
    assert report.unscored["C-scr"]["positive"] == {"clips": N + 6, "unscored": 1}
    assert report.unscored["C-hdri"]["clean_unseen"] == {"clips": UNSEEN, "unscored": 2}


def test_the_perceived_gate_fails_without_separation_and_when_undefined() -> None:
    found = facts()
    flat = dict.fromkeys(separating(found), 0.0)
    assert gates(found, perceived=flat).results["G-L1"].status is Status.FAIL
    empty = gates(found, perceived={})
    assert empty.results["G-L1"].status is Status.FAIL
    assert not empty.perceived_passed


def loud_decoys(found: list[ClipFacts], keep: Callable[[ClipFacts], bool]) -> dict[str, float]:
    scores = separating(found)
    return {f.clip_id: 5.0 if keep(f) else scores[f.clip_id] for f in found}


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
    assert report.unscored["C-dec"]["decoy"]["clips"] == N // 2


@pytest.mark.parametrize("drop", [Role.CLEAN, Role.DECOY])
def test_a_decoy_control_side_without_scores_is_undefined(drop: Role) -> None:
    found = facts()
    scores = without(separating(found), lambda k: k.startswith(drop.value))
    result = gates(found, perceived=scores).results["C-dec"]
    assert result.status is Status.FAIL
    assert result.detail.startswith("undefined") and f"'{drop.value}'" in result.detail


def test_a_decoy_control_without_a_clean_tau_fails() -> None:
    found = [f for f in facts() if f.role is not Role.CLEAN or index(f) < 10]
    result = gates(found).results["C-dec"]
    assert result.status is Status.FAIL
    assert result.detail == "no clean tau"


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
    no_clean = without(separating(found), lambda k: "clean" in k)
    assert gates(found, perceived=no_clean).results["C-inv"].status is Status.FAIL


def test_the_single_frame_control_fails_on_temporal_signal_and_when_undefined() -> None:
    found = facts()
    blind = dict.fromkeys(separating(found), 0.0)
    assert gates(found, single_frame=blind).results["C-scr"].status is Status.FAIL
    undefined = gates(found, single_frame={})
    assert undefined.results["C-scr"].status is Status.FAIL
    assert not undefined.perceived_passed


def test_the_single_frame_control_reports_its_unpaired_clips() -> None:
    found = facts()
    single = without(separating(found), lambda k: k in {"clean5", "clean6"})
    result = gates(found, single_frame=single).results["C-scr"]
    assert result.passed
    assert result.detail == f"n_paired={2 * N + 6 - 2} n_unpaired=2"


def test_the_unseen_background_control_flags_a_drop_or_an_undefined_value_without_stopping() -> (
    None
):
    found = facts()
    unseen = {f.clip_id: LEVELS[index(f)] for f in found if f.unseen_background and f.visible}
    flagged = gates(found, perceived={**separating(found), **unseen})
    assert flagged.results["C-hdri"].status is Status.FAIL
    assert "texture dependence" in flagged.results["C-hdri"].detail
    assert flagged.perceived_passed
    all_seen = gates([replace(f, unseen_background=False) for f in found])
    assert all_seen.results["C-hdri"].status is Status.FAIL
    assert all_seen.results["C-hdri"].detail.startswith("undefined")
    assert all_seen.perceived_passed


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
    assert counts["C-hdri"] == {
        "positive_seen": N - UNSEEN + 5 + 1,
        "clean_seen": N - UNSEEN,
        "positive_unseen": UNSEEN + 1,
        "clean_unseen": UNSEEN,
    }


def test_a_decoy_is_aimed_where_the_perturbation_it_mirrors_is() -> None:
    aimed = aimed_checks(
        {"r1": ("R1",), "r1_decoy": (), "r2": ("R2",), "orphan_decoy": ()},
        {"r1": "r1", "r1_decoy": "r1", "r2": "r2", "orphan_decoy": "orphan_decoy"},
    )
    assert aimed == {"r1": ("R1",), "r1_decoy": ("R1",), "r2": ("R2",), "orphan_decoy": ()}


SHARE = 0.2


def scenes(validation: bool, n: int) -> list[str]:
    found = (f"s{i}" for i in range(10_000))
    return [s for s in found if is_validation(s, SHARE) is validation][:n]


def clean_clip(scene: str, n: int = 0) -> ClipManifest:
    return replace(manifest(), clip_id=f"{scene}_c{n}", scene_id=scene, perturbation=None)


def no_evidence(clip: ClipManifest, record: PerturbationRecord) -> PositiveEvidence:
    raise AssertionError("no positive here")


def test_only_internal_validation_clips_are_read_and_their_background_is_split() -> None:
    held, trained = scenes(True, 2), scenes(False, 3)
    backgrounds = {held[0]: "plain", held[1]: "studio", **dict.fromkeys(trained, "plain")}
    asked: list[str] = []

    def background(clip: ClipManifest) -> str:
        asked.append(clip.scene_id or "")
        return backgrounds[clip.scene_id or ""]

    clips = [*(clean_clip(s) for s in [*held, *trained]), clean_clip(held[0], 1)]
    found = validation_facts(clips, "R1", SHARE, {}, (no_evidence, background))
    assert [f.scene_id for f in found] == [held[0], held[1], held[0]]
    assert [f.unseen_background for f in found] == [False, True, False]
    assert sorted(asked) == sorted([*held, *trained])
