import ast
import inspect
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, ClassVar, get_type_hints

import check_scenarios as sc
import numpy as np
import pytest

import physverify.checks.variants.event_scopes as scopes_package
from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks.variants._image_impulse import Event
from physverify.checks.variants.event_scopes._base import (
    EVENT_SCOPES,
    EventScope,
    ScopeLimits,
    SidedEvent,
    build_scopes,
)
from physverify.checks.variants.event_scopes.two_sided_spheres_isolated import (
    TwoSidedSpheresIsolated,
)
from physverify.checks.variants.r3b_image_relspeed import RelSpeedCheck, RelSpeedParams
from physverify.core.errors import ConfigError
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.verification.runner import RowStatus, aggregate

S3 = "two_sided_spheres_isolated"
SCOPED = RelSpeedParams(event_scopes=(S3,))
LIMITS = RelSpeedParams().scope_limits()
TRUTH = {Stage.CONTACTS.value: GROUND_TRUTH}


def evaluated(params: RelSpeedParams, case: sc.Case, truth: bool = False) -> list[CheckResult]:
    level = InputLevel.ORACLE if truth else InputLevel.PERCEIVED
    return RelSpeedCheck(params).evaluate(case.inputs(level, TRUTH if truth else {}))


def outputs(results: Sequence[CheckResult]) -> list[tuple[Any, ...]]:
    return [
        (r.subject, r.magnitude, r.sigma, r.verdict, r.abstain_reason, r.details) for r in results
    ]


def sided(t: float = 10.0, left: int = 3, right: int = 3) -> SidedEvent:
    return SidedEvent(Event("ball_i", "ball_j", t), tuple(range(left)), tuple(range(right)))


def states(kind: PrimitiveKind = PrimitiveKind.SPHERE, step: int = 1) -> ObjectStateTable:
    table = sc.collision(kind=kind).states
    return replace(table, frame_index=np.asarray(table.frame_index) * step)


def test_the_limits_come_from_the_check_params() -> None:
    assert (
        ScopeLimits(side_min_rows=3, side_max_rows=6, gap_frames=1.0, bound_margin_rows=1) == LIMITS
    )
    assert LIMITS.reach_frames(1) == 8.0
    assert LIMITS.reach_frames(2) == 15.0


@pytest.mark.parametrize(
    ("left", "right", "inside"), [(3, 3, True), (6, 3, True), (2, 6, False), (6, 2, False)]
)
def test_both_sides_need_the_rows_of_a_line_fit(left: int, right: int, inside: bool) -> None:
    assert (
        TwoSidedSpheresIsolated().applies(states(), sided(left=left, right=right), [], LIMITS)
        is inside
    )


@pytest.mark.parametrize("kind", [PrimitiveKind.BOX, PrimitiveKind.CYLINDER, PrimitiveKind.UNKNOWN])
def test_both_bodies_must_be_spheres(kind: PrimitiveKind) -> None:
    assert TwoSidedSpheresIsolated().applies(states(), sided(), [], LIMITS)
    assert not TwoSidedSpheresIsolated().applies(states(kind), sided(), [], LIMITS)


def test_every_failed_predicate_is_named() -> None:
    scope, crowded = TwoSidedSpheresIsolated(), [Event("ball_i", "floor_x", 12.0)]
    assert scope.failed(states(), sided(), [], LIMITS) == ()
    failed = scope.failed(states(PrimitiveKind.BOX), sided(left=2), crowded, LIMITS)
    assert failed == ("two_sided", "spheres", "isolated")


@pytest.mark.parametrize(
    ("step", "other_t", "inside"),
    [
        (1, 18.0, False),
        (1, 2.0, False),
        (1, 18.5, True),
        (1, 1.5, True),
        (2, 25.0, False),
        (2, 25.5, True),
    ],
)
def test_no_other_event_within_the_reach_of_a_side_window(
    step: int, other_t: float, inside: bool
) -> None:
    other = Event("ball_i", "floor_x", other_t)
    applies = TwoSidedSpheresIsolated().applies(states(step=step), sided(t=10.0), [other], LIMITS)
    assert applies is inside


@pytest.mark.parametrize("truth", [False, True], ids=["perceived", "truth"])
def test_an_event_in_scope_is_unchanged(truth: bool) -> None:
    case = sc.collision(gain=0.2)
    plain = evaluated(RelSpeedParams(), case, truth)
    assert len(plain) == 1 and plain[0].magnitude is not None
    assert outputs(evaluated(SCOPED, case, truth)) == outputs(plain)


@pytest.mark.parametrize("truth", [False, True], ids=["perceived", "truth"])
def test_an_event_outside_the_scope_is_not_applicable_and_keeps_its_details(truth: bool) -> None:
    case = sc.collision(gain=0.2, kind=PrimitiveKind.BOX)
    (plain,) = evaluated(RelSpeedParams(), case, truth)
    (scoped,) = evaluated(SCOPED, case, truth)
    assert plain.magnitude is not None
    assert scoped.verdict is Verdict.ABSTAIN and scoped.magnitude is None
    assert scoped.abstain_reason is AbstainReason.NOT_APPLICABLE
    assert scoped.details == {
        **plain.details,
        "not_applicable": S3,
        "not_applicable_predicates": ["spheres"],
    }
    assert scoped.subject == plain.subject and scoped.frame_span == plain.frame_span
    assert aggregate(scoped.clip_id, RelSpeedCheck(SCOPED), [scoped]).status is RowStatus.NO_EVENT


class Recording:
    name: ClassVar[str] = "recording"
    seen: ClassVar[list[tuple[Any, ...]]] = []

    def applies(
        self,
        states: ObjectStateTable,
        event: SidedEvent,
        others: Sequence[Event],
        gates: ScopeLimits,
    ) -> bool:
        return not self.failed(states, event, others, gates)

    def failed(
        self,
        states: ObjectStateTable,
        event: SidedEvent,
        others: Sequence[Event],
        gates: ScopeLimits,
    ) -> tuple[str, ...]:
        Recording.seen.append((states, event, list(others), gates))
        return ("recorded",)


def test_the_scope_receives_only_states_the_event_its_sides_and_the_other_events() -> None:
    case = sc.collision(gain=0.2)
    Recording.seen.clear()
    check = RelSpeedCheck(
        RelSpeedParams(event_scopes=("recording",)), scope_registry={"recording": Recording}
    )
    (result,) = check.evaluate(case.inputs(InputLevel.PERCEIVED, {}))
    ((received_states, event, others, gates),) = Recording.seen
    assert received_states is case.states
    assert (event.event.a, event.event.b) == result.subject and others == []
    assert (len(event.left), len(event.right)) == (
        result.details["rows_before"],
        result.details["rows_after"],
    )
    assert gates == LIMITS
    assert result.details["not_applicable"] == "recording"
    assert result.details["not_applicable_predicates"] == ["recorded"]


def test_the_scope_signature_carries_no_role_perturbation_or_manifest() -> None:
    inputs = {
        "states": ObjectStateTable,
        "event": SidedEvent,
        "others": Sequence[Event],
        "gates": ScopeLimits,
    }
    for method, returned in (("applies", bool), ("failed", tuple[str, ...])):
        params = list(inspect.signature(getattr(EventScope, method)).parameters)
        assert params == ["self", "states", "event", "others", "gates"]
        hints = get_type_hints(getattr(TwoSidedSpheresIsolated, method))
        assert hints == {**inputs, "return": returned}
    assert [f.name for f in fields(SidedEvent)] == ["event", "left", "right"]
    assert [f.name for f in fields(Event)] == ["a", "b", "t", "z", "normal", "point"]


def test_the_scope_modules_import_no_truth_dataset_or_label_code() -> None:
    banned = (
        "physverify.synth",
        "physverify.datasets",
        "physverify.calibration",
        "physverify.verification",
    )
    for path in Path(scopes_package.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not [m for m in imported if m.startswith(banned)], path.name


def test_scopes_are_registered_injectable_and_unknown_names_fail() -> None:
    registry = PLUGIN_KINDS[EVENT_SCOPES.name]()
    assert registry[S3] is TwoSidedSpheresIsolated
    assert RelSpeedCheck(RelSpeedParams()).event_scopes == ()
    assert [
        type(s)
        for s in RelSpeedCheck(SCOPED, scope_registry={S3: TwoSidedSpheresIsolated}).event_scopes
    ] == [TwoSidedSpheresIsolated]
    with pytest.raises(ConfigError, match="unknown"):
        build_scopes(("nope",), registry)
    with pytest.raises(ConfigError, match="unknown"):
        RelSpeedCheck(RelSpeedParams(event_scopes=("nope",)))
