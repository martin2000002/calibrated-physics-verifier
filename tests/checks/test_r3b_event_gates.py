from dataclasses import replace
from typing import Any, ClassVar

import check_scenarios as sc
import numpy as np
import pytest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks.variants._image_impulse import Candidate, Gates, detect
from physverify.checks.variants.event_gates._base import EVENT_GATES, build_gates
from physverify.checks.variants.event_gates.reciprocal_jump import ReciprocalJump
from physverify.checks.variants.r3b_image_relspeed import RelSpeedCheck, RelSpeedParams
from physverify.core.errors import ConfigError
from physverify.schema.checks import InputLevel
from physverify.schema.states import ObjectStateTable

GATED = RelSpeedParams(event_gates=("reciprocal_jump",))


def detected(case: sc.Case, params: RelSpeedParams) -> list[tuple[str, str, float]]:
    found = detect(case.states, case.contacts, params.gates(), RelSpeedCheck(params).event_gates)
    return [(e.a, e.b, e.t) for e in found]


def same_side_kink() -> sc.Case:
    """A light ball reverses off a heavy one whose small velocity change is turned to point the
    same way as the light ball's: the relative track still kinks, the jumps are not opposite."""
    return sc.collision(mass_ratio=0.1, skew_deg=180.0)


def test_opposite_jumps_keep_the_event() -> None:
    case = sc.collision(gain=0.2)
    assert detected(case, GATED) == detected(case, RelSpeedParams())
    assert len(detected(case, GATED)) == 1


def test_a_same_side_kink_is_an_event_only_without_the_gate() -> None:
    case = same_side_kink()
    assert len(detected(case, RelSpeedParams())) == 1
    assert detected(case, GATED) == []


def test_a_body_without_a_jump_is_rejected() -> None:
    case = sc.collision(gain=0.2)
    position = np.array(case.states.position)
    position[:, 1] = [0.0, 0.0, 1.0]
    still = replace(case.states, position=position)
    event = Candidate((0, 1), 5, 5.0, 10.0, (2, 3, 4), (6, 7, 8))
    assert not ReciprocalJump().accepts(still, event, GATED.gates())


class AcceptAll:
    name: ClassVar[str] = "accept_all"

    def accepts(self, states: ObjectStateTable, event: Candidate, gates: Gates) -> bool:
        return True


def outputs(check: RelSpeedCheck, case: sc.Case) -> list[tuple[Any, ...]]:
    results = check.evaluate(case.inputs(InputLevel.PERCEIVED, {}))
    return [(r.subject, r.magnitude, r.sigma, r.verdict, r.details) for r in results]


@pytest.mark.parametrize("case", [sc.collision(gain=0.2), same_side_kink()], ids=["two", "one"])
def test_no_gate_and_an_accepting_gate_leave_every_result_unchanged(case: sc.Case) -> None:
    plain = outputs(RelSpeedCheck(RelSpeedParams()), case)
    passing = RelSpeedParams(event_gates=("accept_all",))
    assert plain
    assert outputs(RelSpeedCheck(passing, {"accept_all": AcceptAll}), case) == plain
    assert RelSpeedCheck(RelSpeedParams()).event_gates == ()


def test_gates_are_registered_and_unknown_names_fail() -> None:
    registry = PLUGIN_KINDS[EVENT_GATES.name]()
    assert registry["reciprocal_jump"] is ReciprocalJump
    with pytest.raises(ConfigError, match="unknown"):
        build_gates(("nope",), registry)
    with pytest.raises(ConfigError, match="unknown"):
        RelSpeedCheck(RelSpeedParams(event_gates=("nope",)))


def test_the_gate_registry_is_injectable() -> None:
    check = RelSpeedCheck(GATED, {"reciprocal_jump": ReciprocalJump})
    assert [type(g) for g in check.event_gates] == [ReciprocalJump]
