"""contacts/proximity tables of the fixture and of analytic scenes, written once by the code
before it3 (commit 2a762ad) into ``golden/contacts_proximity/<case>.json`` and compared after
the move of the proximity helpers into ``_proximity.py`` (it3 T1). The same module runs in both
trees:

    PYTHONPATH=<old>/src:tests/fixtures:tests/extraction python tests/extraction/contacts_golden_case.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import contact_scenes as cs  # noqa: E402
import check_scenarios as sc  # noqa: E402
from synthetic_clip import build_clip  # noqa: E402

from physverify.config.models import RuntimeSettings  # noqa: E402
from physverify.extraction.matching import align_ground_truth  # noqa: E402
from physverify.extraction.providers.contacts_proximity import ProximityContacts  # noqa: E402
from physverify.schema.contacts import ContactTable  # noqa: E402
from physverify.schema.stages import Stage  # noqa: E402
from physverify.schema.states import ObjectStateTable  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "contacts_proximity"


@dataclass(frozen=True)
class StatesOnly:
    """The part of a StageContext that a KINEMATICS-only provider reads."""

    states: ObjectStateTable

    def require(self, stage: Stage, kind: type[ObjectStateTable]) -> ObjectStateTable:
        assert stage is Stage.KINEMATICS and isinstance(self.states, kind)
        return self.states


def fixture_states() -> ObjectStateTable:
    clip = build_clip(RuntimeSettings())
    from support import sampling_of

    sampling = sampling_of(clip.manifest)
    truth = clip.ground_truth
    return align_ground_truth(truth, truth.masks(sampling), sampling).states


def unknown_orientation(states: ObjectStateTable) -> ObjectStateTable:
    return replace(states, quaternion=np.full_like(states.quaternion, np.nan))


CASES: dict[str, Callable[[], ObjectStateTable]] = {
    "fixture": fixture_states,
    "fixture_unknown_orientation": lambda: unknown_orientation(fixture_states()),
    "stack": lambda: sc.stack(pitch_deg=25.0).states,
    "stack_unknown_orientation": lambda: unknown_orientation(sc.stack(pitch_deg=25.0).states),
    "bounce_on_floor": lambda: cs.bounce_on_floor().states,
    "bounce_box_on_floor": lambda: cs.bounce_on_floor(kind=sc.PrimitiveKind.BOX).states,
    "bounce_on_floor_noisy": lambda: cs.jittered(cs.bounce_on_floor().states, 0.01),
    "late_bounce_on_floor": lambda: cs.bounce_on_floor(e=0.5, impact_frame=25.3, speed=7.0).states,
    "collision_over_floor": lambda: cs.collision_over_floor(mass_ratio=3.0, gain=0.5).states,
    "rocking_box": lambda: cs.rocking_case().states,
    "rocking_box_noisy": lambda: cs.jittered(cs.rocking_case().states, 0.002),
    "rolling_ball": lambda: sc.rolling_ball(pitch_deg=25.0, speed=0.01).states,
    "teleport": lambda: sc.teleport().states,
}


def proximity_table(states: ObjectStateTable) -> ContactTable:
    provider = ProximityContacts(ProximityContacts.Params())
    table = provider.run(StatesOnly(states))  # type: ignore[arg-type]
    assert isinstance(table, ContactTable)
    return table


def main(out: Path = GOLDEN) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, states in CASES.items():
        proximity_table(states()).save_json(out / f"{name}.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN)
