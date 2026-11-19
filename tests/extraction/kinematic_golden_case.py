"""contacts/kinematic tables of analytic scenes, written once by the code of commit 10e37ff
into ``golden/contacts_kinematic/<case>.json`` and compared after the move of the pixel floor
and the noise scale into ``physics/tracks.py``. The same module runs in both trees:

    PYTHONPATH=<old>/src:tests/fixtures:tests/extraction:tests/checks \
        python tests/extraction/kinematic_golden_case.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import contact_scenes as cs

from physverify.schema.contacts import ContactTable

sc = cs.sc
GOLDEN = Path(__file__).resolve().parent / "golden" / "contacts_kinematic"


def noisy(case: sc.Case, sigma: float) -> ContactTable:
    return cs.kinematic_table(case, states=cs.jittered(case.states, sigma))


CASES: dict[str, Callable[[], ContactTable]] = {
    "stack": lambda: cs.kinematic_table(sc.stack(pitch_deg=25.0)),
    "bounce_on_floor": lambda: cs.kinematic_table(cs.bounce_on_floor()),
    "bounce_box_on_floor": lambda: cs.kinematic_table(
        cs.bounce_on_floor(kind=sc.PrimitiveKind.BOX)
    ),
    "bounce_on_floor_noisy": lambda: noisy(cs.bounce_on_floor(), 0.01),
    "late_bounce_on_floor": lambda: cs.kinematic_table(
        cs.bounce_on_floor(e=0.5, impact_frame=25.3, speed=7.0)
    ),
    "collision_over_floor": lambda: cs.kinematic_table(
        cs.collision_over_floor(mass_ratio=3.0, gain=0.5)
    ),
    "rocking_box": lambda: cs.kinematic_table(cs.rocking_case()),
    "rocking_box_noisy": lambda: noisy(cs.rocking_case(), 0.002),
    "rolling_ball": lambda: cs.kinematic_table(sc.rolling_ball(pitch_deg=25.0, speed=0.01)),
    "teleport": lambda: cs.kinematic_table(sc.teleport()),
}


def main(out: Path = GOLDEN) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, table in CASES.items():
        table().save_json(out / f"{name}.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN)
