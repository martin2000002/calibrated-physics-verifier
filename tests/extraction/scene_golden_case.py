"""contacts/kinematic_scene tables of analytic scenes seen through disc masks.

``golden/contacts_kinematic_scene/v2`` holds the tables of commit ff171fd (version 2: 3D velocity
rest, gap gate) and ``v3`` those of version 3 with its defaults (image rest, 3D velocity impacts,
proximity pair rest); version 4 with the 3D velocity impacts and the proximity pair rest must
reproduce both, with ``rest="velocity"`` and ``env_gap_gate=True`` for version 2.
``golden/contacts_kinematic_scene/v4`` holds version 4 with its defaults. The same module writes
them all, each from its own tree:

    PYTHONPATH=<tree>/src:tests/fixtures:tests/extraction:tests/checks \
        python tests/extraction/scene_golden_case.py <out>
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import contact_scenes as cs
from support import entry

from physverify.extraction.providers.contacts_kinematic_scene import KinematicSceneContacts
from physverify.schema.clip import SamplingSpec, resolve_sampling
from physverify.schema.contacts import ContactTable
from physverify.schema.stages import Stage
from physverify.schema.states import PrimitiveKind

sc = cs.sc
GOLDEN = Path(__file__).resolve().parent / "golden" / "contacts_kinematic_scene"
ENTRIES = (
    entry(Stage.TRACK, "edgetam", "centroid_px_rel", 0.3526),
    entry(Stage.LIFT, "backproject", "centroid_rel", 0.0552, bias=0.0002),
    entry(Stage.LIFT, "backproject", "size_rel", 0.0782, bias=0.0056),
    entry(Stage.LIFT, "backproject", "support_gap_rel", 0.1),
    entry(Stage.KINEMATICS, "savgol", "vel_rel", 0.3157, bias=-0.0022),
)
V3_EQUIVALENT = {"impacts": "velocity", "pair_rest": "proximity"}
V2_EQUIVALENT = {**V3_EQUIVALENT, "rest": "velocity", "env_gap_gate": True}


def table(case: sc.Case, **params: Any) -> ContactTable:
    sampling = resolve_sampling(case.clip, SamplingSpec())
    ctx = cs.StubContext(
        upstream={
            Stage.TRACK: cs.disk_track(case.states, sampling),
            Stage.KINEMATICS: case.states,
            Stage.CAMERA: cs.scene_camera(case),
        },
        sampling=sampling,
        entries={(e.stage, e.quantity): e for e in ENTRIES},
    )
    provider = KinematicSceneContacts(KinematicSceneContacts.Params(**params))
    found = provider.run(ctx)  # type: ignore[arg-type]
    assert isinstance(found, ContactTable)
    return found


def boxes(bottoms: list[float]) -> sc.Case:
    side = cs.BOX_SIDE
    objects = [
        sc.still(f"box{i}", PrimitiveKind.BOX, sc.cube(side), (2 * side * i, 0.0, b + side / 2))
        for i, b in enumerate(bottoms)
    ]
    return sc.build("boxes", objects, [], 25.0)


def stack_beside_a_floor_box() -> sc.Case:
    side = cs.BOX_SIDE
    stack = [
        sc.still("box0", PrimitiveKind.BOX, sc.cube(side), (0.0, 0.0, side / 2)),
        sc.still("box1", PrimitiveKind.BOX, sc.cube(side), (0.0, 0.0, 1.5 * side)),
        sc.still("box2", PrimitiveKind.BOX, sc.cube(side), (6 * side, 0.0, side / 2)),
    ]
    return sc.build("stack", stack, [], 25.0)


CASES: dict[str, Callable[[], sc.Case]] = {
    "bounce_on_floor": cs.bounce_on_floor,
    "late_bounce_on_floor": lambda: cs.bounce_on_floor(e=0.5, impact_frame=25.3, speed=7.0),
    "collision_over_floor": lambda: cs.collision_over_floor(mass_ratio=3.0, gain=0.5),
    "rocking_box": cs.rocking_case,
    "boxes_on_the_floor": lambda: boxes([0.0, 0.0]),
    "box_in_the_air": lambda: boxes([0.0, 0.0, 0.2]),
    "stack_beside_a_floor_box": stack_beside_a_floor_box,
}


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, case in CASES.items():
        table(case()).save_json(out / f"{name}.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
