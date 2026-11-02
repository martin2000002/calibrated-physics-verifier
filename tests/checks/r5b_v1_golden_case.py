"""R5b v1 results (commit 4671f02) on analytic scenes, written once into
``golden/r5b_v1.json``: the registered ``GravityValueCheck`` must keep reproducing them exactly
after its flight search was made a method that its variants override. To regenerate the golden
from the old tree:

    git archive 4671f02 src | tar -x -C OLD
    PYTHONPATH=OLD/src:tests/checks python tests/checks/r5b_v1_golden_case.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import check_scenarios as sc

from physverify.checks.r5b_gravity_value import GravityValueCheck, GravityValueParams
from physverify.schema.constants import STANDARD_GRAVITY
from physverify.schema.io import json_ready

GOLDEN = Path(__file__).resolve().parent / "golden" / "r5b_v1.json"
G = STANDARD_GRAVITY
JITTER = 0.0002

SCENES: dict[str, Callable[[], sc.Case]] = {
    "parabola": lambda: sc.parabola(pitch_deg=25.0),
    "parabola_weak": lambda: sc.parabola(g=0.6 * G),
    "parabola_strided": lambda: sc.strided(sc.parabola(g=1.2 * G), 2, quantize=True),
    "bounce": lambda: sc.bounce(pitch_deg=25.0),
    "bounce_sigma": lambda: sc.with_sigma(sc.bounce(e=0.7), position=0.003),
    "two_balls": lambda: sc.two_balls(1.2 * G),
    "parabola_jittered": lambda: sc.jittered(sc.parabola(pitch_deg=25.0), JITTER, seed=0),
    "bounce_jittered": lambda: sc.jittered(sc.bounce(e=0.7, pitch_deg=25.0), JITTER, seed=2),
    "unmarked_roll": sc.unmarked_roll,
    "gliding_ball": sc.gliding_ball,
    "flight_beside_box": sc.flight_beside_box,
}


def results() -> dict[str, list[Any]]:
    """Scene -> the R5b results of the registered check with default parameters, as plain JSON."""
    check = GravityValueCheck(GravityValueParams())
    out = {
        name: [json_ready(r.to_dict()) for r in check.evaluate(sc.bare_inputs(scene()))]
        for name, scene in SCENES.items()
    }
    return json.loads(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
