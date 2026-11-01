"""R5a v1 results (commit 5140a4e) on analytic scenes, written once into
``golden/r5a_v1.json``: the registered ``FallShapeCheck`` must keep reproducing them after the
statistic was made a method that its variants override. The exact scenes are degenerate (every
candidate scores ~1e-14, so the chosen one is float noise); the jittered ones carry deterministic
position noise so that the magnitude and the chosen candidate are both well defined. To
regenerate the golden from the old tree:

    git archive 5140a4e src | tar -x -C OLD
    PYTHONPATH=OLD/src:tests/checks python tests/checks/r5a_v1_golden_case.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import check_scenarios as sc

from physverify.checks.r5a_fall_shape import FallShapeCheck, FallShapeParams
from physverify.schema.constants import STANDARD_GRAVITY
from physverify.schema.io import json_ready

GOLDEN = Path(__file__).resolve().parent / "golden" / "r5a_v1.json"
G = STANDARD_GRAVITY
JITTER = 0.0002


SCENES: dict[str, Callable[[], sc.Case]] = {
    "parabola": lambda: sc.parabola(pitch_deg=25.0),
    "gravity_switch": lambda: sc.parabola(g_after=1.3 * G, change_frame=18),
    "gravity_switch_sigma": lambda: sc.with_sigma(
        sc.parabola(g_after=1.3 * G, change_frame=18), position=0.002
    ),
    "gravity_switch_strided": lambda: sc.strided(sc.parabola(g_after=1.3 * G), 2, quantize=False),
    "bounce": lambda: sc.bounce(pitch_deg=25.0),
    "bounce_sigma": lambda: sc.with_sigma(sc.bounce(e=0.7), position=0.003),
    "teleport_flight": lambda: sc.teleport(flight=True, pitch_deg=25.0),
    "two_balls": lambda: sc.two_balls(1.2 * G),
    "two_balls_sigma": lambda: sc.with_sigma(sc.two_balls(0.8 * G), position=0.002),
    "parabola_jittered": lambda: sc.jittered(sc.parabola(pitch_deg=25.0), JITTER, seed=0),
    "gravity_switch_jittered": lambda: sc.jittered(
        sc.with_sigma(sc.parabola(g_after=1.3 * G, change_frame=18), position=0.002), JITTER, seed=1
    ),
    "bounce_jittered": lambda: sc.jittered(sc.bounce(e=0.7, pitch_deg=25.0), JITTER, seed=2),
    "two_balls_jittered": lambda: sc.jittered(sc.two_balls(1.2 * G), JITTER, seed=3),
}


def results() -> dict[str, list[Any]]:
    """Scene -> the R5a results of the registered check with default parameters, as plain JSON."""
    check = FallShapeCheck(FallShapeParams())
    out = {
        name: [json_ready(r.to_dict()) for r in check.evaluate(sc.bare_inputs(scene()))]
        for name, scene in SCENES.items()
    }
    return json.loads(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
