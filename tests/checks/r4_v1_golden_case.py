"""R4_jump and R4_size results of the v1 code (commit 10e37ff, before it4) on analytic scenes,
written once into ``golden/r4_v1.json``. After it4 the v1 jump (``ContinuityCheck``, out of the
registry) and R4_size with ``reference = "self"`` must reproduce them (T8, T14). To regenerate
the golden, run this module with the old ``src`` and the ``tests/checks`` of commit 3775138 (the
old tree has no ``reference`` parameter, so it is passed only when given); today's
``check_scenarios.py`` gives ``CheckInputs`` a camera, which the old ``src`` does not know:

    git archive 10e37ff src | tar -x -C OLD; git archive 3775138 tests/checks | tar -x -C OLD
    PYTHONPATH=OLD/src:OLD/tests/checks python OLD/tests/checks/r4_v1_golden_case.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import check_scenarios as sc
from reference.r4_jump_v1 import ContinuityCheck

from physverify.checks._params import CommonParams
from physverify.checks.r4_integrity import SizeConstancyCheck, SizeParams
from physverify.schema.io import json_ready

GOLDEN = Path(__file__).resolve().parent / "golden" / "r4_v1.json"

SCENES: dict[str, Callable[[], sc.Case]] = {
    "teleport_rest": lambda: sc.teleport(),
    "teleport_flight": lambda: sc.teleport(flight=True, pitch_deg=25.0),
    "teleport_strided": lambda: sc.strided(sc.teleport(delta=0.2), 2, quantize=False),
    "teleport_noisy": lambda: sc.with_sigma(sc.teleport(), position=0.01),
    "parabola": lambda: sc.parabola(pitch_deg=25.0),
    "gravity_switch": lambda: sc.parabola(g_after=4.0),
    "bounce": lambda: sc.bounce(pitch_deg=25.0),
    "collision": lambda: sc.collision(mass_ratio=3.0),
    "size_change": lambda: sc.size_change(),
    "size_change_sigma": lambda: sc.with_sigma(sc.size_change(factor=1.2), size=0.005),
    "stack": lambda: sc.stack(pitch_deg=25.0),
    "box_sphere": lambda: sc.box_sphere(),
    "overlapping_spheres": lambda: sc.overlapping_spheres(),
    "disappearance": lambda: sc.disappearance(),
}


def results(size_params: dict[str, Any] | None = None) -> dict[str, list[Any]]:
    """Scene -> the v1 R4_jump results followed by the R4_size results, as plain JSON."""
    jump = ContinuityCheck(CommonParams())
    size = SizeConstancyCheck(SizeParams(**(size_params or {})))
    out: dict[str, list[Any]] = {}
    for name, scene in SCENES.items():
        inputs = scene().inputs()
        found = [*jump.evaluate(inputs), *size.evaluate(inputs)]
        out[name] = [json_ready(r.to_dict()) for r in found]
    return json.loads(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
