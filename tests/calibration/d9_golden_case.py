"""The fit behind the D9 golden file: two levels, R1 sigma pairs half 0-against-0 (it1 audit d1).

Kept free of it1 names so the same module also runs on the code before it1, which wrote
``golden/d9_all_before_it1.json`` (commit 775bde5, ``write_golden``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import test_devscore as td
from calibration_cases import CHECKS, config, fit_meta

from physverify.calibration.collect import level_data
from physverify.calibration.devscore import dev_score
from physverify.calibration.fit import FitInputs, fit_calibration
from physverify.calibration.sigma import SigmaPair
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split

GOLDEN = Path(__file__).resolve().parent / "golden" / "d9_all_before_it1.json"


def pairs() -> tuple[SigmaPair, ...]:
    rng = np.random.default_rng(7)
    found: list[SigmaPair] = []
    for scene in sorted({r.scene_id for r in td.levels()[0].rows}):
        found += [SigmaPair(scene, "R1", 0.0, 0.0, 0.01 + 0.001 * k) for k in range(2)]
        for _ in range(2):
            oracle = float(abs(rng.normal(0.05, 0.02)))
            found.append(
                SigmaPair(scene, "R1", oracle, oracle + float(rng.normal(0.0, 0.01)), 0.004)
            )
    return tuple(found)


def fitted(**update: Any) -> dict[str, Any]:
    """``to_json()`` of the two-level fit and of its devscore report, as parsed JSON."""
    cfg = config(alpha=0.1, folds=3, bootstrap=30, **update)
    oracle = level_data(InputLevel.ORACLE, td.REPORT, CHECKS, cfg, split=Split.CALIB)
    perceived = level_data(
        InputLevel.PERCEIVED, td.REPORT, CHECKS, cfg, split=Split.CALIB, pairs=pairs()
    )
    artifact = fit_calibration(FitInputs((oracle, perceived), CHECKS, fit_meta()), cfg)
    report = dev_score((oracle, perceived), CHECKS, fit_meta(), cfg).report
    return {"two_level": json.loads(artifact.to_json()), "devscore": json.loads(report.to_json())}


def write_golden() -> None:
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(fitted(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(write_golden())
