"""The ``final-table`` service: the active-set preflight and the table of the final run."""

from __future__ import annotations

import json
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.services import excluded_checks
from physverify.baselines.settings import load_vlm_comparison
from physverify.calibration.apply import unweighted_abstentions
from physverify.calibration.artifact_io import load_calibration
from physverify.core.atomic import write_text_atomic
from physverify.core.output import Echo
from physverify.human_agreement.settings import agreement_settings
from physverify.reporting import final_comparison as comparison
from physverify.reporting.final_agreement import Bootstrap
from physverify.reporting.final_comparison import ComparisonInputs
from physverify.reporting.final_rows import Criteria, input_gaps_over, load_evaluation
from physverify.reporting.final_table import (
    Inputs,
    active_checks,
    as_json,
    build_rows,
    markdown,
    supplementary,
)

__all__ = [
    "ComparisonInputs",
    "Inputs",
    "check_inputs",
    "write_comparison_table",
    "write_final_table",
    "write_preflight",
]

TABLE_MD, TABLE_JSON = "final_table.md", "final_table.json"
COMPARISON_MD, COMPARISON_JSON = "comparison_table.md", "comparison_table.json"


def write_preflight(artifact: Path, out: Path, echo: Echo) -> bool:
    """Per stratum, the active set (weight > 0) and the weight-0 checks whose abstention would
    change a verdict; True when there is none."""
    calibration = load_calibration(artifact)
    strata = {}
    for level, profile in calibration.strata():
        stratum = calibration.stratum(level, profile)
        checks = sorted(stratum.model.checks)
        strata[f"{level.value}/{profile}"] = {
            "active_set": [c for c in checks if stratum.weight(c) > 0.0],
            "unweighted_abstentions": list(unweighted_abstentions(stratum)),
        }
    perceived = next((v for k, v in strata.items() if k.startswith("perceived/")), None)
    found = {"artifact": str(artifact), "strata": strata, **(perceived or {})}
    write_text_atomic(out, json.dumps(found, indent=2, sort_keys=True) + "\n")
    leaking = [c for v in strata.values() for c in v["unweighted_abstentions"]]
    echo(f"preflight: {out} ({len(leaking)} weight-0 checks change a verdict)")
    return not leaking


def check_inputs(evaluation: Path, preflight: Path, echo: Echo) -> bool:
    """True when no active check of the evaluation lacks its input on more than the criteria's
    share of a set's clips; each one that does is echoed."""
    active = active_checks(json.loads(preflight.read_text(encoding="utf-8")))
    limit = Criteria().max_missing_input
    over = input_gaps_over(load_evaluation(evaluation), active, limit)
    for line in over:
        echo(f"missing input: {line} (> {limit})")
    return not over


def write_final_table(context: AppContext, inputs: Inputs, out: Path, echo: Echo) -> None:
    agreement = agreement_settings(context.settings)
    boot = Bootstrap(agreement.bootstrap_n, agreement.bootstrap_seed, agreement.ci)
    rows = build_rows(inputs, boot, Criteria(), excluded=excluded_checks(context))
    extra = supplementary(inputs)
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / TABLE_MD, markdown(rows, extra))
    write_text_atomic(out / TABLE_JSON, as_json(rows, extra))
    counts = {s: sum(r.status == s for r in rows) for s in ("pass", "fail", "not_evaluable")}
    echo(f"final table: {out / TABLE_MD} ({counts})")


def write_comparison_table(
    context: AppContext, inputs: ComparisonInputs, out: Path, echo: Echo
) -> None:
    """The comparison table from the rows the final run wrote; a missing input leaves its rows not
    evaluable."""
    lines = comparison.comparison_lines(inputs, load_vlm_comparison(context.settings))
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / COMPARISON_MD, comparison.markdown(lines))
    write_text_atomic(out / COMPARISON_JSON, comparison.as_json(lines, inputs))
    echo(f"comparison table: {out / COMPARISON_MD}")
