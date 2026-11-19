"""The final table: one row per closing and coverage criterion of the thesis plan, each with its
value, interval, threshold, status and source file."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from physverify.calibration.artifact import Auroc, EvaluationReport, EvaluationStratum
from physverify.core.jsonio import read_json
from physverify.core.markdown import table
from physverify.reporting import final_agreement as agreement
from physverify.reporting.final_coverage import coverage_rows
from physverify.reporting.final_rows import (
    DESCRIPTIVE,
    FAIL,
    NOT_EVALUABLE,
    OUTSIDE_RUN,
    PASS,
    Criteria,
    Row,
    Status,
    ci_text,
    counted,
    fmt,
    input_gaps_over,
    load_evaluation,
    missing,
    stratum_at,
)
from physverify.selective_risk.records import CurveRecord


@dataclass(frozen=True)
class Inputs:
    """Paths of the final run's outputs; None where the run did not produce one."""

    freeze: Path | None
    kubric: Path | None
    curve: Path | None
    agreement_items: Path | None
    preflight: Path | None
    real: Path | None = None
    """The evaluation of the sealed real sets (R2); the Kubric one when None."""
    attribution: Path | None = None
    r3b: Path | None = None
    replications: Mapping[str, Path] = field(default_factory=dict)
    dry_run: bool = False
    """A rehearsal: it freezes nothing, so a missing freeze.json is skipped, not missing."""


def _met_row(id_: str, criterion: str, stratum: EvaluationStratum, need: int, src: str) -> Row:
    met = counted(stratum)
    status = PASS if len(met) >= need else FAIL
    note = "meets: " + (", ".join(met) or "none")
    total = len(stratum.checks)
    return Row(id_, criterion, f"{len(met)}/{total}", "", f">= {need}/{total}", status, src, note)


def clause_rows(
    report: EvaluationReport | None, attribution: Any, c: Criteria, src: str
) -> list[Row]:
    names = (
        ("R1.oracle", "oracle: checks meeting the stratified clause", f">= {c.oracle_met}/9"),
        (
            "R1.perceived_goal",
            "perceived: checks meeting the clause (goal)",
            f">= {c.perceived_goal}/9",
        ),
        (
            "R1.perceived_min",
            "perceived: checks meeting the clause (minimum)",
            f">= {c.perceived_minimum}/9",
        ),
    )
    rows: list[Row] = []
    for (id_, text, threshold), level, need in zip(
        names,
        ("oracle", "perceived", "perceived"),
        (c.oracle_met, c.perceived_goal, c.perceived_minimum),
        strict=True,
    ):
        stratum = None if report is None else stratum_at(report, level)
        rows.append(
            missing(id_, text, threshold, f"{level} evaluation")
            if stratum is None
            else _met_row(id_, text, stratum, need, src)
        )
    perceived = None if report is None else stratum_at(report, "perceived")
    rows.append(auroc_row(perceived, c, src))
    rows.append(auroc_s_row(perceived, src))
    rows.append(_attribution_row(perceived, attribution))
    return rows


def auroc_row(perceived: EvaluationStratum | None, c: Criteria, src: str) -> Row:
    """The aggregate AUROC of the verifier's graded score B1, the one the hypothesis judges."""
    text, threshold = "perceived: aggregate AUROC of B1", f">= {c.perceived_auroc}"
    found = None if perceived is None else perceived.auroc_graded
    if found is None or found.value is None:
        return missing("R1.perceived_auroc", text, threshold, "held-out aggregate AUROC of B1")
    status = PASS if found.value >= c.perceived_auroc else FAIL
    return _auroc_line("R1.perceived_auroc", text, found, threshold, status, src)


def auroc_s_row(perceived: EvaluationStratum | None, src: str) -> Row:
    """The same AUROC of the calibrated score 1 - S, printed beside it and judged by none."""
    text = "perceived: aggregate AUROC of 1 - S (descriptive)"
    found = None if perceived is None else perceived.auroc
    if found is None or found.value is None:
        return missing("R1.perceived_auroc_s", text, "", "held-out aggregate AUROC of 1 - S")
    return _auroc_line("R1.perceived_auroc_s", text, found, "", DESCRIPTIVE, src)


def _auroc_line(id_: str, text: str, found: Auroc, threshold: str, status: Status, src: str) -> Row:
    return Row(
        id_,
        text,
        fmt(found.value),
        ci_text(found.ci.low, found.ci.high),
        threshold,
        status,
        src,
        f"{found.n_positive} perturbed, {found.n_negative} clean clips",
    )


def _attribution_row(perceived: EvaluationStratum | None, attribution: Any) -> Row:
    text = "every check failing the clause has a substitution attribution"
    if perceived is None or attribution is None:
        return missing("R1.attribution", text, "all failing checks", "attribution or stratum")
    met = set(counted(perceived))
    failing = sorted({c.check_id for c in perceived.checks} - met)
    attributed = {
        c["check_id"] for c in attribution[0].get("checks", ()) if c.get("n_clips", 0) > 0
    }
    covered = [check for check in failing if check in attributed]
    status = PASS if len(covered) == len(failing) else FAIL
    absent = sorted(set(failing) - set(covered))
    return Row(
        "R1.attribution",
        text,
        f"{len(covered)}/{len(failing)}",
        "",
        "all failing checks",
        status,
        attribution[1],
        "missing: " + (", ".join(absent) or "none"),
    )


def real_rows(
    report: EvaluationReport | None, c: Criteria, src: str, *, excluded: frozenset[str]
) -> list[Row]:
    """``excluded``: the checks the exclusion record turns off; R2.fpr and R2.deciders list them
    as off by exclusion and judge only the others."""
    perceived = None if report is None else stratum_at(report, "perceived")
    if report is None or perceived is None:
        return [
            missing("R2.fpr", "real video FPR per deciding check", "", "perceived evaluation"),
            missing("R2.decided", "decided fraction per check on real video", "", "evaluation"),
            missing("R2.deciders", "required checks decide on real video", "", "evaluation"),
        ]
    decided = [r for r in perceived.real if r.fpr.n > 0]
    judged = [r for r in decided if r.check_id not in excluded]
    off = sorted({r.check_id for r in perceived.real} & excluded)
    over = [
        f"{r.dataset}/{r.check_id}"
        for r in judged
        if (r.fpr.value or 0.0) > c.real_fpr or (r.fpr.ci.high or 1.0) > c.real_fpr_upper
    ]
    clips = {d.name: d.n_clips for d in report.real_sets}
    shares = [
        f"{r.dataset}/{r.check_id} {r.fpr.n}/{clips.get(r.dataset, 0)}" for r in perceived.real
    ]
    speaking = {r.check_id for r in decided}
    required = [check for check in c.real_deciders if check not in excluded]
    silent = [check for check in required if check not in speaking]
    waived = [check for check in c.real_deciders if check in excluded]
    threshold = f"FPR <= {c.real_fpr}, Clopper-Pearson upper <= {c.real_fpr_upper}"
    return [
        Row(
            "R2.fpr",
            "real video FPR per deciding check",
            f"{len(judged) - len(over)}/{len(judged)}",
            "",
            f"{threshold}, or off by exclusion",
            PASS if decided and not over else FAIL,
            src,
            _with_off("over", over, off),
        ),
        Row(
            "R2.decided",
            "decided fraction per check on real video",
            f"{len(shares)} pairs",
            "",
            "reported",
            PASS if shares else NOT_EVALUABLE,
            src,
            "; ".join(shares),
        ),
        Row(
            "R2.deciders",
            "required checks decide on real video",
            f"{len(required) - len(silent)}/{len(required)}",
            "",
            f"{', '.join(c.real_deciders)}, or off by exclusion",
            FAIL if silent else PASS,
            src,
            _with_off("silent", silent, waived),
        ),
    ]


def _with_off(label: str, found: Sequence[str], off: Sequence[str]) -> str:
    return f"{label}: {', '.join(found) or 'none'}; off by exclusion: {', '.join(off) or 'none'}"


def active_checks(preflight: Mapping[str, Any]) -> set[str]:
    """Every check of weight > 0 in some stratum of the preflight."""
    return {c for found in preflight.get("strata", {}).values() for c in found["active_set"]}


def inputs_row(reports: Mapping[str, EvaluationReport], preflight: Any, c: Criteria) -> Row:
    text = "every active check got its input (no device or extraction gap)"
    threshold = f"MISSING_INPUT <= {c.max_missing_input} of a set's clips"
    if not reports or preflight is None:
        return missing("T.inputs", text, threshold, "evaluation or preflight")
    active = active_checks(preflight[0])
    over = [g for r in reports.values() for g in input_gaps_over(r, active, c.max_missing_input)]
    return Row(
        "T.inputs",
        text,
        f"{len(over)} over",
        "",
        threshold,
        FAIL if over else PASS,
        ", ".join(reports),
        "; ".join(over),
    )


def r0_rows(inputs: Inputs) -> list[Row]:
    required = {
        "freeze": inputs.freeze,
        "kubric": inputs.kubric,
        "curve": inputs.curve,
        "agreement_items": inputs.agreement_items,
        "preflight": inputs.preflight,
    }
    skipped = ("freeze",) if inputs.dry_run and _present(inputs.freeze) is None else ()
    absent = [k for k, p in required.items() if k not in skipped and _present(p) is None]
    note = "; ".join(
        part
        for part in (
            ", ".join(f"{k} skipped (dry run)" for k in skipped),
            "missing: " + ", ".join(absent) if absent else "",
        )
        if part
    )
    return [
        Row(
            "R0.main",
            "everything adopted is in main; branches resolved",
            "n/a",
            "",
            "yes",
            NOT_EVALUABLE,
            "",
            OUTSIDE_RUN,
        ),
        Row(
            "R0.green",
            "suite, ruff and mypy green",
            "n/a",
            "",
            "green",
            NOT_EVALUABLE,
            "",
            OUTSIDE_RUN,
        ),
        Row(
            "R0.one_command",
            "one command reproduces the final test",
            f"{len(absent)} outputs missing",
            "",
            "0 missing",
            FAIL if absent else NOT_EVALUABLE if skipped else PASS,
            str(inputs.freeze or ""),
            note,
        ),
        Row(
            "R0.readme",
            "README with a tutorial",
            "n/a",
            "",
            "present",
            NOT_EVALUABLE,
            "",
            OUTSIDE_RUN,
        ),
    ]


def _json_at(path: Path | None) -> tuple[Any, str] | None:
    return None if path is None or not path.is_file() else (read_json(path), str(path))


def _present(path: Path | None) -> Path | None:
    return path if path is not None and path.is_file() else None


def build_rows(
    inputs: Inputs,
    boot: agreement.Bootstrap,
    criteria: Criteria,
    *,
    excluded: frozenset[str],
) -> list[Row]:
    """Every row of the table; a missing input makes its rows not evaluable. ``excluded``: the
    checks the exclusion record turns off (see ``real_rows``)."""
    kubric_path, curve_path, items_path = (
        _present(p) for p in (inputs.kubric, inputs.curve, inputs.agreement_items)
    )
    kubric = None if kubric_path is None else load_evaluation(kubric_path)
    curve = None if curve_path is None else CurveRecord.model_validate(read_json(curve_path))
    items = None if items_path is None else agreement.read_items(items_path)
    real_path = _present(inputs.real) or kubric_path
    real = None if real_path is None else load_evaluation(real_path)
    srcs = {"kubric": str(inputs.kubric or ""), "curve": str(inputs.curve or "")}
    evaluations = {
        str(path): report
        for path, report in ((kubric_path, kubric), (real_path, real))
        if path is not None and report is not None
    }
    return [
        *r0_rows(inputs),
        *clause_rows(kubric, _json_at(inputs.attribution), criteria, srcs["kubric"]),
        *real_rows(real, criteria, str(real_path or ""), excluded=excluded),
        *agreement.agreement_rows(items, boot, criteria, str(inputs.agreement_items or "")),
        *coverage_rows(curve, kubric, _json_at(inputs.preflight), criteria, srcs, boot.ci),
        inputs_row(evaluations, _json_at(inputs.preflight), criteria),
    ]


def supplementary(inputs: Inputs) -> list[dict[str, Any]]:
    """The R3b extension and the development replications, read the same way, not counted."""
    found = {"r3b_extension": inputs.r3b, **inputs.replications}
    present = sorted((k, v) for k, v in found.items() if v is not None and v.is_file())
    return [
        {"set": name, "level": s.level.value, "met": list(counted(s)), "source": str(path)}
        for name, path in present
        for s in load_evaluation(path).strata
    ]


HEADERS = ("id", "criterion", "value", "CI", "threshold", "status", "source", "note")


def markdown(rows: Sequence[Row], extra: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Final table", ""]
    lines += table(
        HEADERS,
        ([r.id, r.criterion, r.value, r.ci, r.threshold, r.status, r.source, r.note] for r in rows),
    )
    if extra:
        lines += ["", "## Supplementary (not counted)", ""]
        lines += table(
            ("set", "level", "checks meeting the clause", "source"),
            ([e["set"], e["level"], ", ".join(e["met"]) or "none", e["source"]] for e in extra),
        )
    return "\n".join(lines) + "\n"


def as_json(rows: Sequence[Row], extra: Sequence[Mapping[str, Any]]) -> str:
    data = {"rows": [asdict(r) for r in rows], "supplementary": list(extra)}
    return json.dumps(data, indent=2, sort_keys=True) + "\n"
