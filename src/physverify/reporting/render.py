"""``results.json`` and ``results.md``: one table per section, every value with its source."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from physverify.calibration.report import ESTIMATE_NOTE, ci_text, number_text
from physverify.calibration.schema import Ci
from physverify.diagnosis.atomic_io import write_text_atomic
from physverify.reporting.numbers import Results, Section, Value
from physverify.schema.io import json_ready

RESULTS_SCHEMA = "physverify.results"
RESULTS_VERSION = "1.0.0"
RESULTS_JSON = "results.json"
RESULTS_MD = "results.md"
CENSORED = {"below": "<=", "above": ">"}
COLUMNS = ("key", "value", "CI", "k/n", "unit", "estimated", "reportable", "note", "source")


def results_payload(results: Results) -> dict[str, Any]:
    return {
        "schema": RESULTS_SCHEMA,
        "version": RESULTS_VERSION,
        "reportable": results.reportable,
        "banner": results.banner,
        "inputs": results.inputs,
        "sections": {s.key: _section_payload(s) for s in results.sections},
    }


def _section_payload(section: Section) -> dict[str, Any]:
    return {
        "missing": section.missing,
        "caveat": section.caveat,
        "values": {
            v.key: {k: x for k, x in dataclasses.asdict(v).items() if k != "key"}
            for v in section.values
        },
    }


def results_json(results: Results) -> str:
    payload = json_ready(results_payload(results))
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def value_text(v: Value) -> str:
    if v.censored in CENSORED:
        return f"{CENSORED[v.censored]} {number_text(v.bound)}"
    if isinstance(v.value, (bool, int, str)):
        return str(v.value)
    return number_text(v.value)


def _cells(v: Value) -> list[str]:
    has_ci = v.ci_low is not None or v.ci_high is not None
    return [
        v.key,
        value_text(v),
        ci_text(Ci(low=v.ci_low, high=v.ci_high)) if has_ci else "",
        _count(v),
        v.unit or "",
        "estimate" if v.estimated else "",
        "" if v.reportable else "no (rehearsal)",
        v.note or "",
        f"`{v.source}`",
    ]


def _count(v: Value) -> str:
    if v.n is None:
        return ""
    return f"n {v.n}" if v.k is None else f"{v.k}/{v.n}"


def _table(section: Section) -> list[str]:
    lines = [f"## {section.key}", ""]
    if section.caveat:
        lines += [f"Note: {section.caveat}", ""]
    if section.missing:
        return [*lines, f"Missing: {section.missing}", ""]
    lines += ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += [
        "| " + " | ".join(c.replace("|", "\\|") for c in _cells(v)) + " |" for v in section.values
    ]
    return [*lines, ""]


def results_markdown(results: Results) -> str:
    lines = ["# Results", ""]
    if results.banner:
        lines += [f"**{results.banner}**", ""]
    lines += [ESTIMATE_NOTE, "", "## inputs", "", "| input | field | value |", "|---|---|---|"]
    lines += [
        f"| {name} | {field} | `{value}` |"
        for name, fields in results.inputs.items()
        for field, value in fields.items()
    ]
    lines.append("")
    for section in results.sections:
        lines += _table(section)
    return "\n".join(lines)


def write_results(out: Path, results: Results) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = out / RESULTS_JSON, out / RESULTS_MD
    write_text_atomic(paths[0], results_json(results))
    write_text_atomic(paths[1], results_markdown(results))
    return paths
