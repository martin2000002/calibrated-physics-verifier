"""Files of one comparison under ``<verify run>/agreement/`` (ADR 0011 C5)."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from physverify.core.atomic import write_text_atomic
from physverify.core.markdown import number, table
from physverify.human_agreement.breakdown import AGGREGATE
from physverify.human_agreement.coverage import NO_VIDEO
from physverify.human_agreement.error_sample import CATEGORIES
from physverify.human_agreement.items import ComparisonItem
from physverify.human_agreement.pipeline import Comparison
from physverify.schema.checks import AbstainReason
from physverify.schema.io import json_ready

REPORT_DIR = "agreement"
HONESTY = (
    "Verifier magnitudes come from perceived inputs: depth, size and scale are learned "
    "estimates, never measurements, and carry their measured error (sigma, NaN when unknown).",
    "Time comes from the container rate of generated videos, which need not match physical "
    "time; the sensitivity block repeats the primary metrics with the time-dependent checks "
    "moved out of (or back into) the primary.",
    "The reference VLM numbers are the published AutoEval predictions, joined by video URL and "
    "cut at the published thresholds; nothing was tuned on test.",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def item_record(item: ComparisonItem) -> dict[str, object]:
    return {
        "clip_id": item.clip_id,
        "version": item.row.version,
        "generator": item.row.generator,
        "caption": item.row.caption,
        "human_raw": item.row.human_raw,
        "human_implausible": int(item.human_implausible),
        "vlm_raw": item.row.reference_raw,
        "vlm_implausible": "" if item.vlm_implausible is None else int(item.vlm_implausible),
        "verifier_outcome": item.outcome.value,
        "verifier_score": item.decision.score,
        "deciding_check": item.decision.deciding_check or "",
        "abstain_reasons": ";".join(item.decision.reasons),
    }


def write_comparison(out_dir: Path, comparison: Comparison) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [item_record(item) for item in comparison.items]
    _write_csv(out_dir / "items.csv", items, list(items[0]) if items else ["clip_id"])
    for name, rows in (
        ("per_check.csv", comparison.per_check),
        ("per_generator.csv", comparison.per_generator),
        ("errors.csv", comparison.errors),
    ):
        _write_csv(out_dir / name, rows, list(rows[0]) if rows else ["clip_id"])
    text = json.dumps(json_ready(comparison.metrics), indent=2, sort_keys=True)
    write_text_atomic(out_dir / "metrics.json", text + "\n")
    write_text_atomic(out_dir / "summary.md", summary(comparison.metrics))


def _fmt(value: Any) -> str:
    if isinstance(value, float) and math.isfinite(value) and value.is_integer() and abs(value) >= 1:
        return str(int(value))
    if value is None or isinstance(value, float):
        return number(value, ".3f", "n/a", "nonfinite")
    return str(value)


def _interval(block: Mapping[str, Any], key: str) -> str:
    low, high = block.get("bootstrap", {}).get("intervals", {}).get(key, [None, None])
    return f"[{_fmt(low)}, {_fmt(high)}]"


SYSTEM_COLUMNS = (
    ("accuracy", "acc"),
    ("balanced_accuracy", "bal acc"),
    ("f1_implausible", "F1"),
    ("kappa", "kappa"),
    ("pearson", "Pearson"),
    ("spearman", "Spearman"),
    ("auc", "AUC"),
    ("predicted_implausible_rate", "pred implausible"),
)
BOOTSTRAPPED = ("accuracy", "balanced_accuracy", "kappa")
PER_CHECK_COLUMNS = (
    "check_id",
    "n_scored",
    "implausible_rate_alone_human_implausible",
    "implausible_rate_alone_human_plausible",
    "accuracy_alone",
    "balanced_accuracy_alone",
    "accuracy_without",
)
PER_GENERATOR_COLUMNS = (
    "group",
    "n",
    "human_implausible_rate",
    "verifier_accuracy",
    "vlm_accuracy",
    "verifier_balanced_accuracy",
    "vlm_balanced_accuracy",
)


def _cell(block: Mapping[str, Any], system: str, key: str) -> str:
    value = _fmt(block[system].get(key))
    if system in ("verifier", "vlm") and key in BOOTSTRAPPED and block.get("bootstrap"):
        value += f" {_interval(block, f'{system}.{key}')}"
    return value


def _system_rows(block: Mapping[str, Any]) -> list[list[str]]:
    systems = ["verifier", "vlm", *(f"constants.{c}" for c in block["constants"])]
    rows = []
    for name in systems:
        view = {**block, name: block["constants"][name.split(".")[1]]} if "." in name else block
        cells = [_cell(view, name, key) for key, _ in SYSTEM_COLUMNS]
        rows.append([name, _fmt(view[name]["n"]), *cells])
    return rows


def _paired_line(block: Mapping[str, Any]) -> str:
    if not block.get("paired"):
        return "Not paired: the VLM is evaluated on its own clips."
    mcnemar = block["mcnemar"]
    return (
        f"McNemar b (verifier right only) {mcnemar['b_verifier_only']}, c (VLM right only) "
        f"{mcnemar['c_vlm_only']}, p = {_fmt(mcnemar['p_value'])}; discordant rate "
        f"{_fmt(block['discordant_rate'])}; MDE accuracy {_fmt(block['mde_accuracy'])}"
        f"{_clustered_note(block)}; "
        f"differences (verifier - VLM): accuracy {_interval(block, 'difference.accuracy')}, "
        f"balanced accuracy {_interval(block, 'difference.balanced_accuracy')}, kappa "
        f"{_interval(block, 'difference.kappa')}; {_bootstrap_note(block)}."
    )


def _clustered_note(block: Mapping[str, Any]) -> str:
    """The clustered MDE is an estimate: its design effect is estimated from the data."""
    if "mde_accuracy_clustered" not in block:
        return ""
    local = block["design_effect"]
    return (
        f" (estimated with clips clustered by caption: {_fmt(block['mde_accuracy_clustered'])}, "
        f"design effect {_fmt(block['design_effect_used'])} from {block['design_effect_source']}; "
        f"this data: design effect {_fmt(local['deff'])}, ICC {_fmt(local['icc'])})"
    )


def _bootstrap_note(block: Mapping[str, Any]) -> str:
    boot = block.get("bootstrap", {})
    nan = {k: v for k, v in boot.get("non_finite", {}).items() if v}
    return f"caption bootstrap {boot.get('replicas', 0)} replicas, NaN replicas {nan or 'none'}"


def _group_title(group: str) -> str:
    if group == AGGREGATE:
        return f"## Group {group} (aggregate: decides H2)"
    return f"## Group {group} (descriptive only: too few pairs to test H2 on one version)"


def _group_lines(group: str, policies: Mapping[str, Any]) -> list[str]:
    lines = [_group_title(group), ""]
    columns = ["system", "n", *(label for _, label in SYSTEM_COLUMNS)]
    for block in policies.values():
        lines += [f"Policy `{block['policy']}`:", ""]
        lines += [*table(columns, _system_rows(block)), "", _paired_line(block), ""]
    return lines


def _difference(block: Mapping[str, Any], key: str) -> float:
    return float(block["verifier"][key] - block["vlm"][key])


def _sign_flip_lines(groups: Mapping[str, Any], policy: str) -> list[str]:
    """Judge Q4: say so when v1 and v2 disagree on who is ahead."""
    if "v1" not in groups or "v2" not in groups:
        return ["Sign flip between versions: not applicable (one version only)."]
    flips = []
    for key in ("accuracy", "kappa"):
        d1, d2 = (_difference(groups[g][policy], key) for g in ("v1", "v2"))
        if math.isfinite(d1) and math.isfinite(d2) and d1 * d2 < 0:
            flips.append(f"{key} (v1 {_fmt(d1)}, v2 {_fmt(d2)})")
    if not flips:
        return ["Sign flip between versions: none (verifier - VLM has the same sign in v1, v2)."]
    return [f"Sign flip between versions in verifier - VLM (descriptive): {'; '.join(flips)}."]


def _verdict_lines(verdict: Mapping[str, Any], primary: Mapping[str, Any]) -> list[str]:
    ver, vlm = (_fmt(primary[s]["predicted_implausible_rate"]) for s in ("verifier", "vlm"))
    rate = f"verifier predicts implausible on {ver} of clips, the VLM on {vlm}"
    if verdict.get("gate") is not None:
        return _not_evaluable_lines(verdict, rate)
    if not verdict["applicable"]:
        return [f"**H2: {verdict['reason']}.** The numbers below test the pipeline only; {rate}."]
    state = "supported" if verdict["supported"] else "not supported"
    if verdict.get("post_hoc"):
        state += " (POST HOC: the pre-registration was replaced after a test comparison)"
    low, high = verdict["kappa_difference_ci"]
    return [
        f"**H2 {state}** (McNemar p = {_fmt(verdict['mcnemar_p'])}, accuracy difference "
        f"{_fmt(verdict['accuracy_difference'])}, kappa difference CI [{_fmt(low)}, {_fmt(high)}]; "
        f"minimum detectable accuracy difference {_fmt(verdict['mde_accuracy'])}, estimated "
        f"with caption clustering {_fmt(verdict.get('mde_accuracy_clustered'))}); {rate}."
    ]


def _not_evaluable_lines(verdict: Mapping[str, Any], rate: str) -> list[str]:
    """Item 1.3: the stratum was not usable, so the provisional rule ran and H2 has no verdict."""
    return [
        f"**H2 not evaluable with the calibrated rule** ({verdict['gate']}). The numbers below "
        f"are descriptive, from the provisional rule; {rate}.",
        "Abstention and coverage follow; the `decided_only` policy block shows the agreement "
        "on the clips the verifier decided, which is descriptive as well.",
    ]


def _abstention_lines(abstention: Mapping[str, Any], coverage: Mapping[str, Any]) -> list[str]:
    """INTERVAL_CROSSES_THRESHOLD is shown apart: the evidence exists but is too uncertain."""
    reasons = dict(abstention["reasons"])
    crossing = reasons.pop(AbstainReason.INTERVAL_CROSSES_THRESHOLD.value, 0)
    decided = ", ".join(
        f"{g} {_fmt(c['coverage'])} ({_fmt(c['decided'])}/{_fmt(c['n'])})"
        for g, c in coverage.items()
    )
    return [
        f"Verifier outcomes: {abstention['outcomes']}. Coverage (decided fraction): {decided}.",
        f"Undecided clips with an interval crossing the threshold: {crossing}; "
        f"other abstention reasons: {reasons}.",
    ]


def _breakdown_lines(title: str, columns: Sequence[str], rows: Sequence[Any]) -> list[str]:
    body = [[_fmt(row.get(c)) for c in columns] for row in rows]
    return [f"## {title}", "", *(table(columns, body) if body else ["(none)"]), ""]


def _sensitivity_lines(block: Mapping[str, Any]) -> list[str]:
    moved = "without" if block["time_dependent_in_primary"] else "with"
    lines = [
        f"## Sensitivity {moved} the time-dependent checks",
        "",
        f"Time-dependent: {', '.join(block['time_dependent']) or 'none'}; excluded here: "
        f"{', '.join(block['excluded']) or 'none'}; policy {block['policy']}, no bootstrap.",
        "",
    ]
    columns = ["group", "system", "n", *(label for _, label in SYSTEM_COLUMNS)]
    rows = [[g, *row] for g, view in block["groups"].items() for row in _system_rows(view)[:2]]
    return [*lines, *table(columns, rows), ""]


def _audit_lines(audit: Mapping[str, Any] | None) -> list[str]:
    if audit is None:
        return []
    deviations = "; ".join(audit["deviations"]) or "none"
    return [
        f"Pre-registration {audit['prereg_sha256'][:12]} (created {audit['created_utc']}, "
        f"replaces {(audit['replaces_sha256'] or 'none')[:12]}, {audit['versions']} version(s), "
        f"first frozen {audit['first_created_utc']}); verify run files from "
        f"{audit['earliest_eval_utc'] or 'n/a'}, first freeze before them: {audit['blind']}, "
        f"current freeze after them: {audit.get('frozen_after_eval')} "
        "(file times are informational); post hoc: "
        f"{'YES' if audit['post_hoc'] else 'no'}; deviations: {deviations}.",
        "",
    ]


def _gap_lines(gaps: Mapping[str, Any] | None) -> list[str]:
    if gaps is None:
        return []
    missing = [(g, c[NO_VIDEO]) for g, c in sorted(gaps["by_generator"].items()) if c[NO_VIDEO]]
    generators = ", ".join(f"{g} {n}" for g, n in missing) or "none"
    return [
        f"No object query: {gaps['no_query']} rows ({gaps['no_query_with_video']} with video, "
        f"compared as abstentions); videos missing from the source: {gaps['no_video']} "
        f"(by generator: {generators})."
    ]


def _header_lines(metrics: Mapping[str, Any]) -> list[str]:
    counts, run = metrics["counts"], metrics["run"]
    primary = metrics["groups"][AGGREGATE][metrics["primary_policy"]]
    return [
        "# Verifier vs reference VLM on VideoPhy human labels",
        "",
        f"Rule: `{metrics['rule']}`. Primary policy: `{metrics['primary_policy']}`. "
        f"Pre-registered: {'yes' if metrics['preregistered'] else 'no (exploratory)'}. "
        f"Run: `{run['path']}` ({run['profile']}, {run['device']}).",
        "",
        *_audit_lines(metrics.get("preregistration")),
        *_verdict_lines(metrics["h2_verdict"], primary),
        "",
        f"Rows {counts['rows']}, compared {counts['compared']}; left out: no human label "
        f"{counts['human_missing']}, no VLM prediction {counts['vlm_missing']}. Clips missing "
        f"from the run: {counts['not_verified']} (compared as check_error abstentions).",
        *_gap_lines(metrics.get("input_gaps")),
        *_abstention_lines(metrics["abstention"], metrics["coverage"]),
        *_sign_flip_lines(metrics["groups"], metrics["primary_policy"]),
        "",
    ]


def summary(metrics: Mapping[str, Any]) -> str:
    lines = _header_lines(metrics)
    for group, policies in metrics["groups"].items():
        lines += _group_lines(group, policies)
    lines += _breakdown_lines("Per check", PER_CHECK_COLUMNS, metrics["per_check"])
    lines += _breakdown_lines("Per generator", PER_GENERATOR_COLUMNS, metrics["per_generator"])
    lines += _sensitivity_lines(metrics["sensitivity_time_dependent"])
    lines += ["Honesty notes:", "", *(f"- {note}" for note in HONESTY)]
    lines += ["", f"errors.csv categories (filled by hand): {', '.join(CATEGORIES)}.", ""]
    return "\n".join(lines)
