"""The numbers of the results report, one section per function, each value with its source.

A value is read from a ``Doc`` by JSON pointer and cites that ``path#pointer``; nothing is
recomputed. Estimates (L_D, sigma factors, p_c, k*, risks) carry ``estimated``; a censored L_D
has ``value`` None and its ``bound``; NaN is None. A CUDA chain is a rehearsal: every value is
``reportable`` False.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from physverify.reporting.calibration_numbers import (
    aggregate,
    evaluation_stratum,
    monotonicity,
    per_check,
    real_fpr,
)
from physverify.reporting.chain import Missing
from physverify.reporting.sources import (
    CONSUMED_LEVEL,
    Doc,
    Inputs,
    is_calibrated,
    pointer,
)
from physverify.reporting.values import Section, Value, interval, leaf, rate, with_ci
from physverify.schema.checks import InputLevel

INTERVAL_ONLY = "interval only: the point difference is not stored"
NO_AUC_CI = "no interval: the AUROC bootstrap is not implemented (ADR 0011 D6)"
NOT_PREREGISTERED = "NOT PREREGISTERED: not the preregistered calibrated rule, no H2 claim"
REHEARSAL_NOTE = "REHEARSAL (device {device}): not reportable"


@dataclass(frozen=True)
class Results:
    reportable: bool
    banner: str | None
    inputs: Mapping[str, Mapping[str, str]]
    sections: tuple[Section, ...]


# --- sections --------------------------------------------------------------------------------


def _risk_at(key: str, doc: Doc, where: str) -> list[Value]:
    return [
        with_ci(f"{key}.risk", doc, f"{where}/risk", f"{where}/risk_ci", estimated=True),
        leaf(f"{key}.coverage", doc, f"{where}/coverage", estimated=True),
        leaf(f"{key}.false_fail_rate", doc, f"{where}/false_fail_rate", estimated=True),
    ]


def _curve_row(doc: Doc, where: str, k: float | None) -> str | None:
    if k is None:
        return None
    rows = doc.at(where)
    index = next((i for i, r in enumerate(rows) if r.get("k_sigma") == k), None)
    return None if index is None else f"{where}/{index}"


def abstention(inputs: Inputs) -> Section:
    """P5: k* on calib (``abstention.json``) and the held-out curve of ``selective-risk curve``."""
    fit = inputs.abstention
    if isinstance(fit, Missing):
        return Section("abstention", missing=fit.reason)
    k_star = fit.at("/k_star")
    values = [
        leaf("k_star_calib", fit, "/k_star", estimated=True),
        leaf("alpha_star", fit, "/alpha_star", estimated=True),
        leaf("risk_target", fit, "/risk_target"),
    ]
    row = _curve_row(fit, "/curve", k_star)
    if row is not None:
        values += _risk_at("calib_at_k_star", fit, row)
    curve = inputs.abstention_report
    if isinstance(curve, Missing):
        return Section(
            "abstention", tuple(values), caveat=f"held-out curve missing: {curve.reason}"
        )
    values += _held_out(curve, k_star)
    return Section("abstention", tuple(values), caveat=f"curve: {curve.path.parent / 'curve.csv'}")


def _held_out(doc: Doc, k_star: float | None) -> list[Value]:
    values = [
        leaf(f"k_star_eval.{name}", doc, f"/tables/{name}/k_star", estimated=True)
        for name in ("raw", "calibrated")
    ]
    diff = "/tables/aurc_calibrated_minus_raw"
    values.append(with_ci("aurc_difference", doc, f"{diff}/value", f"{diff}/ci", estimated=True))
    row = _curve_row(doc, "/tables/calibrated/curve", k_star)
    if row is not None:
        values += _risk_at("eval_at_k_star", doc, row)
    for dataset, block in sorted(doc.at("/real_video").items()):
        for name in sorted(set(block) & {"k_star", "k_2"}):
            values += _risk_at(f"real.{dataset}.{name}", doc, pointer("real_video", dataset, name))
    return values


def _agreement_doc(inputs: Inputs, key: str) -> Doc | Section:
    doc = inputs.metrics
    if doc is None:
        return Section(key, missing="no --agreement-run given")
    return doc


def _h2_caveat(doc: Doc) -> str | None:
    calibrated = is_calibrated(str(doc.at("/rule")))
    return None if calibrated and doc.at("/preregistered") is True else NOT_PREREGISTERED


def agreement(inputs: Inputs) -> Section:
    """P6: the primary policy of the H2 group: verifier and VLM, McNemar, differences."""
    doc = _agreement_doc(inputs, "agreement")
    if isinstance(doc, Section):
        return doc
    block = pointer("groups", doc.at("/h2_group"), doc.at("/primary_policy"))
    intervals = f"{block}/bootstrap/intervals"
    known = doc.at(intervals) if "bootstrap" in doc.at(block) else {}
    values = [leaf("rule", doc, "/rule"), leaf("preregistered", doc, "/preregistered")]
    for system in ("verifier", "vlm"):
        values.append(leaf(f"{system}.n", doc, f"{block}/{system}/n"))
        for metric in ("accuracy", "kappa"):
            name = f"{system}.{metric}"
            low, high = interval(doc, intervals + pointer(name)) if name in known else (None, None)
            values.append(leaf(name, doc, f"{block}/{system}/{metric}", ci_low=low, ci_high=high))
        values.append(leaf(f"{system}.auc", doc, f"{block}/{system}/auc", note=NO_AUC_CI))
    if "mcnemar" in doc.at(block):
        mcnemar = f"{block}/mcnemar"
        values += [
            leaf("mcnemar.p_value", doc, f"{mcnemar}/p_value"),
            leaf("mcnemar.b_verifier_only", doc, f"{mcnemar}/b_verifier_only"),
            leaf("mcnemar.c_vlm_only", doc, f"{mcnemar}/c_vlm_only"),
        ]
    for metric in ("accuracy", "kappa"):
        name = f"difference.{metric}"
        if name in known:
            values.append(_interval_only(name, doc, intervals + pointer(name)))
    return Section("agreement", tuple(values), caveat=_h2_caveat(doc))


def _interval_only(key: str, doc: Doc, where: str) -> Value:
    low, high = interval(doc, where)
    return Value(
        key=key,
        value=None,
        source=doc.source(where),
        ci_low=low,
        ci_high=high,
        note=INTERVAL_ONLY,
    )


def h2_verdict(inputs: Inputs) -> Section:
    """P7: ``h2_verdict`` as ``agreement compare`` wrote it."""
    doc = _agreement_doc(inputs, "h2_verdict")
    if isinstance(doc, Section):
        return doc
    names = ("applicable", "supported", "status", "reason", "gate", "post_hoc", "mcnemar_p")
    values = [leaf(n, doc, f"/h2_verdict/{n}") for n in names]
    values += [
        leaf("accuracy_difference", doc, "/h2_verdict/accuracy_difference"),
        _interval_only("kappa_difference", doc, "/h2_verdict/kappa_difference_ci"),
    ]
    return Section("h2_verdict", tuple(values), caveat=_h2_caveat(doc))


def exclusions(inputs: Inputs) -> Section:
    """What rules 1.1, 1.2 and 1.5 decided: one value per (rule, check)."""
    doc = inputs.record
    if isinstance(doc, Missing):
        return Section("exclusions", missing=doc.reason)
    values = [leaf("sha256", doc, "/sha256")]
    values += [
        leaf(f"{e['rule']}.{e['check_id']}.excluded", doc, f"/evidence/{i}/excluded", note=_why(e))
        for i, e in enumerate(doc.at("/evidence"))
    ]
    return Section("exclusions", tuple(values))


def _why(evidence: Mapping[str, Any]) -> str:
    reason = str(evidence["reason"])
    return reason if evidence["decided"] else f"undecided: {reason}"


def oracle_reference(inputs: Inputs) -> Section:
    """Held-out clean FPR of the oracle stratum: the checks on the simulator's truth."""
    doc = inputs.evaluation
    where = evaluation_stratum(doc, InputLevel.ORACLE, inputs.profile)
    if where is None:
        return Section("oracle_reference", missing="no oracle stratum in the evaluation")
    values = [
        rate(f"{c['check_id']}.fpr_heldout", doc, f"{where}/checks/{i}/fpr_clean")
        for i, c in enumerate(doc.at(f"{where}/checks"))
    ]
    return Section("oracle_reference", tuple(values))


SECTIONS: tuple[Callable[[Inputs], Section], ...] = (
    per_check,
    aggregate,
    monotonicity,
    real_fpr,
    abstention,
    agreement,
    h2_verdict,
    exclusions,
    oracle_reference,
)


def collect_results(inputs: Inputs) -> Results:
    sections = tuple(section(inputs) for section in SECTIONS)
    chain = inputs.chain
    banner = REHEARSAL_NOTE.format(device=chain.device) if chain.rehearsal else None
    if chain.rehearsal:
        sections = tuple(_not_reportable(s) for s in sections)
    return Results(
        reportable=not chain.rehearsal,
        banner=banner,
        inputs=_inputs_summary(inputs),
        sections=sections,
    )


def _not_reportable(section: Section) -> Section:
    values = tuple(dataclasses.replace(v, reportable=False) for v in section.values)
    return dataclasses.replace(section, values=values)


def _inputs_summary(inputs: Inputs) -> dict[str, dict[str, str]]:
    docs: dict[str, Doc | Missing | None] = {
        "artifact": inputs.artifact,
        "evaluation": inputs.evaluation,
        "h1_evaluation": inputs.h1,
        "abstention": inputs.abstention,
        "abstention_report": inputs.abstention_report,
        "metrics": inputs.metrics,
        "exclusion_record": inputs.record,
    }
    found = {
        "chain": {"dir": str(inputs.chain.root), "device": inputs.chain.device},
        "stratum": {"level": CONSUMED_LEVEL.value, "profile": inputs.profile},
        "artifact_run": {"run_id": inputs.calibration.run_id},
    }
    for name, doc in docs.items():
        if isinstance(doc, Doc):
            found[name] = {"path": str(doc.path), "sha256": doc.sha256}
        elif isinstance(doc, Missing):
            found[name] = {"missing": doc.reason}
    return found
