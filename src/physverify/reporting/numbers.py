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

from physverify.reporting.chain import Missing
from physverify.reporting.sources import (
    CONSUMED_LEVEL,
    Doc,
    Inputs,
    check_index,
    is_calibrated,
    pointer,
    stratum_index,
)
from physverify.schema.checks import InputLevel
from physverify.stats.intervals import Rate

Scalar = float | int | str | bool | None
IN_SAMPLE = "in-sample, optimistic"
DESCRIPTIVE = "descriptive: does not condition h1_pass"
INTERVAL_ONLY = "interval only: the point difference is not stored"
NO_AUC_CI = "no interval: the AUROC bootstrap is not implemented (ADR 0011 D6)"
NOT_PREREGISTERED = "NOT PREREGISTERED: not the preregistered calibrated rule, no H2 claim"
REHEARSAL_NOTE = "REHEARSAL (device {device}): not reportable"
CI_LOW, CI_HIGH = "low", "high"
EXACT_CI = Rate.model_fields["ci_method"].default


@dataclass(frozen=True)
class Value:
    """``source`` names ``value`` (for an interval-only value, the ``[low, high]`` pair)."""

    key: str
    value: Scalar
    source: str
    ci_low: float | None = None
    ci_high: float | None = None
    k: int | None = None
    n: int | None = None
    unit: str | None = None
    estimated: bool = False
    censored: str | None = None
    bound: float | None = None
    note: str | None = None
    reportable: bool = True


@dataclass(frozen=True)
class Section:
    key: str
    values: tuple[Value, ...] = ()
    missing: str | None = None
    caveat: str | None = None


@dataclass(frozen=True)
class Results:
    reportable: bool
    banner: str | None
    inputs: Mapping[str, Mapping[str, str]]
    sections: tuple[Section, ...]


# --- value readers ---------------------------------------------------------------------------


def _finite(value: Any) -> Any:
    return None if isinstance(value, float) and value != value else value


def leaf(key: str, doc: Doc, where: str, **extra: Any) -> Value:
    return Value(key=key, value=_finite(doc.at(where)), source=doc.source(where), **extra)


def interval(doc: Doc, where: str) -> tuple[float | None, float | None]:
    """A ``{low, high}`` object or a ``[low, high]`` pair."""
    found = doc.at(where)
    low, high = (found[CI_LOW], found[CI_HIGH]) if isinstance(found, Mapping) else found
    return _finite(low), _finite(high)


def with_ci(key: str, doc: Doc, where: str, ci: str, **extra: Any) -> Value:
    low, high = interval(doc, ci)
    return leaf(key, doc, where, ci_low=low, ci_high=high, **extra)


def rate(key: str, doc: Doc, where: str, **extra: Any) -> Value:
    found = doc.at(where)
    method = found.get("ci_method", EXACT_CI)
    note = None if method == EXACT_CI else f"{found.get('n_scenes')} scenes, {method}"
    return with_ci(
        key, doc, f"{where}/value", f"{where}/ci", k=found["k"], n=found["n"], note=note, **extra
    )


def detection_limit(key: str, doc: Doc, where: str) -> Value:
    found = doc.at(where)
    return with_ci(
        key,
        doc,
        f"{where}/value",
        f"{where}/ci",
        n=found["n"],
        unit=found["unit"],
        estimated=True,
        censored=found["censored"],
        bound=found["bound"],
        note=found.get("reason"),
    )


# --- sections --------------------------------------------------------------------------------


def _artifact_stratum(inputs: Inputs, level: InputLevel) -> str | None:
    index = stratum_index(inputs.artifact.data["strata"], level, inputs.profile)
    return None if index is None else pointer("strata", index)


def _evaluation_stratum(doc: Doc, level: InputLevel, profile: str, *root: str) -> str | None:
    base = pointer(*root)
    strata = doc.at(f"{base}/strata") if root else doc.data["strata"]
    index = stratum_index(strata, level, profile)
    return None if index is None else f"{base}/strata/{index}"


def _heldout_rate(key: str, doc: Doc, stratum: str | None, check_id: str) -> tuple[Value, ...]:
    if stratum is None:
        return ()
    index = check_index(doc.at(f"{stratum}/checks"), check_id)
    if index is None:
        return ()
    return (rate(key, doc, f"{stratum}/checks/{index}/fpr_clean"),)


def per_check(inputs: Inputs) -> Section:
    """P1: tau, cross-validated and held-out FPR (all scenes and H1 primary), L_D, sigma."""
    base = _artifact_stratum(inputs, CONSUMED_LEVEL)
    if base is None:
        return Section("per_check", missing=f"no {CONSUMED_LEVEL.value}/{inputs.profile} stratum")
    doc, ev = inputs.artifact, inputs.evaluation
    whole = _evaluation_stratum(ev, CONSUMED_LEVEL, inputs.profile)
    primary = _primary_stratum(ev, inputs.profile)
    values: list[Value] = []
    for check_id in sorted(doc.at(f"{base}/checks")):
        c = f"{base}/checks/{check_id}"
        status = doc.at(f"{c}/status")
        values += [
            leaf(f"{check_id}.tau", doc, f"{c}/tau", unit=doc.at(f"{c}/unit"), note=status),
            rate(f"{check_id}.fpr_cv", doc, f"{c}/fpr/cv"),
            *_heldout_rate(f"{check_id}.fpr_heldout", ev, whole, check_id),
            *_heldout_rate(f"{check_id}.fpr_heldout_primary", ev, primary, check_id),
            detection_limit(f"{check_id}.detection_limit", doc, f"{c}/detection_limit"),
            with_ci(
                f"{check_id}.sigma_factor",
                doc,
                f"{c}/sigma/factor",
                f"{c}/sigma/factor_ci",
                estimated=True,
                note=doc.at(f"{c}/sigma").get("reason"),
            ),
        ]
    return Section("per_check", tuple(values))


def _primary_stratum(doc: Doc, profile: str) -> str | None:
    if doc.data.get("h1_primary") is None:
        return None
    return _evaluation_stratum(doc, CONSUMED_LEVEL, profile, "h1_primary")


def aggregate(inputs: Inputs) -> Section:
    """P2: weights (with the informedness interval they come from), p_c, s* and its AUROC."""
    base = _artifact_stratum(inputs, CONSUMED_LEVEL)
    if base is None:
        return Section("aggregate", missing=f"no {CONSUMED_LEVEL.value}/{inputs.profile} stratum")
    doc = inputs.artifact
    values: list[Value] = []
    for check_id in sorted(doc.at(f"{base}/checks")):
        c = f"{base}/checks/{check_id}"
        values += [
            leaf(f"weight.{check_id}", doc, f"{c}/weight/value"),
            with_ci(
                f"informedness.{check_id}",
                doc,
                f"{c}/weight/informedness",
                f"{c}/weight/informedness_ci",
            ),
            leaf(f"probability.{check_id}.a", doc, f"{c}/probability/a", estimated=True),
            leaf(f"probability.{check_id}.b", doc, f"{c}/probability/b", estimated=True),
        ]
    agg = f"{base}/aggregate"
    in_sample = IN_SAMPLE if doc.at(f"{agg}/score_threshold_in_sample") else None
    values += [
        leaf("score_threshold", doc, f"{agg}/score_threshold", note=in_sample),
        leaf("score_threshold_sufficient", doc, f"{agg}/sufficient"),
        with_ci(
            "score_auroc",
            doc,
            f"{agg}/auroc/value",
            f"{agg}/auroc/ci",
            note=IN_SAMPLE if doc.at(f"{agg}/auroc/in_sample") else None,
        ),
    ]
    return Section("aggregate", tuple(values))


def _tau_values(key: str, doc: Doc, where: str, *, medians: bool) -> list[Value]:
    found = doc.at(where)
    values = [
        with_ci(
            f"{key}.tau_b",
            doc,
            f"{where}/kendall_tau_b",
            f"{where}/ci",
            n=found["n_pairs"],
            note=f"{found['n_scenes']} scenes",
        ),
        leaf(f"{key}.h1_pass", doc, f"{where}/h1_pass", note="lower CI bound of tau_b > 0"),
    ]
    if medians:
        values.append(
            leaf(f"{key}.monotone_medians", doc, f"{where}/monotone_medians", note=DESCRIPTIVE)
        )
        values += [
            leaf(
                f"{key}.median.L{level}",
                doc,
                f"{where}/median_delta_by_level{pointer(level)}",
                note=DESCRIPTIVE,
            )
            for level in found["median_delta_by_level"]
        ]
    return values


def monotonicity(inputs: Inputs) -> Section:
    """P3: D8's pooled tau_b on every calib scene (the artifact, read by rule 1.1) and on the
    H1 primary calib scenes (ADR 0015 1.4); ``h1_pass`` is the lower bound alone."""
    base = _artifact_stratum(inputs, CONSUMED_LEVEL)
    if base is None:
        return Section(
            "monotonicity", missing=f"no {CONSUMED_LEVEL.value}/{inputs.profile} stratum"
        )
    doc = inputs.artifact
    primary = _pooled_primary(inputs)
    values: list[Value] = []
    for check_id in sorted(doc.at(f"{base}/checks")):
        where = f"{base}/checks/{check_id}/monotonicity"
        values += _tau_values(f"{check_id}.calib_all", doc, where, medians=True)
        if check_id in primary:
            values += _tau_values(
                f"{check_id}.calib_primary", inputs.h1, primary[check_id], medians=True
            )
    caveat = None if primary else _no_primary_reason(inputs)
    return Section("monotonicity", tuple(values), caveat=caveat)


def _pooled_primary(inputs: Inputs) -> dict[str, str]:
    stratum = _primary_stratum(inputs.h1, inputs.profile)
    if stratum is None:
        return {}
    entries = inputs.h1.at(stratum).get("monotonicity", [])
    return {
        e["check_id"]: f"{stratum}/monotonicity/{i}/monotonicity" for i, e in enumerate(entries)
    }


def _no_primary_reason(inputs: Inputs) -> str:
    return (
        f"calib_primary missing: {inputs.h1.path} has no pooled tau_b of the H1 primary view "
        "(an evaluation before 1.4.0; rerun `calibrate evaluate --no-real` and pass "
        "--h1-evaluation)"
    )


def real_fpr(inputs: Inputs) -> Section:
    """P4: per-check and clip-score false-positive rate on real video (all negatives)."""
    doc = inputs.evaluation
    where = _evaluation_stratum(doc, CONSUMED_LEVEL, inputs.profile)
    if where is None:
        return Section("real_fpr", missing="no perceived stratum in the evaluation")
    values = [
        rate(f"{r['dataset']}.{r['check_id']}", doc, f"{where}/{field}/{i}/fpr")
        for field in ("real", "real_clip_flags")
        for i, r in enumerate(doc.at(f"{where}/{field}"))
    ]
    return Section("real_fpr", tuple(values), missing=None if values else "no real video scored")


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
    """P5: k* on calib (``abstention.json``) and the held-out curve of ``abstain curve``."""
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
        return Section(key, missing="no --h2-run given")
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
    where = _evaluation_stratum(doc, InputLevel.ORACLE, inputs.profile)
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
