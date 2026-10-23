"""The calibration sections of the results report, each value with its source."""

from __future__ import annotations

from physverify.reporting.sources import (
    CONSUMED_LEVEL,
    Doc,
    Inputs,
    check_index,
    pointer,
    stratum_index,
)
from physverify.reporting.values import Section, Value, detection_limit, leaf, rate, with_ci
from physverify.schema.checks import InputLevel

IN_SAMPLE = "in-sample, optimistic"
DESCRIPTIVE = "descriptive: does not condition h1_pass"


def _artifact_stratum(inputs: Inputs, level: InputLevel) -> str | None:
    index = stratum_index(inputs.artifact.data["strata"], level, inputs.profile)
    return None if index is None else pointer("strata", index)


def evaluation_stratum(doc: Doc, level: InputLevel, profile: str, *root: str) -> str | None:
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
    whole = evaluation_stratum(ev, CONSUMED_LEVEL, inputs.profile)
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
    return evaluation_stratum(doc, CONSUMED_LEVEL, profile, "h1_primary")


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
                f"{check_id}.calib_primary",
                inputs.heldout_evaluation,
                primary[check_id],
                medians=True,
            )
    caveat = None if primary else _no_primary_reason(inputs)
    return Section("monotonicity", tuple(values), caveat=caveat)


def _pooled_primary(inputs: Inputs) -> dict[str, str]:
    stratum = _primary_stratum(inputs.heldout_evaluation, inputs.profile)
    if stratum is None:
        return {}
    entries = inputs.heldout_evaluation.at(stratum).get("monotonicity", [])
    return {
        e["check_id"]: f"{stratum}/monotonicity/{i}/monotonicity" for i, e in enumerate(entries)
    }


def _no_primary_reason(inputs: Inputs) -> str:
    return (
        f"calib_primary missing: {inputs.heldout_evaluation.path} has no pooled tau_b of the "
        "H1 primary view "
        "(an evaluation before 1.4.0; rerun `calibrate evaluate --no-real` and pass "
        "--calibration-evaluation)"
    )


def real_fpr(inputs: Inputs) -> Section:
    """P4: per-check and clip-score false-positive rate on real video (all negatives)."""
    doc = inputs.evaluation
    where = evaluation_stratum(doc, CONSUMED_LEVEL, inputs.profile)
    if where is None:
        return Section("real_fpr", missing="no perceived stratum in the evaluation")
    values = [
        rate(f"{r['dataset']}.{r['check_id']}", doc, f"{where}/{field}/{i}/fpr")
        for field in ("real", "real_clip_flags")
        for i, r in enumerate(doc.at(f"{where}/{field}"))
    ]
    return Section("real_fpr", tuple(values), missing=None if values else "no real video scored")
