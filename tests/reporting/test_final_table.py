"""The final table's row logic: agreement statistics, coverage rows and missing inputs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from physverify.calibration.artifact import Auroc
from physverify.calibration.gap_record import InputGap
from physverify.reporting import final_agreement as agreement
from physverify.reporting import final_table
from physverify.reporting.final_coverage import best_point_risk, coverage_rows
from physverify.reporting.final_rows import (
    DESCRIPTIVE,
    FAIL,
    NOT_EVALUABLE,
    PASS,
    Criteria,
    Row,
)
from physverify.reporting.final_table import Inputs, as_json, build_rows, markdown
from physverify.schema.checks import InputLevel
from physverify.selective_risk.records import PointRow
from physverify.stats.intervals import Ci

BOOT = agreement.Bootstrap(n=200, seed=7, ci=0.95)
LEVEL = InputLevel.PERCEIVED


def _item(
    caption: str,
    human: bool,
    vlm: bool,
    outcome: str,
    version: str = "v2",
    scoped: bool | None = True,
) -> agreement.Item:
    return agreement.Item(
        caption=caption,
        version=version,
        human=human,
        vlm=vlm,
        decided=outcome in agreement.DECIDED,
        verifier=outcome == "implausible",
        scoped=scoped,
    )


def _point(coverage: float, risk: float) -> PointRow:
    return PointRow(
        k_sigma=1.0,
        alpha=0.1,
        n=100,
        decided=int(coverage * 100),
        coverage=coverage,
        risk=risk,
        risk_ci=(risk / 2, risk * 2),
        false_fail_rate=0.0,
        missed_fail_rate=0.0,
    )


def test_items_csv_keeps_only_vlm_labelled_rows_and_reads_the_outcome(tmp_path: Path) -> None:
    path = tmp_path / "items.csv"
    path.write_text(
        "clip_id,version,caption,human_implausible,vlm_implausible,verifier_outcome\n"
        "a,v1,c1,1,1,implausible\nb,v2,c2,0,,plausible\nc,v2,c3,0,0,abstain\n",
        encoding="utf-8",
    )
    items = agreement.read_items(path)
    assert [(i.version, i.decided, i.verifier) for i in items] == [
        ("v1", True, True),
        ("v2", False, False),
    ]


def test_items_csv_reads_the_scope_from_the_decided_checks(tmp_path: Path) -> None:
    head = "clip_id,version,caption,human_implausible,vlm_implausible,verifier_outcome"
    path = tmp_path / "items.csv"
    path.write_text(f"{head},decided_checks\na,v1,c1,1,1,abstain,R1;R4\nb,v1,c2,0,0,abstain,\n")
    assert [i.scoped for i in agreement.read_items(path)] == [True, False]
    path.write_text(f"{head}\na,v1,c1,1,1,abstain\n")
    assert [i.scoped for i in agreement.read_items(path)] == [None]


def test_the_scoped_gain_is_read_on_the_items_an_active_check_decides() -> None:
    hit = [_item(f"p{k}", True, False, "implausible") for k in range(6)]
    rest = [_item(f"q{k}", False, False, "abstain") for k in range(6)]
    outside = [_item(f"o{k}", True, False, "abstain", scoped=False) for k in range(12)]
    row = agreement.scoped_row([*hit, *rest, *outside], "v2", "VideoPhy-2 test", BOOT, "items.csv")
    assert row.id == "R3.balanced_scoped.v2" and row.value == "0.500" and row.status == PASS
    assert row.note == "n 12 of 24, coverage 0.500"


@pytest.mark.parametrize(
    ("items", "note"),
    [
        ((), "no paired items"),
        ((_item("p", True, False, "abstain", scoped=None),), "no decided_checks column"),
    ],
)
def test_the_scoped_gain_is_not_evaluable_without_its_scope(
    items: tuple[agreement.Item, ...], note: str
) -> None:
    row = agreement.scoped_row(items, "v1", "v1", BOOT, "items.csv")
    assert row.status == NOT_EVALUABLE and row.note == note


def test_the_or_combination_gains_balanced_accuracy_where_the_vlm_misses() -> None:
    items = [_item("p", True, False, "implausible"), _item("q", False, False, "plausible")]
    assert agreement.balanced_accuracy(items, lambda i: i.vlm) == pytest.approx(0.5)
    assert agreement.balanced_accuracy_gain(items)["value"] == pytest.approx(0.5)


def test_decided_accuracy_compares_on_the_decided_items_only() -> None:
    items = [
        _item("p", True, False, "implausible"),
        _item("q", True, True, "abstain"),
        _item("r", False, False, "plausible"),
    ]
    assert agreement.decided_accuracy_difference(items)["value"] == pytest.approx(0.5)


def test_agreement_rows_split_by_version_and_judge_each_against_its_threshold() -> None:
    items = [
        *(
            _item(f"v2-{k}", k % 2 == 0, False, "implausible" if k % 2 == 0 else "plausible")
            for k in range(12)
        ),
        *(_item(f"v1-{k}", True, True, "abstain", "v1") for k in range(6)),
    ]
    rows = {r.id: r for r in agreement.agreement_rows(items, BOOT, Criteria(), "items.csv")}
    assert rows["R3.decided.v2"].status == PASS and rows["R3.decided.v1"].status == FAIL
    assert rows["R3.balanced.v2"].status == PASS
    assert rows["R3.accuracy.v1"].status == NOT_EVALUABLE
    assert rows["R3.decided.all"].note == "n 18"


def test_the_secondary_point_risk_row_takes_the_largest_coverage_within_target() -> None:
    curve = [_point(0.9, 0.2), _point(0.6, 0.08), _point(0.3, 0.02)]
    found = best_point_risk(curve, 0.10)
    assert found is not None and found.coverage == 0.6
    assert best_point_risk([_point(0.9, 0.2)], 0.10) is None


def test_missing_outputs_make_their_rows_not_evaluable_never_pass() -> None:
    inputs = Inputs(freeze=None, kubric=None, curve=None, agreement_items=None, preflight=None)
    rows = build_rows(inputs, BOOT, Criteria(), excluded=frozenset())
    statuses = {r.id: r.status for r in rows}
    assert statuses["R0.one_command"] == FAIL
    assert PASS not in statuses.values()
    ids = [r.id for r in rows]
    assert len(ids) == len(set(ids))
    for needed in (
        "R1.oracle",
        "R1.attribution",
        "R2.fpr",
        "R3.balanced.v1",
        "T.point_risk",
        "T.active_set",
        "T.floor_kubric",
    ):
        assert needed in ids
    assert "| R1.oracle |" in markdown(rows, [])
    assert len(json.loads(as_json(rows, []))["rows"]) == len(rows)


def test_a_leaking_weight_zero_check_fails_the_active_set_row(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"active_set": ["R2"], "unweighted_abstentions": ["R5a"]}))
    inputs = Inputs(freeze=None, kubric=None, curve=None, agreement_items=None, preflight=preflight)
    rows = build_rows(inputs, BOOT, Criteria(), excluded=frozenset())
    row = next(r for r in rows if r.id == "T.active_set")
    assert (row.status, row.value) == (FAIL, "1 leaking")


def _real_report(*checks: tuple[str, float, float]) -> SimpleNamespace:
    """A perceived evaluation of one real set: per check its FPR and Clopper-Pearson upper."""
    real = [
        SimpleNamespace(
            dataset="physics_iq",
            check_id=check_id,
            fpr=SimpleNamespace(n=40, value=value, ci=SimpleNamespace(high=high)),
        )
        for check_id, value, high in checks
    ]
    stratum = SimpleNamespace(level=SimpleNamespace(value="perceived"), real=real)
    return SimpleNamespace(
        strata=[stratum], real_sets=[SimpleNamespace(name="physics_iq", n_clips=50)]
    )


@pytest.mark.parametrize(
    ("excluded", "value", "status", "note"),
    [
        ({"R1"}, "1/1", PASS, "over: none; off by exclusion: R1"),
        ({"R5a"}, "1/2", FAIL, "over: physics_iq/R1; off by exclusion: none"),
        (set(), "1/2", FAIL, "over: physics_iq/R1; off by exclusion: none"),
    ],
    ids=["excluded_over", "other_excluded", "no_exclusion"],
)
def test_a_check_off_by_exclusion_is_listed_and_not_judged(
    excluded: set[str], value: str, status: str, note: str
) -> None:
    report = _real_report(("R1", 0.30, 0.45), ("R2", 0.02, 0.10))
    rows = final_table.real_rows(report, Criteria(), "e.json", excluded=frozenset(excluded))  # type: ignore[arg-type]
    fpr = next(r for r in rows if r.id == "R2.fpr")
    assert (fpr.value, fpr.status, fpr.note) == (value, status, note)
    assert fpr.threshold.endswith("or off by exclusion")


@pytest.mark.parametrize(
    ("excluded", "value", "status", "note"),
    [
        ({"R1"}, "4/4", PASS, "silent: none; off by exclusion: R1"),
        (set(), "4/5", FAIL, "silent: R1; off by exclusion: none"),
    ],
    ids=["excluded_silent", "required_silent"],
)
def test_a_silent_required_check_fails_unless_off_by_exclusion(
    excluded: set[str], value: str, status: str, note: str
) -> None:
    speaking = [(check, 0.02, 0.10) for check in ("R2", "R4_jump", "R4_missing", "R4_size")]
    report = _real_report(*speaking)
    rows = final_table.real_rows(report, Criteria(), "e.json", excluded=frozenset(excluded))  # type: ignore[arg-type]
    deciders = next(r for r in rows if r.id == "R2.deciders")
    assert (deciders.value, deciders.status, deciders.note) == (value, status, note)


def _only_freeze_missing(tmp_path: Path, dry_run: bool) -> Inputs:
    paths = {}
    for name in ("kubric", "curve", "agreement_items", "preflight"):
        paths[name] = tmp_path / name
        paths[name].write_text("{}")
    return Inputs(freeze=None, dry_run=dry_run, **paths)


def test_a_rehearsal_skips_the_freeze_and_a_real_run_requires_it(tmp_path: Path) -> None:
    def r0(dry_run: bool) -> Row:
        inputs = _only_freeze_missing(tmp_path, dry_run)
        return next(r for r in final_table.r0_rows(inputs) if r.id == "R0.one_command")

    dry, real = r0(True), r0(False)
    assert (dry.status, dry.note) == (NOT_EVALUABLE, "freeze skipped (dry run)")
    assert (real.status, real.note) == (FAIL, "missing: freeze")


@pytest.mark.parametrize(("value", "status"), [(0.75, PASS), (0.65, FAIL), (None, NOT_EVALUABLE)])
def test_the_perceived_auroc_row_reads_the_heldout_aggregate(
    value: float | None, status: str
) -> None:
    found = Auroc(value=value, ci=Ci(low=0.6, high=0.8), n_positive=30, n_negative=10)
    s_only = SimpleNamespace(auroc=found, auroc_graded=None)
    graded = SimpleNamespace(auroc=None, auroc_graded=found)
    row = final_table.auroc_row(graded, Criteria(), "e.json")  # type: ignore[arg-type]
    assert row.status == status
    assert final_table.auroc_row(s_only, Criteria(), "").status == NOT_EVALUABLE  # type: ignore[arg-type]
    assert final_table.auroc_row(None, Criteria(), "").status == NOT_EVALUABLE


@pytest.mark.parametrize("value", [0.95, 0.40])
def test_the_s_auroc_row_is_printed_beside_and_judged_by_none(value: float) -> None:
    """The hypothesis judges B1; the AUROC of 1 - S has no threshold and no pass or fail."""
    found = Auroc(value=value, ci=Ci(low=0.3, high=0.99), n_positive=30, n_negative=10)
    perceived = SimpleNamespace(auroc=found, auroc_graded=None)
    row = final_table.auroc_s_row(perceived, "e.json")  # type: ignore[arg-type]
    assert (row.id, row.status, row.threshold) == ("R1.perceived_auroc_s", DESCRIPTIVE, "")
    assert row.value == f"{value:.3f}"
    rows = final_table.clause_rows(None, None, Criteria(), "")
    ids = [r.id for r in rows]
    assert ids.index("R1.perceived_auroc_s") == ids.index("R1.perceived_auroc") + 1


def test_the_rule_row_labels_the_coverage_interval_and_the_risk_interval() -> None:
    points = [_point(0.9, 0.2), _point(0.6, 0.04)]
    calibrated = SimpleNamespace(curve=points, best=1)
    curve = SimpleNamespace(tables=SimpleNamespace(calibrated=calibrated))
    srcs = {"curve": "a.json", "kubric": ""}
    rows = coverage_rows(curve, None, None, Criteria(), srcs, 0.95)  # type: ignore[arg-type]
    rule = next(r for r in rows if r.id == "T.rule")
    assert rule.status == PASS
    assert rule.value.startswith("coverage ") and rule.ci.startswith("coverage [")
    assert rule.note.startswith("risk 0.04")


def test_an_active_check_without_input_fails_the_inputs_row() -> None:
    gaps = (
        InputGap(
            dataset="physics_iq", level=InputLevel.PERCEIVED, check_id="R4_size", missing=54, n=54
        ),
        InputGap(
            dataset="physics_iq", level=InputLevel.PERCEIVED, check_id="R5a", missing=54, n=54
        ),
        InputGap(
            dataset="kubric_fresh", level=InputLevel.PERCEIVED, check_id="R2", missing=1, n=40
        ),
    )
    report = SimpleNamespace(missing_input=gaps)
    preflight = ({"strata": {"perceived/precise": {"active_set": ["R2", "R4_size"]}}}, "p.json")
    row = final_table.inputs_row({"e.json": report}, preflight, Criteria())  # type: ignore[dict-item]
    assert (row.status, row.note) == (FAIL, "physics_iq perceived R4_size 54/54")
    clean = SimpleNamespace(missing_input=gaps[2:])
    assert final_table.inputs_row({"e.json": clean}, preflight, Criteria()).status == PASS  # type: ignore[dict-item]
    assert final_table.inputs_row({}, preflight, Criteria()).status == NOT_EVALUABLE


def test_a_check_counts_as_attributed_only_with_its_own_clips() -> None:
    perceived = SimpleNamespace(
        checks=[SimpleNamespace(check_id=c) for c in ("R2", "R3a", "R5a")], heldout_h1=()
    )
    attribution = (
        {
            "checks": [{"check_id": "R2", "n_clips": 4}, {"check_id": "R3a", "n_clips": 0}],
            "failures": [{"check_id": "R5a"}],
        },
        "a.json",
    )
    row = final_table._attribution_row(perceived, attribution)  # type: ignore[arg-type]
    assert (row.value, row.note) == ("1/3", "missing: R3a, R5a")
