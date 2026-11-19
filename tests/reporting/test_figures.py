"""The data behind each thesis figure on hand-built records, where a run's outputs are found, and
the labels that name what a figure shows."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from child_python import run_python
from cli_runner import CliRunner
from comparison_cases import KUBRIC, REAL, clip, config, part
from figure_cases import (
    LEVEL_CI,
    TABLE,
    attributed,
    attribution,
    auroc,
    built_curve,
    counted,
    curve,
    dataset,
    evaluation,
    events,
    heldout,
    point,
    share,
    stratum,
    swept,
)
from reporting_cases import write

from physverify.calibration.artifact import RealFpr
from physverify.core.errors import ConfigError, UsageError
from physverify.reporting.chain import STATE_FILE
from physverify.reporting.figures._base import (
    MalformedInputError,
    NoDataError,
    Output,
    Value,
    figures,
)
from physverify.reporting.figures._labels import COMMON, FORMAT, MINUS, NAMES, load_labels
from physverify.reporting.figures._publish import select
from physverify.reporting.figures._records import RUN_DIR, Located, Records, locate
from physverify.reporting.figures.aggregate_auroc import B1, ONE_MINUS_S, auroc_rows
from physverify.reporting.figures.check_detection import detection_panels
from physverify.reporting.figures.pair_accuracy import pair_accuracy
from physverify.reporting.figures.real_false_alarms import real_alarms
from physverify.reporting.figures.risk_coverage import (
    CALIBRATED,
    CALIBRATION,
    CONFIGURED,
    NO_K_STAR,
    NOT_EXACT,
    OFF_GRID,
    RAW,
    SUMMED,
    UNDECIDED,
    Gap,
    Marker,
    Operating,
    calibration_marker,
    configured_marker,
    risk_coverage,
)
from physverify.reporting.figures.stage_attribution import stage_shares
from physverify.reporting.final_rows import Criteria
from physverify.schema.checks import InputLevel
from physverify.schema.stages import Stage
from physverify.selective_risk.curve import (
    curve_point,
    decision_codes,
    reference_codes,
    risk_bounds,
)
from physverify.selective_risk.records import CurveRecord, PointRow
from physverify.selective_risk.settings import REPORT_FILE

ORACLE, PERCEIVED, MIXED = InputLevel.ORACLE, InputLevel.PERCEIVED, InputLevel.MIXED
LABELS = {
    COMMON: {"oracle": "Oracle inputs"},
    "fig": {"x_label": "Pair accuracy over {n} pairs", "limit": "Limit ({value})"},
}


def bounded(value: Value) -> bool:
    low, high = value.low, value.high
    return value.value is not None and low is not None and high is not None and low <= high


def test_detection_puts_clean_clips_first_and_keeps_a_check_seen_in_one_stratum() -> None:
    report = evaluation(
        stratum(
            ORACLE,
            [
                heldout("R1", share(1, 10), [(0.1, share(4, 8)), (0.2, share(8, 8))]),
                heldout("R2", share(0, 10), [(1.0, share(2, 4))]),
            ],
        ),
        stratum(PERCEIVED, [heldout("R1", share(2, 10), [(0.1, share(3, 8))])]),
    )
    r1, r2 = detection_panels(report)
    assert (r1.check_id, r2.check_id) == ("R1", "R2")
    assert r1.magnitudes == (0.1, 0.2)
    assert [c.stratum for c in r1.curves] == ["oracle", "perceived"]
    assert [v.value for v in r1.curves[0].rates] == [0.1, 0.5, 1.0]
    assert r1.curves[0].rates[0] == Value.of_rate(share(1, 10))
    assert [v.value for v in r1.curves[1].rates] == [0.2, 3 / 8, None]
    assert [c.stratum for c in r2.curves] == ["oracle"]


def test_detection_aligns_the_input_levels_by_level_id() -> None:
    report = evaluation(
        stratum(ORACLE, [heldout("R1", share(1, 10), [(0.1, share(4, 8)), (0.2, share(8, 8))])]),
        stratum(PERCEIVED, [heldout("R1", share(2, 10), [(0.25, share(3, 8))], first_level=2)]),
        stratum(MIXED, [heldout("R1", share(0, 10), [(None, share(1, 4))], first_level=3)]),
    )
    (panel,) = detection_panels(report)
    assert panel.magnitudes == (0.1, 0.2, None)
    oracle, perceived, mixed = ([v.value for v in c.rates] for c in panel.curves)
    assert oracle == [0.1, 0.5, 1.0, None]
    assert perceived == [0.2, None, 3 / 8, None]
    assert mixed == [0.0, None, None, 0.25]


def test_detection_without_held_out_checks_has_nothing_to_draw() -> None:
    with pytest.raises(NoDataError, match="no held-out check"):
        detection_panels(evaluation(stratum(ORACLE), stratum(PERCEIVED)))


def test_auroc_rows_put_b1_before_one_minus_s_and_skip_missing_areas() -> None:
    report = evaluation(
        stratum(ORACLE, graded=auroc(0.9, 0.8, 1.0), plain=auroc(0.7, 0.6, 0.8, in_sample=True)),
        stratum(PERCEIVED, graded=auroc(None, 0.0, 1.0), plain=auroc(0.6, 0.5, 0.7)),
        stratum(MIXED),
    )
    found = auroc_rows(report, 0.7)
    assert [(r.score, r.stratum) for r in found.rows] == [
        (B1, "oracle"),
        (ONE_MINUS_S, "oracle"),
        (ONE_MINUS_S, "perceived"),
    ]
    assert found.rows[0].auroc == Value(0.9, 0.8, 1.0)
    assert [r.in_sample for r in found.rows] == [False, True, False]
    assert found.target == 0.7


def test_auroc_rows_without_any_area_have_nothing_to_draw() -> None:
    with pytest.raises(NoDataError, match="no AUROC"):
        auroc_rows(evaluation(stratum(ORACLE, plain=auroc(None, 0.0, 1.0))), 0.7)


CALIBRATED_CURVE = swept(
    point(2.0, 0.4, 0.05, 0.01, 0.09),
    point(1.0, 0.9, 0.2, 0.1, 0.3),
    point(0.5, math.nan, math.nan),
    point(1.5, 0.6, 0.08),
    best=3,
)
RAW_CURVE = swept(point(1.0, 1.0, 0.3, 0.2, 0.4))


def test_risk_coverage_sorts_the_finite_points_of_both_tables() -> None:
    found = risk_coverage(curve(CALIBRATED_CURVE, RAW_CURVE), 0.1, LEVEL_CI)
    ours, raw = found.curves
    assert (ours.table, raw.table) == (CALIBRATED, RAW)
    assert ours.coverage == (0.4, 0.6, 0.9)
    assert ours.risk == (Value(0.05, 0.01, 0.09), Value(0.08), Value(0.2, 0.1, 0.3))
    assert [m.label for m in found.markers] == [CALIBRATION, CONFIGURED]
    assert (found.target, found.estimated) == (0.1, True)


def test_risk_coverage_marks_k_star_of_calibration_not_the_point_the_eval_curve_chose() -> None:
    record = curve(CALIBRATED_CURVE, RAW_CURVE, k_star=2.0)
    marker = risk_coverage(record, 0.1, LEVEL_CI).markers[0]
    assert marker == Marker(CALIBRATION, Operating(0.4, Value(0.05, 0.01, 0.09), 2.0))


def test_k_star_of_zero_is_a_k_and_is_marked() -> None:
    calibrated = swept(*CALIBRATED_CURVE.curve, point(0.0, 1.0, 0.3, 0.2, 0.4))
    found = calibration_marker(curve(calibrated, RAW_CURVE, k_star=0.0), LEVEL_CI).found
    assert found == Operating(1.0, Value(0.3, 0.2, 0.4), 0.0)


@pytest.mark.parametrize(
    ("k_star", "marker"),
    [
        (None, Marker(CALIBRATION, Gap(NO_K_STAR))),
        (2.0 * (1 + 1e-12), Marker(SUMMED, Gap(OFF_GRID, 2.0 * (1 + 1e-12)))),
        (3.0, Marker(CALIBRATION, Gap(UNDECIDED, 3.0))),
    ],
    ids=["no k met the target", "off the curve, no per-check rows", "nothing decided"],
)
def test_calibration_marker_names_why_k_star_is_not_drawn(
    k_star: float | None, marker: Marker
) -> None:
    undecided = point(3.0, 0.0, math.nan).model_copy(update={"decided": 0})
    calibrated = swept(*CALIBRATED_CURVE.curve, undecided)
    assert calibration_marker(curve(calibrated, RAW_CURVE, k_star=k_star), LEVEL_CI) == marker


def test_k_star_off_the_curve_is_summed_from_the_per_check_rows_at_k_star() -> None:
    per_check = {
        "R1": {"k_star": counted(1.7, 6, 5, 2), "k_2": counted(2.0, 6, 4, 1)},
        "R2": {"k_star": counted(1.7, 4, 3, 1), "k_2": counted(2.0, 4, 2, 0)},
    }
    record = curve(CALIBRATED_CURVE, RAW_CURVE, k_star=1.7, per_check=per_check)
    risk = Value(3 / 8, *risk_bounds(3, 8, LEVEL_CI))
    assert calibration_marker(record, LEVEL_CI) == Marker(SUMMED, Operating(0.8, risk, 1.7))


def test_configured_marker_sums_the_per_check_rows_into_the_aggregate_at_the_record_k() -> None:
    per_check = {
        "R1": {"k_star": counted(1.0, 6, 5, 2), "k_2": counted(2.5, 6, 4, 1)},
        "R2": {"k_2": counted(2.5, 4, 2, 0)},
    }
    record = curve(CALIBRATED_CURVE, RAW_CURVE, per_check=per_check)
    risk = Value(1 / 6, *risk_bounds(1, 6, LEVEL_CI))
    assert configured_marker(record, LEVEL_CI) == Marker(CONFIGURED, Operating(0.6, risk, 2.5))


@pytest.mark.parametrize(
    "per_check",
    [
        {},
        {"R1": {"k_2": counted(2.0, 10, 4, 1)}, "R2": {"k_star": counted(1.0, 4, 2, 0)}},
        {"R1": {"k_2": counted(2.0, 6, 4, 1)}, "R2": {"k_2": counted(2.0, 3, 2, 0)}},
        {"R1": {"k_2": counted(2.0, 6, 4, 1)}, "R2": {"k_2": counted(2.5, 4, 2, 0)}},
        {"R1": {"k_2": counted(2.0, 10, 4, 1).model_copy(update={"risk": 0.3})}},
    ],
    ids=["no rows", "a check without k_2", "fewer cases", "two k", "no count gives the risk"],
)
def test_configured_marker_without_an_exact_aggregate_is_named(
    per_check: dict[str, dict[str, PointRow]],
) -> None:
    record = curve(CALIBRATED_CURVE, RAW_CURVE, per_check=per_check)
    assert configured_marker(record, LEVEL_CI) == Marker(CONFIGURED, Gap(NOT_EXACT))


def test_configured_marker_with_nothing_decided_is_named() -> None:
    per_check = {"R1": {"k_2": counted(2.0, 6, 0, 0)}, "R2": {"k_2": counted(2.0, 4, 0, 0)}}
    record = curve(CALIBRATED_CURVE, RAW_CURVE, per_check=per_check)
    assert configured_marker(record, LEVEL_CI).found == Gap(UNDECIDED, 2.0)


def test_risk_coverage_on_the_study_records_marks_calibration_k_star_and_the_exact_k_2() -> None:
    held = events(240, 2)
    record = CurveRecord.model_validate_json(built_curve(held, 2.0).model_dump_json())
    calibrated = record.tables.calibrated
    assert calibrated.k_star != record.k_star_from_calib
    (row,) = [p for p in calibrated.curve if p.k_sigma == record.k_star_from_calib]
    calibration, configured = risk_coverage(record, 0.1, LEVEL_CI).markers
    at_k_star = Operating(row.coverage, Value(row.risk, *row.risk_ci), row.k_sigma)
    assert calibration == Marker(CALIBRATION, at_k_star)
    codes = decision_codes(held.cases, np.array([2.0]), TABLE)
    direct = curve_point(codes[0], reference_codes(held.cases))
    risk = Value(direct.risk, *risk_bounds(direct.errors, direct.decided, LEVEL_CI))
    assert configured == Marker(CONFIGURED, Operating(direct.coverage, risk, 2.0))


def test_risk_coverage_without_a_finite_point_has_nothing_to_draw() -> None:
    empty = swept(point(1.0, math.nan, math.nan))
    with pytest.raises(NoDataError, match="no finite point"):
        risk_coverage(curve(empty, empty), 0.1, LEVEL_CI)


def test_stage_shares_follow_the_pipeline_and_skip_checks_without_attributed_clips() -> None:
    report = attribution(
        attributed(
            "R5a",
            3,
            {Stage.SCALE: 0.3, Stage.TRACK: 0.1, Stage.DEPTH: math.nan},
            {Stage.TRACK: (0.0, 0.2)},
        ),
        attributed("R1", 0, {Stage.TRACK: 0.5}),
        attributed("R2", 4, {}),
    )
    (only,) = stage_shares(report)
    assert (only.check_id, only.unit, only.n_clips) == ("R5a", "m", 3)
    assert [s.stage for s in only.stages] == ["track", "depth", "scale"]
    assert [s.share for s in only.stages] == [Value(0.1, 0.0, 0.2), Value(None), Value(0.3)]


def test_stage_shares_without_attributed_clips_have_nothing_to_draw() -> None:
    with pytest.raises(NoDataError, match="no check"):
        stage_shares(attribution(attributed("R1", 0, {Stage.TRACK: 0.5})))


def test_real_alarms_group_perceived_rows_by_set_with_the_share_decided() -> None:
    rows = [
        RealFpr(dataset="physics_iq", check_id="R1", fpr=share(1, 8)),
        RealFpr(dataset="physics_iq", check_id="R2", fpr=share(0, 4)),
        RealFpr(dataset="pisa_real", check_id="R1", fpr=share(2, 5)),
        RealFpr(dataset="unlisted", check_id="R1", fpr=share(0, 2)),
    ]
    report = evaluation(
        stratum(ORACLE, real=[RealFpr(dataset="physics_iq", check_id="R9", fpr=share(1, 1))]),
        stratum(PERCEIVED, real=rows),
        real_sets=[dataset("pisa_real", 10), dataset("physics_iq", 16), dataset("unscored", 3)],
    )
    found = real_alarms(report, frozenset({"R2"}), Criteria(real_fpr=0.1, real_fpr_upper=0.2))
    assert [(s.dataset, s.n_clips) for s in found.sets] == [
        ("pisa_real", 10),
        ("physics_iq", 16),
        ("unlisted", None),
    ]
    physics, unlisted = found.sets[1], found.sets[2]
    assert [(c.check_id, c.decided, c.excluded) for c in physics.checks] == [
        ("R1", 8, False),
        ("R2", 4, True),
    ]
    assert physics.checks[0].fpr == Value.of_rate(share(1, 8))
    assert physics.decided_share(physics.checks[0]) == 0.5
    assert unlisted.decided_share(unlisted.checks[0]) is None
    assert (found.limit, found.upper) == (0.1, 0.2)


def test_real_alarms_need_real_rows_with_perceived_inputs() -> None:
    oracle = stratum(ORACLE, real=[RealFpr(dataset="d", check_id="R1", fpr=share(0, 1))])
    for report in (evaluation(oracle), evaluation(oracle, stratum(PERCEIVED))):
        with pytest.raises(NoDataError, match="perceived inputs"):
            real_alarms(report, frozenset(), Criteria())


def test_pair_accuracy_of_b1_and_the_judge_with_scene_intervals() -> None:
    found = pair_accuracy(KUBRIC, config().pairs)
    assert found.judge == "judge"
    assert found.b1.value == pytest.approx(2 / 3)
    assert found.judged.value == pytest.approx(1 / 3)
    assert found.delta.value == pytest.approx(1 / 3)
    assert (found.n_pairs, found.unread) == (3, pytest.approx(1 / 6))
    assert all(bounded(v) for v in (found.b1, found.judged, found.delta))


def test_pair_accuracy_needs_kubric_scene_pairs() -> None:
    with pytest.raises(NoDataError, match="Kubric part"):
        pair_accuracy(REAL, config().pairs)
    lone = part("kubric", (clip("s1c", "s1", False, 0.1, 2.0),))
    with pytest.raises(NoDataError, match="no scene"):
        pair_accuracy(lone, config().pairs)


def test_labels_replace_the_defaults_and_write_numbers_with_the_chosen_decimal(
    tmp_path: Path,
) -> None:
    path = write(
        tmp_path / "labels.toml",
        '[format]\ndecimal = ","\n[fig]\nx_label = "Exactitud sobre {n} pares"\n'
        '[checks]\nR1 = "Gravedad"\n',
    )
    text = load_labels(path, LABELS).scope("fig")
    assert text("x_label", n=3) == "Exactitud sobre 3 pares"
    assert text("limit", value=text.number(0.1)) == "Limit (0,10)"
    assert text("oracle") == "Oracle inputs"
    assert (text.name("checks", "R1"), text.name("checks", "R2")) == ("Gravedad", "R2")
    assert text.name("stages", "track") == "track"
    assert (text.number(-0.25), text.tick(0.5), text.tick(2.0)) == (f"{MINUS}0,25", "0,5", "2")


def test_without_a_labels_file_the_english_defaults_and_the_point_hold() -> None:
    text = load_labels(None, LABELS).scope("unlabelled")
    assert text("oracle") == "Oracle inputs"
    assert text.number(0.5, 1) == "0.5"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ('[figure]\nx = "y"\n', "unknown label sections"),
        ('[fig]\ntitle = "T"\n', "unknown keys"),
        ('[format]\ndecimal = ";"\n', "decimal must be"),
        ("[fig]\nx_label = 3\n", "must be a string"),
        ("[checks]\nR1 = 1\n", "must be a string"),
        ('fig = "x"\n', "must be a string"),
        ("[fig\n", "unreadable labels file"),
    ],
)
def test_a_wrong_labels_file_is_a_config_error(tmp_path: Path, body: str, reason: str) -> None:
    with pytest.raises(ConfigError, match=reason):
        load_labels(write(tmp_path / "labels.toml", body), LABELS)


def test_no_figure_is_named_like_a_labels_section() -> None:
    assert not set(figures()) & {COMMON, FORMAT, *NAMES}


def test_locate_reads_the_state_file_and_takes_the_last_curve(tmp_path: Path) -> None:
    final = tmp_path / "final"
    kubric = write(final / "kubric.json", "{}")
    missing = final / "missing.json"
    state = f"KUBRIC_EVAL={kubric}\nREAL_EVAL={missing}\nVLM_KUBRIC=\n"
    write(final / RUN_DIR / STATE_FILE, state)
    for run in ("run_a", "run_b"):
        write(final / "runs" / run / REPORT_FILE, "{}")
    located = locate(final)
    assert located.paths == {
        Output.KUBRIC_EVAL: kubric,
        Output.CURVE: final / "runs" / "run_b" / REPORT_FILE,
    }
    assert located.gaps[Output.REAL_EVAL] == f"REAL_EVAL={missing} is not a file"
    assert located.gaps[Output.VLM_KUBRIC].endswith("records no VLM_KUBRIC")
    assert located.gaps[Output.ATTRIBUTION].endswith("records no ATTRIBUTION")


def test_locate_without_a_state_file_leaves_every_recorded_output_missing(tmp_path: Path) -> None:
    located = locate(tmp_path)
    assert located.paths == {}
    assert set(located.gaps) == set(Output)
    assert located.gaps[Output.KUBRIC_EVAL] == f"no {tmp_path / RUN_DIR / STATE_FILE}"
    with pytest.raises(NoDataError, match="no "):
        Records(located, config()).kubric()
    with pytest.raises(UsageError, match="not a directory"):
        locate(tmp_path / "absent")


def test_records_read_each_output_once_and_keep_a_malformed_one_as_its_reason(
    tmp_path: Path,
) -> None:
    rows = write(tmp_path / "rows.json", KUBRIC.to_json())
    broken = write(tmp_path / "attribution.json", "{")
    invalid = write(tmp_path / "evaluation.json", "{}")
    paths = {Output.VLM_KUBRIC: rows, Output.ATTRIBUTION: broken, Output.KUBRIC_EVAL: invalid}
    records = Records(Located(paths, {}), config())
    first = records.kubric_rows()
    rows.unlink()
    assert records.kubric_rows() is first
    with pytest.raises(MalformedInputError, match="ATTRIBUTION cannot be read"):
        records.attribution()
    write(broken, attribution(attributed("R1", 1, {Stage.TRACK: 0.5})).to_json())
    with pytest.raises(MalformedInputError, match="ATTRIBUTION cannot be read"):
        records.attribution()
    with pytest.raises(MalformedInputError, match="KUBRIC_EVAL cannot be read"):
        records.kubric()


def test_report_figures_without_matplotlib_is_a_usage_error_that_writes_nothing(
    cli: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    out = tmp_path / "figures"
    result = cli("report", "figures", "--final-dir", str(tmp_path), "--out", str(out))
    assert result.code == 2
    assert "matplotlib is not installed" in result.out + result.err
    assert not out.exists()


def test_select_keeps_the_registry_order_and_rejects_unknown_names() -> None:
    registry = figures()
    assert list(select(registry, ())) == list(registry)
    picked = select(registry, ("risk_coverage", "aggregate_auroc"))
    assert list(picked) == ["aggregate_auroc", "risk_coverage"]
    with pytest.raises(UsageError, match="unknown figures"):
        select(registry, ("histogram",))


def test_the_registry_is_built_without_loading_matplotlib() -> None:
    script = (
        "import sys\n"
        "from physverify.reporting.figures._base import figures\n"
        "print(*figures())\n"
        "print('matplotlib' in sys.modules)\n"
    )
    names, loaded = run_python("-c", script).splitlines()
    assert names.split() == sorted(names.split()) and len(names.split()) == len(figures())
    assert loaded == "False"
