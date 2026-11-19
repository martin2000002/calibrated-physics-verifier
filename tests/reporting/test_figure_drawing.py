"""Every registered figure draws a PDF from small hand-built data with the decimal comma, including
the uneven cases: a check missing from a stratum or a real set, a level missing from an input
level, a stage without an estimate, a real set without a clip count. A missing value is named in
its row, never drawn as a zero bar."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pytest
from comparison_cases import KUBRIC, config
from figure_cases import (
    LEVEL_CI,
    attributed,
    attribution,
    auroc,
    counted,
    curve,
    dataset,
    evaluation,
    heldout,
    point,
    share,
    stratum,
    swept,
)

from physverify.calibration.artifact import RealFpr
from physverify.reporting.figures._base import figures
from physverify.reporting.figures._labels import load_labels
from physverify.reporting.figures._publish import label_defaults
from physverify.reporting.figures._render import STYLE, save
from physverify.reporting.figures.aggregate_auroc import auroc_rows
from physverify.reporting.figures.check_detection import detection_panels
from physverify.reporting.figures.pair_accuracy import pair_accuracy
from physverify.reporting.figures.real_false_alarms import real_alarms
from physverify.reporting.figures.risk_coverage import risk_coverage
from physverify.reporting.figures.stage_attribution import stage_shares
from physverify.reporting.final_rows import Criteria
from physverify.schema.checks import InputLevel
from physverify.schema.stages import Stage

matplotlib = pytest.importorskip("matplotlib")

EVALUATION = evaluation(
    stratum(
        InputLevel.ORACLE,
        [
            heldout("R1", share(1, 10), [(0.1, share(4, 8)), (0.2, share(8, 8))]),
            heldout("R2", share(0, 10), [(1.0, share(2, 4))]),
        ],
        graded=auroc(0.9, 0.8, 1.0),
        plain=auroc(0.85, 0.7, 0.95),
    ),
    stratum(
        InputLevel.PERCEIVED,
        [
            heldout("R1", share(2, 10), [(0.1, share(3, 8)), (0.2, share(5, 8))]),
            heldout("R2", share(1, 10), [(1.5, share(1, 4))], first_level=2),
        ],
        [
            RealFpr(dataset="physics_iq", check_id="R1", fpr=share(1, 5)),
            RealFpr(dataset="physics_iq", check_id="R2", fpr=share(0, 3)),
            RealFpr(dataset="pisa_real", check_id="R2", fpr=share(1, 9)),
            RealFpr(dataset="unlisted", check_id="R1", fpr=share(0, 2)),
        ],
        graded=auroc(0.72, 0.6, 0.85),
    ),
    real_sets=[dataset("physics_iq", 6), dataset("pisa_real", 12)],
)
CURVE = curve(
    swept(point(2.0, 0.4, 0.05, 0.01, 0.09), point(1.0, 0.9, 0.2, 0.1, 0.3), best=0),
    swept(point(2.0, 0.5, 0.1), point(1.0, 1.0, 0.3)),
    k_star=1.0,
    per_check={"R1": {"k_2": counted(2.0, 6, 3, 1)}, "R2": {"k_2": counted(2.0, 4, 2, 0)}},
)
ATTRIBUTION = attribution(
    attributed(
        "R5a",
        3,
        {Stage.TRACK: 0.1, Stage.DEPTH: -0.02, Stage.SCALE: math.nan},
        {Stage.TRACK: (0.0, 0.2)},
    ),
    attributed("R4_size", 2, {Stage.DEPTH: 0.3}),
)
CASES: dict[str, Callable[[], object]] = {
    "aggregate_auroc": lambda: auroc_rows(EVALUATION, 0.7),
    "check_detection": lambda: detection_panels(EVALUATION),
    "pair_accuracy": lambda: pair_accuracy(KUBRIC, config().pairs),
    "real_false_alarms": lambda: real_alarms(EVALUATION, frozenset({"R2"}), Criteria()),
    "risk_coverage": lambda: risk_coverage(CURVE, 0.1, LEVEL_CI),
    "stage_attribution": lambda: stage_shares(ATTRIBUTION),
}


def test_every_registered_figure_has_a_drawing_case() -> None:
    assert set(CASES) == set(figures())


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_figure_draws_a_pdf(name: str, tmp_path: Path) -> None:
    registry = figures()
    labels_file = tmp_path / "labels.toml"
    labels_file.write_text('[format]\ndecimal = ","\n', encoding="utf-8")
    text = load_labels(labels_file, label_defaults(registry)).scope(name)
    figure, data, path = registry[name](), CASES[name](), tmp_path / f"{name}.pdf"
    save(lambda: figure.draw(data, text), path)
    assert path.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize(
    ("name", "mark", "n_bars"),
    [("real_false_alarms", "no clip count", 3), ("stage_attribution", "not estimated", 3)],
)
def test_a_missing_value_is_named_in_its_row_and_draws_no_bar(
    name: str, mark: str, n_bars: int
) -> None:
    registry = figures()
    text = load_labels(None, label_defaults(registry)).scope(name)
    canvas = registry[name]().draw(CASES[name](), text)
    assert [t.get_text() for ax in canvas.axes for t in ax.texts] == [mark]
    assert sum(len(ax.patches) for ax in canvas.axes) == n_bars
    low, high = sorted(canvas.axes[0].get_ylim())
    rows = canvas.axes[0].get_yticks()
    assert low < min(rows) and max(rows) < high


def test_a_narrow_attribution_panel_keeps_few_ticks() -> None:
    small = attribution(
        attributed(
            "R4_missing",
            4,
            {Stage.TRACK: -0.0101, Stage.DEPTH: -0.0004},
            {Stage.TRACK: (-0.0112, -0.009)},
        )
    )
    registry = figures()
    text = load_labels(None, label_defaults(registry)).scope("stage_attribution")
    with matplotlib.rc_context(dict(STYLE)):
        canvas = registry["stage_attribution"]().draw(stage_shares(small), text)
        canvas.draw_without_rendering()
        ax = canvas.axes[0]
        low, high = sorted(ax.get_xlim())
        ticks = [tick for tick in ax.get_xticks() if low <= tick <= high]
    assert len(ticks) <= 4


def risk_coverage_legend(data: object) -> tuple[list[str], list[str]]:
    registry = figures()
    text = load_labels(None, label_defaults(registry)).scope("risk_coverage")
    canvas = registry["risk_coverage"]().draw(data, text)
    (legend,) = canvas.legends
    return [t.get_text() for t in legend.get_texts()], [t.get_text() for t in canvas.axes[0].texts]


def test_risk_coverage_shows_both_reported_points_in_the_legend() -> None:
    shown, named = risk_coverage_legend(CASES["risk_coverage"]())
    assert {"Calibration $k^*$ ($k$ = 1.00)", "Configured $k$ ($k$ = 2.00)"} <= set(shown)
    assert not named


def test_risk_coverage_names_in_the_legend_a_point_it_cannot_place_and_draws_no_marker() -> None:
    data = risk_coverage(curve(CURVE.tables.calibrated, CURVE.tables.raw), 0.1, LEVEL_CI)
    shown, named = risk_coverage_legend(data)
    gaps = [
        "No calibration $k^*$: no $k$ met the target",
        "Configured $k$: no exact sum over checks",
    ]
    assert set(gaps) <= set(shown)
    assert not any("($k$ =" in label for label in shown)
    assert not named
