"""``pv report figures`` on a tiny finished run: each figure whose inputs the run wrote becomes a
PDF, the others are listed as skipped with the reason."""

from __future__ import annotations

from pathlib import Path

import pytest
from cli_runner import CliRunner
from comparison_cases import KUBRIC
from figure_cases import auroc, dataset, evaluation, heldout, share, stratum
from reporting_cases import write

from physverify.calibration.artifact import RealFpr
from physverify.reporting.chain import STATE_FILE
from physverify.reporting.figures._records import RUN_DIR
from physverify.schema.checks import InputLevel

pytest.importorskip("matplotlib")

DRAWN = ("aggregate_auroc", "check_detection", "pair_accuracy", "real_false_alarms")
LABELS = '[format]\ndecimal = ","\n[pair_accuracy]\ndelta = "B1 menos el juez"\n'


def final_run(root: Path) -> Path:
    """A run that wrote both evaluations and the Kubric comparison rows, but no attribution and
    no selective-risk curve."""
    checks = [heldout("R1", share(1, 10), [(0.1, share(4, 8)), (0.2, share(7, 8))])]
    real = [RealFpr(dataset="physics_iq", check_id="R1", fpr=share(1, 5))]
    report = evaluation(
        stratum(InputLevel.ORACLE, checks, graded=auroc(0.9, 0.8, 1.0)),
        stratum(InputLevel.PERCEIVED, checks, real, graded=auroc(0.75, 0.6, 0.9)),
        real_sets=[dataset("physics_iq", 6)],
    )
    scored = write(root / "evaluation.json", report.to_json())
    rows = write(root / "kubric_rows.json", KUBRIC.to_json())
    final = root / "final"
    state = f"KUBRIC_EVAL={scored}\nREAL_EVAL={scored}\nVLM_KUBRIC={rows}\n"
    write(final / RUN_DIR / STATE_FILE, state)
    return final


def draw(cli: CliRunner, final: Path, out: Path, *more: str) -> tuple[int, str]:
    result = cli("report", "figures", "--final-dir", str(final), "--out", str(out), *more)
    return result.code, result.out + result.err


def test_report_figures_draws_a_pdf_per_figure_with_inputs_and_lists_the_rest(
    cli: CliRunner, tmp_path: Path
) -> None:
    final, out = final_run(tmp_path), tmp_path / "figures"
    labels = write(tmp_path / "labels.toml", LABELS)
    code, text = draw(cli, final, out, "--labels", str(labels))
    assert code == 0, text
    assert sorted(p.stem for p in out.iterdir()) == list(DRAWN)
    pdfs = {p.name: p.read_bytes() for p in out.iterdir()}
    assert all(body.startswith(b"%PDF") for body in pdfs.values())
    assert "skipped risk_coverage: no runs/" in text
    assert "skipped stage_attribution:" in text and "records no ATTRIBUTION" in text
    assert draw(cli, final, out, "--labels", str(labels))[0] == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == pdfs


def test_report_figures_draws_only_the_named_ones(cli: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "figures"
    code, text = draw(cli, final_run(tmp_path), out, "--only", "pair_accuracy")
    assert code == 0, text
    assert [p.name for p in out.iterdir()] == ["pair_accuracy.pdf"]
    code, text = draw(cli, final_run(tmp_path), out, "--only", "histogram")
    assert code != 0 and "unknown figures" in text


def test_a_malformed_output_fails_its_figure_while_the_others_are_drawn(
    cli: CliRunner, tmp_path: Path
) -> None:
    final, out = final_run(tmp_path), tmp_path / "figures"
    write(tmp_path / "kubric_rows.json", '{"part": "kubric"')
    code, text = draw(cli, final, out)
    assert code == 1
    assert "failed pair_accuracy: VLM_KUBRIC cannot be read" in text
    assert sorted(p.stem for p in out.iterdir()) == [n for n in DRAWN if n != "pair_accuracy"]


def test_report_figures_fails_when_nothing_can_be_drawn(cli: CliRunner, tmp_path: Path) -> None:
    code, text = draw(cli, tmp_path, tmp_path / "figures")
    assert code == 1
    assert text.count("skipped ") == len(DRAWN) + 2
