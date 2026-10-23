"""Byte-exact text of the reports' number cells and Markdown tables."""

import importlib
from collections.abc import Callable
from enum import Enum
from typing import Any

import pytest

from physverify.core.markdown import table
from physverify.datasets.local_video import QueryEntry
from physverify.demo.report import CheckRow, ClipBlock, RuleInfo, VideoReport, render_markdown
from physverify.diagnosis.report import _table as diagnosis_table
from physverify.finetune.records import PromptRecord
from physverify.reporting.numbers import Section, Value
from physverify.reporting.render import _table as results_table
from physverify.timing.report import _table as timing_table

NAN = float("nan")
INF = float("inf")


class Colour(Enum):
    RED = "red"


FORMATTERS: dict[str, list[tuple[Any, str]]] = {
    "physverify.cli.commands.calibrate:_num": [
        (None, "-"),
        (0.0, "0"),
        (-0.0, "-0"),
        (1.0, "1"),
        (-2.5, "-2.5"),
        (0.1234567, "0.1235"),
        (12345.678, "1.235e+04"),
        (1e-07, "1e-07"),
        (123456789.0, "1.235e+08"),
        (3.0, "3"),
        (0.5, "0.5"),
        (NAN, "nan"),
        (INF, "inf"),
        (-INF, "-inf"),
        (0, "0"),
        (3, "3"),
        (-12, "-12"),
    ],
    "physverify.calibration.report:number_text": [
        (None, "-"),
        (0.0, "0"),
        (-0.0, "-0"),
        (1.0, "1"),
        (-2.5, "-2.5"),
        (0.1234567, "0.123"),
        (12345.678, "1.23e+04"),
        (1e-07, "1e-07"),
        (123456789.0, "1.23e+08"),
        (3.0, "3"),
        (0.5, "0.5"),
        (NAN, "nan"),
        (INF, "inf"),
        (-INF, "-inf"),
        (0, "0"),
        (3, "3"),
        (-12, "-12"),
    ],
    "physverify.demo.report:_num": [
        (None, "-"),
        (0.0, "0"),
        (-0.0, "-0"),
        (1.0, "1"),
        (-2.5, "-2.5"),
        (0.1234567, "0.123"),
        (12345.678, "1.23e+04"),
        (1e-07, "1e-07"),
        (123456789.0, "1.23e+08"),
        (3.0, "3"),
        (0.5, "0.5"),
        (NAN, "-"),
        (INF, "-"),
        (-INF, "-"),
        (0, "0"),
        (3, "3"),
        (-12, "-12"),
    ],
    "physverify.evaluation.report:_num": [
        (None, "-"),
        (0.0, "0"),
        (-0.0, "-0"),
        (1.0, "1"),
        (-2.5, "-2.5"),
        (0.1234567, "0.1235"),
        (12345.678, "1.235e+04"),
        (1e-07, "1e-07"),
        (123456789.0, "1.235e+08"),
        (3.0, "3"),
        (0.5, "0.5"),
        (NAN, "nan"),
        (INF, "inf"),
        (-INF, "-inf"),
        (0, "0"),
        (3, "3"),
        (-12, "-12"),
    ],
    "physverify.cli.commands.verify:_number": [
        (None, "-"),
        (0.0, "0"),
        (-0.0, "-0"),
        (1.0, "1"),
        (-2.5, "-2.5"),
        (0.1234567, "0.1235"),
        (12345.678, "1.235e+04"),
        (1e-07, "1e-07"),
        (123456789.0, "1.235e+08"),
        (3.0, "3"),
        (0.5, "0.5"),
        (NAN, "-"),
        (INF, "inf"),
        (-INF, "-inf"),
        (0, "0"),
        (3, "3"),
        (-12, "-12"),
    ],
    "physverify.cli.commands.diagnose:_number": [
        (None, "-"),
        (0.0, "0"),
        (-0.0, "-0"),
        (1.0, "1"),
        (-2.5, "-2.5"),
        (0.1234567, "0.123"),
        (12345.678, "1.23e+04"),
        (1e-07, "1e-07"),
        (123456789.0, "1.23e+08"),
        (3.0, "3"),
        (0.5, "0.5"),
        (NAN, "-"),
        (INF, "inf"),
        (-INF, "-inf"),
        (0, "0"),
        (3, "3"),
        (-12, "-12"),
    ],
    "physverify.cli.commands.size:_number": [
        (None, "n/a"),
        (0.0, "0.000"),
        (-0.0, "-0.000"),
        (1.0, "1.000"),
        (-2.5, "-2.500"),
        (0.1234567, "0.123"),
        (12345.678, "12345.678"),
        (1e-07, "0.000"),
        (123456789.0, "123456789.000"),
        (3.0, "3.000"),
        (0.5, "0.500"),
        (NAN, "nan"),
        (INF, "inf"),
        (-INF, "-inf"),
        (0, "0.000"),
        (3, "3.000"),
        (-12, "-12.000"),
    ],
    "physverify.timing.report:_number": [
        (0.0, "0"),
        (-0.0, "-0"),
        (1.0, "1"),
        (-2.5, "-2.5"),
        (0.1234567, "0.1235"),
        (12345.678, "1.235e+04"),
        (1e-07, "1e-07"),
        (123456789.0, "1.235e+08"),
        (3.0, "3"),
        (0.5, "0.5"),
        (NAN, "nan"),
        (INF, "inf"),
        (-INF, "-inf"),
        (0, "0"),
        (3, "3"),
        (-12, "-12"),
    ],
    "physverify.human_agreement.report:_fmt": [
        (None, "n/a"),
        (0.0, "0.000"),
        (-0.0, "-0.000"),
        (1.0, "1"),
        (-2.5, "-2.500"),
        (0.1234567, "0.123"),
        (12345.678, "12345.678"),
        (1e-07, "0.000"),
        (123456789.0, "123456789"),
        (3.0, "3"),
        (0.5, "0.500"),
        (NAN, "n/a"),
        (INF, "n/a"),
        (-INF, "n/a"),
        (0, "0"),
        (3, "3"),
        (-12, "-12"),
        (True, "True"),
        (False, "False"),
        ("a|b", "a|b"),
        ("", ""),
        (("x", 1.5, None), "('x', 1.5, None)"),
        ((), "()"),
        (Colour.RED, "Colour.RED"),
    ],
    "physverify.diagnosis.report:_cell": [
        (None, "n/a"),
        (0.0, "0"),
        (-0.0, "-0"),
        (1.0, "1"),
        (-2.5, "-2.5"),
        (0.1234567, "0.1235"),
        (12345.678, "1.235e+04"),
        (1e-07, "1e-07"),
        (123456789.0, "1.235e+08"),
        (3.0, "3"),
        (0.5, "0.5"),
        (NAN, "n/a"),
        (INF, "n/a"),
        (-INF, "n/a"),
        (0, "0"),
        (3, "3"),
        (-12, "-12"),
        (True, "yes"),
        (False, "no"),
        ("a|b", "a\\|b"),
        ("", ""),
        (("x", 1.5, None), "x, 1.5, n/a"),
        ((), "-"),
        (Colour.RED, "red"),
    ],
}


def _formatter(site: str) -> Callable[[Any], str]:
    module, name = site.split(":")
    formatter: Callable[[Any], str] = getattr(importlib.import_module(module), name)
    return formatter


@pytest.mark.parametrize(
    ("site", "value", "expected"),
    [(site, value, text) for site, cases in FORMATTERS.items() for value, text in cases],
)
def test_number_cells_keep_their_text(site: str, value: Any, expected: str) -> None:
    assert _formatter(site)(value) == expected


PLAIN_ROWS = [["1", "x|y"], ["", "2"]]
PLAIN_TABLE = ["| a | b |", "|---|---|", "| 1 | x|y |", "|  | 2 |"]


def test_plain_tables_keep_their_text() -> None:
    assert table(["a", "b"], PLAIN_ROWS) == PLAIN_TABLE


def test_converting_tables_keep_their_text() -> None:
    assert diagnosis_table(("a", "b"), [(None, 1.5), (True, ("p", 2.0))]) == [
        "| a | b |",
        "|---|---|",
        "| n/a | 1.5 |",
        "| yes | p, 2 |",
        "",
    ]
    assert timing_table(("a", "b"), [(1, 2.5), ("s", None)]) == [
        "| a | b |",
        "|---|---|",
        "| 1 | 2.5 |",
        "| s | None |",
    ]


def test_results_sections_keep_their_text() -> None:
    values = (
        Value(
            key="auroc",
            value=0.81234,
            source="e.json#/a",
            ci_low=0.7,
            ci_high=0.9,
            k=8,
            n=10,
            unit="",
            estimated=True,
            note="a|b",
        ),
        Value(key="count", value=12, source="e.json#/n", n=40, reportable=False),
        Value(key="ld", value=None, source="c.json#/ld", censored="below", bound=0.05, unit="m"),
    )
    assert results_table(Section(key="h1", values=values, caveat="careful")) == [
        "## h1",
        "",
        "Note: careful",
        "",
        "| key | value | CI | k/n | unit | estimated | reportable | note | source |",
        "|---|---|---|---|---|---|---|---|---|",
        "| auroc | 0.812 | [0.7, 0.9] | 8/10 |  | estimate |  | a\\|b | `e.json#/a` |",
        "| count | 12 |  | n 40 |  |  | no (rehearsal) |  | `e.json#/n` |",
        "| ld | <= 0.05 |  |  | m |  |  |  | `c.json#/ld` |",
        "",
    ]
    assert results_table(Section(key="h2", missing="no agreement run")) == [
        "## h2",
        "",
        "Missing: no agreement run",
        "",
    ]


def _demo_report(calibrated: bool) -> VideoReport:
    prompt = PromptRecord(
        prompt_id="p",
        text="a ball falls",
        queries=[QueryEntry(phrase="ball", expected_count=1)],
        family="fall",
        template_id="fixed",
        target_checks=["R2"],
        split="eval",
        tags=[],
    )
    rows = [
        CheckRow(
            check_id="R1",
            status="scored",
            verdict="PASS",
            magnitude=0.2,
            sigma=0.0043,
            unit="m",
            threshold=1.5,
            abstain_reasons=[],
            frame_span=(2, 4),
            n_subjects=1,
            rule_decided=True,
            rule_term=0.8333,
        ),
        CheckRow(
            check_id="R3a",
            status="abstained",
            verdict="ABSTAIN",
            magnitude=None,
            sigma=None,
            unit="",
            threshold=None,
            abstain_reasons=["MISSING_INPUT", "LOW"],
            frame_span=None,
            n_subjects=0,
            rule_decided=False,
            rule_term=NAN,
        ),
    ]
    clip = ClipBlock(
        score=0.61234,
        sigma=None,
        coverage=2 / 3,
        abstained=False,
        reason=None,
        n_decided=2,
        n_failed=1,
        motion=INF,
        objects_ok=None,
    )
    rule = RuleInfo(
        scorer="provisional",
        calibrated=calibrated,
        label="tau",
        note=None if calibrated else "no artifact",
    )
    return VideoReport(
        prompt=prompt,
        model_tag="base",
        candidate_id="p__k00",
        video_path="v.mp4",
        results_path="r.jsonl",
        rule=rule,
        clip=clip,
        checks=rows,
        provenance={"commit": "abc"},
        timings={"score_s": 1.5},
    )


DEMO_BODY = (
    "- Prompt (p, family fall): a ball falls\n- Video: `v.mp4`\n"
    "- Rule: provisional, status **{status}**, label `tau`\n"
    "- S = 0.612 +/- unknown; coverage 0.67; 2 checks decided, 1 failed\n"
    "- Motion index -; expected objects found: -\n"
    "- Magnitudes from perceived inputs are estimates; sigma is their propagated error "
    "('-' when not yet measured).\n\n## Checks\n\n"
    "| check | verdict | status | magnitude | sigma | threshold | abstain reasons | frames "
    "| decided by rule | rule term |\n|---|---|---|---|---|---|---|---|---|---|\n"
    "| R1 | PASS | scored | 0.2 m | 0.0043 | 1.5 | - | 2-4 | yes | 0.833 |\n"
    "| R3a | ABSTAIN | abstained | - | - | - | MISSING_INPUT, LOW | - | no | - |\n\n"
    "Rule term: {term}.\n\n## Provenance\n\n- commit: abc\n- score_s: 1.5 s\n"
)


def test_demo_reports_keep_their_text() -> None:
    title = "# Verifier report: p__k00 (base)\n\n"
    banner = "> **PROVISIONAL RULE: not a calibrated verdict.**\n\n> no artifact\n\n"
    provisional = DEMO_BODY.format(
        status="provisional", term="s_j = 1 / (1 + m/tau), provisional factor"
    )
    calibrated = DEMO_BODY.format(status="calibrated", term="p_c, estimated violation probability")
    assert render_markdown(_demo_report(False)) == title + banner + provisional
    assert render_markdown(_demo_report(True)) == title + calibrated
