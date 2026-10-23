"""The printed reports of ``physverify size eval`` and ``size e2e-eval``."""

from __future__ import annotations

from physverify.app.context import AppContext
from physverify.app.sizing import evaluate_end_to_end, evaluate_test_split
from physverify.sizing.end_to_end import e2e_lines
from physverify.sizing.evaluate import eval_lines


def evaluation_lines(context: AppContext) -> list[str]:
    """The error on the official test split against the baselines, then the run directory."""
    report, run_dir = evaluate_test_split(context)
    return [*eval_lines(report), f"run: {run_dir}"]


def end_to_end_lines(context: AppContext) -> list[str]:
    """The perceived scale against ARKit depth on the prepared clips, then the run directory."""
    report, run_dir = evaluate_end_to_end(context)
    return [*e2e_lines(report), f"run: {run_dir}"]
