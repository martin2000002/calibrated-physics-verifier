"""``calibrate compare-devscore``: paired deltas b - a of two devscore runs (ADR 0036)."""

from __future__ import annotations

from pathlib import Path

from physverify.app.context import AppContext
from physverify.calibration.devscore_compare import DEVSCORE_COMPARE_FILE, compare_oof
from physverify.calibration.oof import DEVSCORE_OOF_FILE, DevScoreOof
from physverify.calibration.summary import compare_lines
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import UsageError
from physverify.core.output import Echo
from physverify.runtime.runs import new_run
from physverify.schema.checks import InputLevel
from physverify.synth.plan import family_of


def load_oof(path: Path) -> DevScoreOof:
    """``path`` is a dev-score run directory or its ``devscore_oof.json``."""
    found = path / DEVSCORE_OOF_FILE if path.is_dir() else path
    if not found.is_file():
        raise UsageError(f"no {DEVSCORE_OOF_FILE} at {path}: run `calibrate dev-score` (1.2.0+)")
    return DevScoreOof.model_validate_json(found.read_text(encoding="utf-8"))


def compare_devscore(
    context: AppContext, a: Path, b: Path, level: str, out: Path | None, *, echo: Echo
) -> None:
    if level not in (InputLevel.ORACLE.value, InputLevel.PERCEIVED.value):
        raise UsageError("--level must be oracle or perceived")
    found = compare_oof(load_oof(a), load_oof(b), InputLevel(level), family_of)
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    text = found.to_json()
    for path in (run_dir / DEVSCORE_COMPARE_FILE, *(() if out is None else (out,))):
        write_text_atomic(path, text)
        echo(f"devscore compare: {path}")
    for line in compare_lines(found):
        echo(line)
