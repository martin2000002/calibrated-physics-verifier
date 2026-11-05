"""The ``verify-separate`` service: the rows reported apart from the nine checks (ADR 0050).

They run unscored (``raw`` policy, no thresholds) until a calibration of their own exists.
"""

from __future__ import annotations

from collections.abc import Sequence

from physverify.app.context import AppContext
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import dataset_for
from physverify.app.verification import input_source
from physverify.checks.separate._base import SEPARATE_CHECKS
from physverify.config.plugin_context import registry_of
from physverify.core.output import Echo
from physverify.datasets._base import select_clips
from physverify.runtime.runs import new_run
from physverify.schema.stages import parse_stages
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import RowStatus, VerificationRunner, write_report
from physverify.verification.summary import parse_level, summary_lines
from physverify.verification.thresholding import ThresholdTable


def run_separate(
    context: AppContext,
    command: str,
    dataset: str,
    level: str | None,
    gt: Sequence[str],
    *,
    subset: str | None,
    clips: Sequence[str],
    echo: Echo,
    err: Echo,
) -> bool:
    """Run the separate rows with their default parameters; whether a row errored."""
    input_level = parse_level(level, gt, sweep=False)
    source = dataset_for(context, dataset)
    selected = select_clips(source.clips(subset), clips)
    checks = [cls(cls.Params()) for cls in registry_of(context, SEPARATE_CHECKS).values()]
    inputs = input_source(context, source, input_level, parse_stages(gt))
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    echo(f"run: {run_dir}")
    guard_sealed(
        context,
        command,
        [Selection(source.name, subset, tuple(selected))],
        err=err,
        run_dir=run_dir,
    )
    runner = VerificationRunner(checks, RawPolicy(), ThresholdTable.provisional({}))
    report = runner.run(inputs, selected)
    write_report(run_dir, report)
    for line in summary_lines(report):
        echo(line)
    return any(row.status is RowStatus.ERROR for row in report.table)
