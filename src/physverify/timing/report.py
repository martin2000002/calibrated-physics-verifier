"""Layout and summary of a timing run under ``data/runs/<run_id>/timing/``.

``stats.csv`` and ``summary.md`` are always rebuilt from ``samples.csv``.
"""

from __future__ import annotations

import contextlib
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from physverify.core.atomic import write_bytes_atomic
from physverify.core.markdown import number, table
from physverify.timing.records import (
    CallFailure,
    MeasureStats,
    ResultsFile,
    Sample,
    TargetResult,
    unit_of,
)
from physverify.timing.results_io import (
    TIMING_DIR,
    flag,
    load_failures,
    load_results,
    load_samples,
    write_results,
    write_stats,
)
from physverify.timing.stats import ACROSS_CLIPS, across_medians, stat_rows

SUMMARY: Final = "summary.md"
SCRATCH_DIR: Final = "scratch"
BANNER: Final = (
    "> HARNESS CHECK ONLY: not a cost measurement ({reasons}). "
    "Per-clip cost is measured only on the H200."
)
TOTAL_LABEL: Final = "per-clip total (sum of medians)"


def timing_dir(run_dir: Path) -> Path:
    return run_dir / TIMING_DIR


def scratch_dir(run_dir: Path, label: str, index: int) -> Path:
    """``<run_dir>/timing/scratch/<label>-<index>``: working data of one call, deleted after it."""
    return timing_dir(run_dir) / SCRATCH_DIR / f"{label}-{index}"


def discard_scratch(scratch: Path) -> None:
    """Deletes one call's scratch directory, then ``scratch/`` itself once no call uses it."""
    shutil.rmtree(scratch, ignore_errors=True)
    with contextlib.suppress(OSError):
        scratch.parent.rmdir()


def _number(value: float) -> str:
    return number(value, ".4g", "nan", "nan")


def _stat_text(row: MeasureStats) -> str:
    unit = f" {row.unit}"
    return f"{_number(row.median)}{unit} [{_number(row.p25)}-{_number(row.p75)}] ({row.n})"


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    return table(header, ([str(cell) for cell in cells] for cells in rows))


def _gpu_text(environment: Mapping[str, Any]) -> str:
    gpus = environment.get("gpus") or []
    return ", ".join(str(gpu.get("name", "?")) for gpu in gpus) or "none"


def _environment_lines(results: ResultsFile) -> list[str]:
    environment, run = results.environment, results.run
    packages = environment.get("packages") or {}
    rows = [
        ("GPU", _gpu_text(environment)),
        ("driver", environment.get("driver")),
        ("CUDA", environment.get("cuda")),
        ("cuDNN", environment.get("cudnn")),
        ("torch", packages.get("torch")),
        ("diffusers", packages.get("diffusers")),
        ("physverify", packages.get("physverify")),
        ("device", environment.get("device")),
        ("profile", environment.get("profile")),
        ("threads", environment.get("threads")),
        ("usable CPUs", environment.get("usable_cpus")),
        ("commit", run.get("commit")),
        ("config hash", run.get("config_hash")),
    ]
    return ["## Environment", "", *_table(("item", "value"), rows), ""]


def _targets_lines(targets: Sequence[TargetResult]) -> list[str]:
    rows = [
        (
            target.name,
            target.status.value,
            flag(target.requires_gpu),
            target.run.warmup,
            target.run.warmup_scope,
            target.run.repetitions,
            len(target.clip_ids),
            target.n_calls,
            len(target.failures),
            target.reason or "",
        )
        for target in targets
    ]
    header = (
        "target",
        "status",
        "requires GPU",
        "warm-up",
        "scope",
        "repetitions",
        "clips",
        "calls",
        "failures",
        "reason",
    )
    return ["## Targets", "", *_table(header, rows), ""]


def _target_lines(target: TargetResult) -> list[str]:
    lines = [f"## {target.name}", ""]
    if target.setup:
        setup = (
            (name, f"{_number(value)} {unit_of(name)}") for name, value in target.setup.items()
        )
        lines += ["Setup, once per run:", "", *_table(("measure", "value"), setup), ""]
    across = [row for row in target.stats if row.clip_id == ACROSS_CLIPS]
    if across:
        rows = (
            (row.measure, _stat_text(row), "unstable" if row.unstable else "") for row in across
        )
        lines += [
            "Across clips, over per-clip medians: median [p25-p75] (clips).",
            "",
            *_table(("measure", "median [p25-p75] (n)", "note"), rows),
            "",
        ]
    lines += [f"- {key}: {value}" for key, value in target.info.items()]
    return [*lines, ""] if target.info else lines


def _failure_lines(targets: Sequence[TargetResult]) -> list[str]:
    failures = [failure for target in targets for failure in target.failures]
    if not failures:
        return []
    items = [
        f"- {failure.target} {failure.clip_id} call {failure.repetition}: {failure.error}"
        for failure in failures
    ]
    return ["## Failures", "", *items, ""]


def total_line(rows: Sequence[MeasureStats], total_measures: Sequence[str]) -> str:
    medians = across_medians(rows)
    missing = [measure for measure in total_measures if measure not in medians]
    if missing:
        return f"{TOTAL_LABEL}: not available, missing {', '.join(missing)}"
    parts = " + ".join(f"{measure} {_number(medians[measure])}" for measure in total_measures)
    total = math.fsum(medians[measure] for measure in total_measures)
    return f"{TOTAL_LABEL}: {_number(total)} s = {parts}"


def banner(results: ResultsFile) -> str | None:
    if results.valid_measurement:
        return None
    return BANNER.format(reasons="; ".join(results.validity_reasons))


def render_summary(results: ResultsFile, total_measures: Sequence[str]) -> str:
    head = banner(results)
    lines = [head, ""] if head else []
    lines += [f"# Timing run {results.run.get('run_id', '?')}", ""]
    lines += _environment_lines(results) + _targets_lines(results.targets)
    for target in results.targets:
        lines += _target_lines(target)
    lines += _failure_lines(results.targets)
    if results.valid_measurement:
        rows = [row for target in results.targets for row in target.stats]
        lines += [total_line(rows, total_measures), ""]
    return "\n".join(lines)


def _calls_seen(samples: Sequence[Sample], failures: Sequence[CallFailure], name: str) -> int:
    seen = {(s.clip_id, s.repetition) for s in samples if s.target == name}
    seen |= {(f.clip_id, f.repetition) for f in failures if f.target == name}
    return len(seen)


def _rebuilt_target(
    target: TargetResult,
    rows: Sequence[MeasureStats],
    samples: Sequence[Sample],
    failures: Sequence[CallFailure],
) -> TargetResult:
    own_failures = [failure for failure in failures if failure.target == target.name]
    return target.model_copy(
        update={
            "stats": [row for row in rows if row.target == target.name],
            "failures": own_failures if own_failures else target.failures,
            "n_calls": max(target.n_calls, _calls_seen(samples, own_failures, target.name)),
        }
    )


def rebuild(directory: Path, max_iqr_rel: float, total_measures: Sequence[str]) -> ResultsFile:
    """Recompute ``stats.csv``, the statistics in ``results.json`` and ``summary.md`` from
    ``samples.csv`` (and ``failures.jsonl``)."""
    results = load_results(directory)
    samples = load_samples(directory)
    failures = load_failures(directory)
    rows = stat_rows(samples, max_iqr_rel)
    targets = [_rebuilt_target(target, rows, samples, failures) for target in results.targets]
    results = results.model_copy(update={"targets": targets})
    write_results(directory, results)
    write_stats(directory, rows)
    summary = render_summary(results, total_measures)
    write_bytes_atomic(directory / SUMMARY, summary.encode("utf-8"))
    return results
