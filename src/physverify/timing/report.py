"""Files of a timing run under ``data/runs/<run_id>/timing/``.

``samples.csv`` and ``failures.jsonl`` grow row by row while the run goes, and ``results.json`` is
rewritten after each target, so an interrupted run keeps what it measured. ``stats.csv`` and
``summary.md`` are always rebuilt from ``samples.csv``.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import math
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Final

from pydantic import ValidationError

from physverify.core.atomic import write_bytes_atomic
from physverify.core.markdown import number, table
from physverify.timing.records import (
    CallFailure,
    MeasureStats,
    ResultsFile,
    Sample,
    TargetOutcome,
    TargetResult,
    TimingError,
    unit_of,
)
from physverify.timing.stats import ACROSS_CLIPS, across_medians, stat_rows

TIMING_DIR: Final = "timing"
RESULTS: Final = "results.json"
SAMPLES: Final = "samples.csv"
STATS: Final = "stats.csv"
SUMMARY: Final = "summary.md"
SCRATCH_DIR: Final = "scratch"
FAILURES: Final = "failures.jsonl"
SAMPLE_COLUMNS: Final = ("target", "clip_id", "repetition", "warmup", "measure", "unit", "value")
STAT_COLUMNS: Final = tuple(MeasureStats.model_fields)
BANNER: Final = (
    "> HARNESS CHECK ONLY: not a cost measurement ({reasons}). "
    "Per-clip cost is measured only on the H200."
)
TOTAL_LABEL: Final = "per-clip total (sum of medians)"

_log = logging.getLogger(__name__)


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


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _parse_flag(text: str) -> bool:
    if text not in ("true", "false"):
        raise ValueError(f"expected true or false, got {text!r}")
    return text == "true"


class CsvSampleSink:
    """``SampleSink`` that flushes each sample to ``samples.csv`` and each failure to
    ``failures.jsonl`` as soon as it arrives."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._samples: IO[str] | None = None
        self._failures: IO[str] | None = None

    def __enter__(self) -> CsvSampleSink:
        self._directory.mkdir(parents=True, exist_ok=True)
        self._samples = (self._directory / SAMPLES).open("w", newline="", encoding="utf-8")
        self._failures = (self._directory / FAILURES).open("w", encoding="utf-8")
        self._write_row(SAMPLE_COLUMNS)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for handle in (self._samples, self._failures):
            if handle is not None:
                handle.close()
        self._samples = self._failures = None

    def add(self, sample: Sample) -> None:
        self._write_row(
            (
                sample.target,
                sample.clip_id,
                sample.repetition,
                _flag(sample.warmup),
                sample.measure,
                sample.unit,
                repr(float(sample.value)),
            )
        )

    def fail(self, failure: CallFailure) -> None:
        handle = self._open(self._failures)
        handle.write(json.dumps(asdict(failure)) + "\n")
        handle.flush()

    def _write_row(self, row: Sequence[object]) -> None:
        handle = self._open(self._samples)
        csv.writer(handle).writerow(row)
        handle.flush()

    @staticmethod
    def _open(handle: IO[str] | None) -> IO[str]:
        if handle is None:
            raise TimingError("CsvSampleSink is used outside its `with` block")
        return handle


def write_results(directory: Path, results: ResultsFile) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(directory / RESULTS, results.model_dump_json(indent=2).encode("utf-8"))


def load_results(directory: Path) -> ResultsFile:
    path = directory / RESULTS
    try:
        return ResultsFile.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TimingError(f"{path} does not exist: not a timing run directory") from exc
    except ValidationError as exc:
        raise TimingError(f"{path} is not a valid timing results file: {exc}") from exc


class ResultsWriter:
    """Keeps ``results.json`` current: written at start, rewritten after each target."""

    def __init__(self, directory: Path, results: ResultsFile) -> None:
        self._directory = directory
        self.results = results
        write_results(directory, results)

    def record(self, outcome: TargetOutcome) -> None:
        targets = list(self.results.targets)
        names = [target.name for target in targets]
        result = TargetResult.from_outcome(outcome)
        if outcome.name in names:
            targets[names.index(outcome.name)] = result
        else:
            targets.append(result)
        self.results = self.results.model_copy(update={"targets": targets})
        write_results(self._directory, self.results)


def _sample(row: Sequence[str]) -> Sample:
    target, clip_id, repetition, warmup, measure, unit, value = row
    if unit != unit_of(measure):
        raise ValueError(f"unit {unit!r} does not match measure {measure!r}")
    return Sample(
        target, clip_id, int(repetition), _parse_flag(warmup), measure, unit, float(value)
    )


def _parsed[U, T](rows: Sequence[U], parse: Callable[[U], T], path: Path) -> list[T]:
    """Parse every row; only a malformed last row (a write cut by a kill) is dropped."""
    parsed: list[T] = []
    for index, row in enumerate(rows):
        try:
            parsed.append(parse(row))
        except (ValueError, TypeError, TimingError, KeyError) as exc:
            if index != len(rows) - 1:
                raise TimingError(f"{path}: malformed row {index + 1}: {exc}") from exc
            _log.warning("%s: dropping a truncated last row: %s", path, exc)
    return parsed


def load_samples(directory: Path) -> list[Sample]:
    path = directory / SAMPLES
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TimingError(f"{path} does not exist: not a timing run directory") from exc
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or tuple(rows[0]) != SAMPLE_COLUMNS:
        raise TimingError(f"{path}: expected the header {','.join(SAMPLE_COLUMNS)}")
    return _parsed(rows[1:], _sample, path)


def _failure(line: str) -> CallFailure:
    record = json.loads(line)
    return CallFailure(
        str(record["target"]),
        str(record["clip_id"]),
        int(record["repetition"]),
        str(record["error"]),
    )


def load_failures(directory: Path) -> list[CallFailure]:
    path = directory / FAILURES
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return _parsed(lines, _failure, path)


def _cell(value: object) -> str:
    if isinstance(value, bool):
        return _flag(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_stats(directory: Path, rows: Iterable[MeasureStats]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(STAT_COLUMNS)
    for row in rows:
        values = row.model_dump()
        writer.writerow([_cell(values[column]) for column in STAT_COLUMNS])
    write_bytes_atomic(directory / STATS, buffer.getvalue().encode("utf-8"))


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
            _flag(target.requires_gpu),
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
