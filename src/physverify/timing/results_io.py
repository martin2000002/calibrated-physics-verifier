"""``samples.csv``, ``failures.jsonl``, ``results.json`` and ``stats.csv`` of a timing run.

``samples.csv`` and ``failures.jsonl`` grow row by row while the run goes, and ``results.json`` is
rewritten after each target, so an interrupted run keeps what it measured.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import IO, Final

from pydantic import ValidationError

from physverify.core.atomic import write_bytes_atomic
from physverify.core.errors import TimingError
from physverify.timing.records import (
    CallFailure,
    MeasureStats,
    ResultsFile,
    Sample,
    TargetOutcome,
    TargetResult,
    unit_of,
)

RESULTS: Final = "results.json"
SAMPLES: Final = "samples.csv"
STATS: Final = "stats.csv"
FAILURES: Final = "failures.jsonl"
SAMPLE_COLUMNS: Final = ("target", "clip_id", "repetition", "warmup", "measure", "unit", "value")
STAT_COLUMNS: Final = tuple(MeasureStats.model_fields)

_log = logging.getLogger(__name__)


def flag(value: bool) -> str:
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
                flag(sample.warmup),
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
        return flag(value)
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
