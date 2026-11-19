"""Inputs and checks for the ``ping`` container task, shared by the host and Docker tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from records import SAMPLES

from physverify.schema.io import Record

SUFFIXES = (".json", ".npz")


@dataclass(frozen=True)
class PingJob:
    jobs_file: Path
    output_dir: Path
    report: Path


def write_ping_job(data_dir: Path, job_path: Callable[[Path], str]) -> PingJob:
    """Write every sample record in both formats under ``data_dir`` plus a one-job jobs file.

    ``job_path`` maps a host path to the path the worker sees (identity on the host,
    ``/data/...`` inside the container)."""
    inputs_dir = data_dir / "ping" / "in"
    inputs_dir.mkdir(parents=True)
    job = PingJob(
        data_dir / "ping" / "jobs.jsonl",
        data_dir / "ping" / "out",
        data_dir / "ping" / "report.json",
    )
    inputs = []
    for name, build in SAMPLES.items():
        for suffix in SUFFIXES:
            path = inputs_dir / f"{name}{suffix}"
            _save(build(), path)
            inputs.append({"type": name, "path": job_path(path)})
    payload = {
        "inputs": inputs,
        "output_dir": job_path(job.output_dir),
        "report": job_path(job.report),
    }
    job.jobs_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return job


def assert_round_tripped(job: PingJob) -> dict[str, object]:
    """Every record written back by the worker equals the original; returns the report."""
    for name, build in SAMPLES.items():
        expected = build()
        cls = type(expected)
        assert cls.load_json(job.output_dir / f"{name}.json") == expected, name
        assert cls.load_npz(job.output_dir / f"{name}.npz") == expected, name
    report: dict[str, object] = json.loads(job.report.read_text(encoding="utf-8"))
    assert set(SAMPLES) <= set(report["record_types"])  # type: ignore[arg-type]
    return report


def _save(record: Record, path: Path) -> None:
    if path.suffix == ".npz":
        record.save_npz(path)
    else:
        record.save_json(path)
