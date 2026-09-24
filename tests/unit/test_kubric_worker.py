from __future__ import annotations

from pathlib import Path

import pytest
from ping_job import assert_round_tripped, write_ping_job

from physverify.kubric_worker.__main__ import EXIT_USAGE, available_tasks, main, read_jobs


def test_ping_round_trips_every_record_on_the_host(tmp_path: Path) -> None:
    job = write_ping_job(tmp_path, str)
    assert main(["ping", "--jobs", str(job.jobs_file), "--threads", "2"]) == 0
    report = assert_round_tripped(job)
    assert "io.py" in report["compiled"]  # type: ignore[operator]


def test_unknown_task_is_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("", encoding="utf-8")
    assert main(["no_such_task", "--jobs", str(jobs)]) == EXIT_USAGE
    assert "ping" in capsys.readouterr().err


def test_tasks_are_discovered_from_the_package() -> None:
    assert "ping" in available_tasks()


def test_jobs_must_be_objects(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text('{"a": 1}\n\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":3:"):
        read_jobs(jobs)
