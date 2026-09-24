from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest

from physverify.core.errors import ConfigError, UsageError
from physverify.runtime.docker import CONTAINER_SRC, WORKER_MODULE, DockerRunner
from physverify.runtime.runs import RunContext, current_commit, new_run_id
from physverify.runtime.threads import THREAD_VARIABLES, thread_env


def _runner(tmp_path: Path, image: str = "image@sha256:abc") -> DockerRunner:
    return DockerRunner(
        image=image,
        src_dir=tmp_path / "src",
        data_dir=tmp_path / "data",
        threads=3,
        uid=1000,
        gid=1000,
    )


def test_docker_argv_mounts_source_read_only_and_maps_jobs(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    argv = runner.argv("ping", tmp_path / "data" / "ping" / "jobs.jsonl")
    assert argv[:3] == ["docker", "run", "--rm"]
    assert f"{(tmp_path / 'src').resolve()}:{CONTAINER_SRC}:ro" in argv
    assert argv[argv.index("--user") + 1] == "1000:1000"
    assert argv[-8:] == [
        "python3",
        "-m",
        WORKER_MODULE,
        "ping",
        "--jobs",
        "/data/ping/jobs.jsonl",
        "--threads",
        "3",
    ]
    assert "OMP_NUM_THREADS=3" in argv


def test_paths_outside_the_data_dir_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="outside the data directory"):
        _runner(tmp_path).container_path(tmp_path / "elsewhere" / "jobs.jsonl")


def test_empty_image_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _runner(tmp_path, image="").argv("ping", tmp_path / "data" / "jobs.jsonl")


def test_run_uses_the_injected_runner(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake(argv: list[str], check: bool) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    result = _runner(tmp_path).run("ping", tmp_path / "data" / "jobs.jsonl", runner=fake)
    assert result.returncode == 0 and calls[0][0] == "docker"


def test_thread_env_caps_every_pool() -> None:
    assert thread_env(2) == {name: "2" for name in THREAD_VARIABLES}
    with pytest.raises(ValueError):
        thread_env(0)


def test_run_context_records_how_a_result_was_produced(tmp_path: Path) -> None:
    now = dt.datetime(2026, 9, 24, 13, 5, 7, tzinfo=dt.UTC)
    assert new_run_id(now, "abc123") == "20260924-130507-abc123"
    context = RunContext.create(
        tmp_path / "runs",
        argv=["verify", "x"],
        device="cpu",
        profile="default",
        config_hash="h",
        commit="c",
        now=now,
    )
    record = json.loads((context.run_dir / "run.json").read_text(encoding="utf-8"))
    assert (
        record["argv"] == ["verify", "x"] and record["created_utc"] == "2026-09-24T13:05:07+00:00"
    )
    assert context.run_dir.parent == tmp_path / "runs"


def test_current_commit_is_empty_outside_git(tmp_path: Path) -> None:
    assert current_commit(tmp_path) == ""
