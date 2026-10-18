"""Container engine selection and the udocker argv, without udocker or Docker."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.core.errors import ConfigError, UsageError
from physverify.runtime.container import describe, make_runner, missing_reason
from physverify.runtime.docker import CONTAINER_SRC, DockerRunner
from physverify.runtime.udocker import UdockerRunner

UDOCKER = "/opt/udocker/bin/udocker"


def _udocker(tmp_path: Path, container: str = "kub") -> UdockerRunner:
    return UdockerRunner(
        bin=UDOCKER,
        container=container,
        src_dir=tmp_path / "src",
        data_dir=tmp_path / "data",
        threads=3,
    )


def _docker(tmp_path: Path) -> DockerRunner:
    return DockerRunner(
        image="image@sha256:abc",
        src_dir=tmp_path / "src",
        data_dir=tmp_path / "data",
        threads=3,
        uid=1000,
        gid=1000,
    )


def _jobs(tmp_path: Path) -> Path:
    return tmp_path / "data" / "render" / "jobs.jsonl"


def _settings(config_dir: Path, environ: dict[str, str], local: str = "") -> Settings:
    if local:
        with (config_dir / "local.toml").open("a", encoding="utf-8") as handle:
            handle.write(local)
    return load_settings(config_dir, environ=environ)


def test_udocker_argv(tmp_path: Path) -> None:
    argv = _udocker(tmp_path).argv("render", _jobs(tmp_path))
    assert argv[:3] == [UDOCKER, "run", "--nobanner"]
    assert f"--volume={(tmp_path / 'src').resolve()}:{CONTAINER_SRC}" in argv
    assert f"--volume={(tmp_path / 'data').resolve()}:/data" in argv
    assert "--workdir=/data" in argv
    for env in ("PYTHONPATH=/app/src", "HOME=/tmp", "OMP_NUM_THREADS=3"):
        assert f"--env={env}" in argv
    assert "--env=PYTHONDONTWRITEBYTECODE=1" in argv
    assert not any(a.endswith(":ro") or a in ("--rm", "--user") for a in argv)
    assert argv[argv.index("python3") - 1] == "kub"


def test_both_engines_share_the_worker_command(tmp_path: Path) -> None:
    udocker = _udocker(tmp_path).argv("render", _jobs(tmp_path))
    docker = _docker(tmp_path).argv("render", _jobs(tmp_path))
    assert udocker[-8:] == docker[-8:]
    assert udocker[-8:][-4:] == ["--jobs", "/data/render/jobs.jsonl", "--threads", "3"]


@pytest.mark.parametrize("build", [_udocker, _docker])
def test_container_path_is_the_same_for_both_engines(tmp_path: Path, build: Any) -> None:
    runner = build(tmp_path)
    assert runner.container_path(tmp_path / "data" / "a" / "b.npz") == "/data/a/b.npz"
    with pytest.raises(UsageError, match="outside the data directory"):
        runner.container_path(tmp_path / "elsewhere.npz")


def test_empty_udocker_container_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="udocker_container"):
        _udocker(tmp_path, container="")


def test_udocker_run_uses_the_injected_runner(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake(argv: list[str], check: bool) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    runner = _udocker(tmp_path)
    assert runner.run("ping", _jobs(tmp_path), runner=fake).returncode == 0
    assert calls == [runner.argv("ping", _jobs(tmp_path))]


def test_docker_is_the_default_engine(config_dir: Path, environ: dict[str, str]) -> None:
    settings = _settings(config_dir, environ)
    runner = make_runner(settings, 1000, 1000)
    assert isinstance(runner, DockerRunner)
    assert runner.image == settings.runtime.docker_image
    assert describe(settings) == "docker"


def test_udocker_from_the_config_file(config_dir: Path, environ: dict[str, str]) -> None:
    settings = _settings(
        config_dir, environ, f'\n[container]\nengine = "udocker"\nudocker_bin = "{UDOCKER}"\n'
    )
    runner = make_runner(settings, 1000, 1000)
    assert isinstance(runner, UdockerRunner)
    assert (runner.bin, runner.container) == (UDOCKER, "kub")
    assert runner.data_dir == settings.paths.data_dir
    assert runner.threads == settings.runtime.threads
    assert describe(settings) == f"udocker (kub via {UDOCKER})"


def test_udocker_from_the_environment(config_dir: Path, environ: dict[str, str]) -> None:
    env = {
        **environ,
        "PHYSVERIFY__CONTAINER__ENGINE": "udocker",
        "PHYSVERIFY__CONTAINER__UDOCKER_CONTAINER": "kub2",
    }
    runner = make_runner(_settings(config_dir, env), 1000, 1000)
    assert isinstance(runner, UdockerRunner)
    assert runner.container == "kub2"


def test_unknown_engine_is_a_config_error(config_dir: Path, environ: dict[str, str]) -> None:
    settings = _settings(config_dir, {**environ, "PHYSVERIFY__CONTAINER__ENGINE": "podman"})
    with pytest.raises(ConfigError, match=r"\[container\]") as caught:
        make_runner(settings, 1000, 1000)
    assert caught.value.exit_code == 2


def test_probe_argv_per_engine(tmp_path: Path) -> None:
    assert _docker(tmp_path).probe_argv() == ["docker", "image", "inspect", "image@sha256:abc"]
    assert _udocker(tmp_path).probe_argv() == [UDOCKER, "inspect", "kub"]


def _probe(code: int | None) -> Any:
    def run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        if code is None:
            raise FileNotFoundError(argv[0])
        return subprocess.CompletedProcess(argv, code)

    return run


def test_missing_reason(config_dir: Path, environ: dict[str, str]) -> None:
    docker_settings = _settings(config_dir, environ)
    docker = make_runner(docker_settings, 1000, 1000)
    assert missing_reason(docker_settings, docker, _probe(0)) is None
    reason = missing_reason(docker_settings, docker, _probe(1))
    assert reason is not None and "docker image inspect" in reason and "pull-image" in reason

    env = {**environ, "PHYSVERIFY__CONTAINER__ENGINE": "udocker"}
    udocker_settings = load_settings(config_dir, environ=env)
    udocker = make_runner(udocker_settings, 1000, 1000)
    reason = missing_reason(udocker_settings, udocker, _probe(1))
    assert reason is not None and "inspect kub" in reason and "run_kubric.sh" in reason
    missing = missing_reason(udocker_settings, udocker, _probe(None))
    assert missing is not None and "cannot run" in missing


def test_doctor_reports_the_engine(cli: Any) -> None:
    assert "container_engine: docker" in cli("doctor").out
    udocker = cli("doctor", env={"PHYSVERIFY__CONTAINER__ENGINE": "udocker"}).out
    assert "container_engine: udocker (kub via udocker)" in udocker


def test_pull_image_refuses_other_engines(cli: Any) -> None:
    result = cli("dataset", "pull-image", env={"PHYSVERIFY__CONTAINER__ENGINE": "udocker"})
    assert result.code == 2
    assert "run_kubric.sh" in result.err


def test_dataset_steps_stop_when_the_container_is_missing(cli: Any, tmp_path: Path) -> None:
    env = {
        "PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data"),
        "PHYSVERIFY__CONTAINER__ENGINE": "udocker",
        "PHYSVERIFY__CONTAINER__UDOCKER_BIN": str(tmp_path / "no-such-udocker"),
    }
    assert cli("dataset", "generate", "--subset", "smoke", env=env).code == 0
    result = cli("dataset", "simulate", "--subset", "smoke", env=env)
    assert result.code == 1
    assert "Kubric container is not available" in result.err
