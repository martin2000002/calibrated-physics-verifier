"""Host/container boundary: the Kubric image (Python 3.9, numpy 1.21) reads and writes the
records produced on the host. Run with ``pytest -m docker`` under the heavy lock."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from ping_job import assert_round_tripped, write_ping_job

from physverify.config.loader import load_settings
from physverify.runtime.docker import DockerRunner

pytestmark = pytest.mark.docker


def _image(config_dir: Path, environ: dict[str, str]) -> str:
    return load_settings(config_dir, environ=environ).runtime.docker_image


def test_ping_round_trips_every_record_in_the_container(
    tmp_path: Path, repo_root: Path, config_dir: Path, environ: dict[str, str]
) -> None:
    docker = shutil.which("docker")
    image = _image(config_dir, environ)
    if docker is None:
        pytest.skip("docker is not installed")
    inspect = subprocess.run([docker, "image", "inspect", image], capture_output=True, check=False)
    if inspect.returncode != 0:
        pytest.skip(f"image {image} is not pulled")
    data_dir = tmp_path / "data"
    runner = DockerRunner(
        image=image,
        src_dir=repo_root / "src",
        data_dir=data_dir,
        threads=1,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    job = write_ping_job(data_dir, runner.container_path)
    result = runner.run("ping", job.jobs_file)
    assert result.returncode == 0
    report = assert_round_tripped(job)
    assert str(report["python"]).startswith("3.9")
