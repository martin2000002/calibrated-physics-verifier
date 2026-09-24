"""``physverify dataset``: the light subcommands end to end, the heavy ones up to Docker."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from physverify.cli.commands.dataset import image_present


def data_env(tmp_path: Path) -> dict[str, str]:
    return {"PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")}


def test_generate_then_list(cli: Any, tmp_path: Path) -> None:
    env = data_env(tmp_path)
    result = cli("dataset", "generate", "--subset", "smoke", env=env)
    assert result.code == 0, result.err
    assert "2 clips, 1 simulations; 1 specs written, 0 skipped" in result.out
    again = cli("dataset", "generate", "--subset", "smoke", env=env)
    assert "0 specs written" in again.out
    listed = cli("dataset", "list", "--subset", "smoke", env=env)
    assert listed.code == 0, listed.err
    assert listed.out.splitlines()[0].split() == [
        "generate",
        "1",
        "done",
        "0",
        "failed",
        "0",
        "pending",
    ]
    assert any(line.startswith("render") for line in listed.out.splitlines())


def test_list_needs_a_plan(cli: Any, tmp_path: Path) -> None:
    result = cli("dataset", "list", "--subset", "smoke", env=data_env(tmp_path))
    assert result.code == 1
    assert "dataset generate" in result.err


def test_unplanned_subset_is_a_usage_error(cli: Any, tmp_path: Path) -> None:
    env = data_env(tmp_path)
    assert cli("dataset", "generate", "--subset", "smoke", env=env).code == 0
    result = cli("dataset", "list", "--subset", "local", env=env)
    assert result.code == 2
    assert "local" in result.err


def test_missing_image_is_detected() -> None:
    def run(args: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        assert args[:3] == ["docker", "image", "inspect"]
        return subprocess.CompletedProcess(args, 1)

    assert not image_present("kubricdockerhub/kubruntu:latest", run)
