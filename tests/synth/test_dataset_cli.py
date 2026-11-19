"""``physverify synth``: the light subcommands end to end, the heavy ones up to Docker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from physverify.runtime.docker import image_present


def data_env(tmp_path: Path) -> dict[str, str]:
    return {"PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")}


def test_generate_then_list(cli: Any, tmp_path: Path) -> None:
    env = data_env(tmp_path)
    result = cli("synth", "generate", "--subset", "smoke", env=env)
    assert result.code == 0, result.err
    assert "2 clips, 1 simulations; 1 specs written, 0 skipped" in result.out
    again = cli("synth", "generate", "--subset", "smoke", env=env)
    assert "0 specs written" in again.out
    listed = cli("synth", "list", "--subset", "smoke", env=env)
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
    result = cli("synth", "list", "--subset", "smoke", env=data_env(tmp_path))
    assert result.code == 1
    assert "synth generate" in result.err


def test_unplanned_subset_is_a_usage_error(cli: Any, tmp_path: Path) -> None:
    env = data_env(tmp_path)
    assert cli("synth", "generate", "--subset", "smoke", env=env).code == 0
    result = cli("synth", "list", "--subset", "local", env=env)
    assert result.code == 2
    assert "local" in result.err


def test_missing_image_is_detected() -> None:
    def run(args: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        assert args[:3] == ["docker", "image", "inspect"]
        return subprocess.CompletedProcess(args, 1)

    assert not image_present("kubricdockerhub/kubruntu:latest", run)


def test_a_plan_from_an_older_gt_revision_is_migrated_before_building(
    cli: Any, tmp_path: Path
) -> None:
    env = data_env(tmp_path)
    assert cli("synth", "generate", "--subset", "smoke", env=env).code == 0
    (plan_path,) = (tmp_path / "data").rglob("plan.json")
    stored = json.loads(plan_path.read_text(encoding="utf-8"))
    del stored["gt_revision"]
    plan_path.write_text(json.dumps(stored), encoding="utf-8")
    refused = cli("synth", "generate", "--subset", "smoke", env=env)
    assert refused.code == 2
    assert "synth migrate --name floor-crossing" in refused.err
    assert cli("synth", "list", "--subset", "smoke", env=env).code == 0
    unknown = cli("synth", "migrate", "--name", "bogus", "--commit", "c0ffee", env=env)
    assert unknown.code == 2 and "unknown migration" in unknown.err
    migrated = cli("synth", "migrate", "--name", "floor-crossing", "--commit", "c0ffee", env=env)
    assert migrated.code == 0, migrated.err
    expected = "migrate floor-crossing: 0 applied, 0 renders discarded, 0 status changes;"
    assert expected in migrated.out
    assert cli("synth", "generate", "--subset", "smoke", env=env).code == 0


def test_a_missing_docker_binary_means_no_image() -> None:
    def run(args: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError(args[0])

    assert not image_present("kubricdockerhub/kubruntu:latest", run)
