"""scripts/h200/migrate_velocity.sh against a fake ``uv``, ``docker`` and ``git``."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "h200" / "migrate_velocity.sh"
TOOLS = (
    "bash", "cat", "date", "dirname", "env", "grep", "mkdir", "realpath", "rsync", "sed", "tee",
)  # fmt: skip
FAKE_UV = """#!/bin/bash
printf 'uv %s\\n' "$*" >> "$CALLS"
case "$*" in
  *" doctor")
    echo "paths.data_dir: $DATA"
    echo "paths.cache_dir: ${CACHE:-$DATA/cache}"
    echo "paths.runs_dir: $DATA/runs"
    echo "paths.synthetic_dir: $DATA/synthetic"
    echo "heavy_lock: $PHYSVERIFY__RUNTIME__HEAVY_LOCK (free)" ;;
  *" verify "*) mkdir -p "$DATA/runs/r$$"; echo "run: $DATA/runs/r$$" ;;
esac
"""
FAKE_OK = "#!/bin/bash\nexit 0\n"
FAKE_GIT = "#!/bin/bash\necho c0ffee\n"


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.data = self.root / "data"
        self.calls = self.root / "calls.log"
        scripts = self.root / "repo" / "scripts" / "h200"
        scripts.mkdir(parents=True)
        self.script = scripts / SCRIPT.name
        shutil.copy2(SCRIPT, self.script)
        tools, fakes = self.root / "tools", self.root / "fakes"
        tools.mkdir()
        fakes.mkdir()
        for tool in TOOLS:
            found = shutil.which(tool)
            assert found is not None, tool
            (tools / tool).symlink_to(found)
        _executable(fakes / "uv", FAKE_UV)
        _executable(fakes / "docker", FAKE_OK)
        _executable(fakes / "git", FAKE_GIT)
        (self.data / "synthetic" / "v1").mkdir(parents=True)
        (self.data / "synthetic" / "v1" / "plan.json").write_text("{}")
        self.env = {
            "PATH": f"{fakes}:{tools}",
            "CALLS": str(self.calls),
            "DATA": str(self.data),
            "PHYSVERIFY_MIGRATE_ROOT": str(self.root),
            "PHYSVERIFY_MIGRATE_VERSION": "v1",
            "PHYSVERIFY__PATHS__DATA_DIR": str(self.data),
        }

    def run(self, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.root / "tools" / "bash"), str(self.script)],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def commands(self) -> list[str]:
        lines = self.calls.read_text(encoding="utf-8").splitlines() if self.calls.exists() else []
        return [line.split("jobs_per_process=4 ", 1)[-1] for line in lines]


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def test_script_parses_and_is_executable() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(SCRIPT)], check=True)
    assert os.access(SCRIPT, os.X_OK)


def test_a_path_outside_the_root_stops_before_anything_is_written(sandbox: Sandbox) -> None:
    result = sandbox.run(CACHE="/elsewhere/cache")
    assert result.returncode == 2
    assert "outside" in result.stderr
    assert sandbox.commands() == ["doctor"]
    assert not (sandbox.data / "migration-before").exists()


def test_the_steps_run_in_order_on_every_subset_and_cache_root(sandbox: Sandbox) -> None:
    extra = sandbox.root / "p1-cache"
    result = sandbox.run(
        PHYSVERIFY_MIGRATE_SUBSETS="local levels",
        PHYSVERIFY_MIGRATE_VERIFY="local",
        PHYSVERIFY_MIGRATE_CACHE_ROOTS=str(extra),
    )
    assert result.returncode == 0, result.stderr
    manifest = sandbox.data / "synthetic" / "v1" / "migrations" / "velocity-position.json"
    roots = f"--cache-root {sandbox.data / 'cache'} --cache-root {extra}"
    commands = sandbox.commands()
    assert commands[:7] == [
        "doctor",
        "dataset migrate-velocity --commit c0ffee",
        "dataset build --subset local",
        "dataset list --subset local",
        "dataset build --subset levels",
        "dataset list --subset levels",
        f"extract purge --clips-from {manifest} {roots}",
    ]
    sweeps = [c for c in commands if " verify " in f" {c}"]
    assert [("fit_source=position" in c, "fit_source=velocity" in c) for c in sweeps] == [
        (True, False),
        (False, True),
    ]
    assert commands[-1].startswith(
        f"dataset audit-migration --before {sandbox.data / 'migration-before' / 'v1'} "
    )
    assert commands[-1].count("--table ") == 1
    assert (sandbox.data / "migration-before" / "v1" / "plan.json").is_file()
