"""Every shell script under scripts/h200 parses, and every entry point (no leading underscore,
not the env example) is executable."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

H200 = Path(__file__).resolve().parents[2] / "scripts" / "h200"
SCRIPTS = sorted(p.name for p in H200.glob("*.sh"))
ENTRY_POINTS = [n for n in SCRIPTS if not n.startswith("_") and n != "env.example.sh"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_script_parses(name: str) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(H200 / name)], check=True)


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_the_entry_point_is_executable(name: str) -> None:
    assert os.access(H200 / name, os.X_OK)
