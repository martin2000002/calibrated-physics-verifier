"""The test interpreter as a child process that imports the ``physverify`` under test.

The child runs from that package's ``src`` with PYTHONPATH set to it, never inherited: an inherited
PYTHONPATH can reach the child rewritten (the H200's APM injector does so), and the child then
imports whichever ``physverify`` the interpreter's site-packages point at."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import physverify

SRC = Path(physverify.__file__).resolve().parents[1]


def run_python(*args: str, env: Mapping[str, str] | None = None, timeout: float = 120) -> str:
    """stdout of ``python args`` with ``env`` over this process's environment; a non-zero exit
    fails the call with the child's stderr."""
    done = subprocess.run(
        [sys.executable, *args],
        cwd=SRC,
        env={**os.environ, **(env or {}), "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout
