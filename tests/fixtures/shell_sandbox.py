"""A copy of ``scripts/h200`` in a fake repository, run with a PATH of fakes and basic tools."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[2]
H200_SCRIPTS = CHECKOUT / "scripts" / "h200"
CHECKOUT_SRC = CHECKOUT / "src"
THREAD_LIMITS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
APM_INJECTION = "DD_INSTRUMENT_SERVICE_WITH_APM"
# A guard against a hung child, not a speed check: a loaded H200 stretches one call to minutes.
CALL_TIMEOUT_S = 600


def executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def thread_limits(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ``'s thread limit for each math library, 1 where it is unset or empty: a library
    left unlimited sizes its pool to every core (224 on the H200) in each child."""
    return {name: environ.get(name) or "1" for name in THREAD_LIMITS}


class Sandbox:
    """``repo/scripts/h200`` holds ``script_names``; PATH is ``fakes`` then ``tools``.

    ``repo/src`` links to this checkout's ``src``, which also leads PYTHONPATH, so the scripts
    import ``physverify`` from this checkout and not from an editable install elsewhere. Of the
    parent's environment only its ``thread_limits`` reach the scripts, and the host's Datadog APM
    injection is off: it adds five threads and about 2.5 s to the start of every Python child.

    ``tools`` links each named tool to the host binary; ``fakes`` starts empty and shadows them.
    Fakes log to ``calls``, which ``lines`` reads back.
    """

    def __init__(self, root: Path, script_names: Iterable[str], tools: Iterable[str]) -> None:
        self.root = root
        self.repo = root / "repo"
        self.calls = root / "calls.log"
        self.scripts = self.repo / "scripts" / "h200"
        self.scripts.mkdir(parents=True)
        for name in script_names:
            shutil.copy2(H200_SCRIPTS / name, self.scripts / name)
        (self.repo / "src").symlink_to(CHECKOUT_SRC, target_is_directory=True)
        self.tools = root / "tools"
        self.tools.mkdir()
        for tool in tools:
            found = shutil.which(tool)
            assert found is not None, tool
            (self.tools / tool).symlink_to(found)
        self.fakes = root / "fakes"
        self.fakes.mkdir()
        self.bash = self.tools / "bash"
        self.env = {
            "PATH": f"{self.fakes}:{self.tools}",
            "HOME": str(root),
            "CALLS": str(self.calls),
            "PYTHONPATH": str(CHECKOUT_SRC),
            APM_INJECTION: "false",
            **thread_limits(os.environ),
        }

    def call(self, script: str, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.bash), str(self.scripts / script), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=CALL_TIMEOUT_S,
            check=False,
        )

    def lines(self) -> list[str]:
        return self.calls.read_text(encoding="utf-8").splitlines() if self.calls.exists() else []
