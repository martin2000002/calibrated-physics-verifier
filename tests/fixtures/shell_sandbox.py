"""A copy of ``scripts/h200`` in a fake repository, run with a PATH of fakes and basic tools."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

H200_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "h200"


def executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class Sandbox:
    """``repo/scripts/h200`` holds ``script_names``; PATH is ``fakes`` then ``tools``.

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
        }

    def call(self, script: str, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.bash), str(self.scripts / script), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def lines(self) -> list[str]:
        return self.calls.read_text(encoding="utf-8").splitlines() if self.calls.exists() else []
