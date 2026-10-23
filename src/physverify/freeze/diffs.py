"""The diffs a freeze records: the files changed under a check's paths, from a commit to HEAD."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from physverify.freeze.errors import QueryError

DIFF_PATHS: Mapping[str, tuple[str, ...]] = {
    "j3": (
        *(f"src/physverify/{d}" for d in ("checks", "physics", "extraction", "verification")),
        "src/physverify/selective_risk/curve.py",
    ),
    "j13": ("src/physverify/extraction", "configs/verifier/extraction.toml"),
}
Git = Callable[[Sequence[str]], str]


def run_git(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise QueryError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def changed(git: Git, name: str, start: str) -> list[str]:
    if not start:
        raise QueryError(f"no commit to diff {name} from")
    output = git(["diff", "--name-only", start, "HEAD", "--", *DIFF_PATHS[name]])
    return [line for line in output.splitlines() if line]


def diff_record(git: Git, name: str, start: str, head: str) -> dict[str, Any]:
    files = changed(git, name, start)
    return {
        "from": start,
        "to": head,
        "paths": list(DIFF_PATHS[name]),
        "changed": files,
        "empty": not files,
    }
