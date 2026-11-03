"""scripts/h200/checkout_at.sh with the real ``git`` on a repository made here."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from shell_sandbox import Sandbox

TOOLS = ("bash", "dirname", "git", "mkdir", "nproc", "realpath")
IDENTITY = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


class CheckoutSandbox(Sandbox):
    """A source repository with two commits; ``first`` is the older, not the tip."""

    def __init__(self, root: Path) -> None:
        super().__init__(root.resolve(), ("_common.sh", "checkout_at.sh"), TOOLS)
        self.env.update(IDENTITY)
        self.source = self.root / "source"
        self.source.mkdir()
        self.git("init", "-q")
        for text in ("one", "two"):
            (self.source / "file.txt").write_text(text, encoding="utf-8")
            self.git("add", "file.txt")
            self.git("commit", "-q", "-m", text)
        self.first = self.git("rev-parse", "HEAD~1")

    def git(self, *args: str, cwd: Path | None = None) -> str:
        done = subprocess.run(
            ["git", *args], cwd=cwd or self.source, env=self.env, check=True,
            capture_output=True, text=True,
        )  # fmt: skip
        return done.stdout.strip()

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self.call("checkout_at.sh", *args)


@pytest.fixture
def sandbox(tmp_path: Path) -> CheckoutSandbox:
    return CheckoutSandbox(tmp_path)


def test_a_short_commit_is_checked_out_at_its_full_sha(sandbox: CheckoutSandbox) -> None:
    dest = sandbox.root / "dest"
    result = sandbox.run(str(sandbox.source), sandbox.first[:7], str(dest))
    assert result.returncode == 0, result.stderr
    assert sandbox.git("rev-parse", "HEAD", cwd=dest) == sandbox.first
    assert sandbox.git("status", "--porcelain", cwd=dest) == ""
    assert (dest / "file.txt").read_text(encoding="utf-8") == "one"
    assert sandbox.git("rev-parse", "HEAD") != sandbox.first


@pytest.mark.parametrize(
    ("commit", "message"),
    [("deadbee", "no single commit deadbee"), ("HEAD", "7 to 40 hex"), ("abc", "7 to 40 hex")],
)
def test_a_commit_that_is_not_one_stops_before_any_write(
    sandbox: CheckoutSandbox, commit: str, message: str
) -> None:
    dest = sandbox.root / "dest"
    result = sandbox.run(str(sandbox.source), commit, str(dest))
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not dest.exists()


def test_an_existing_destination_is_refused(sandbox: CheckoutSandbox) -> None:
    dest = sandbox.root / "dest"
    dest.mkdir()
    result = sandbox.run(str(sandbox.source), sandbox.first, str(dest))
    assert result.returncode == 2 and "exists" in result.stderr
    assert list(dest.iterdir()) == []
