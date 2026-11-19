"""A shell sandbox child gets the parent's math-library thread limits, one thread for any the
parent leaves unset or empty, and the APM injection off whatever the parent has."""

from __future__ import annotations

from pathlib import Path

import pytest
from shell_sandbox import APM_INJECTION, THREAD_LIMITS, Sandbox, executable

PROBE = """for name in "$@"; do printf '%s=%s\\n' "$name" "${!name-unset}"; done
"""
ONE = dict.fromkeys(THREAD_LIMITS, "1")
DISTINCT = {name: str(n) for n, name in enumerate(THREAD_LIMITS, start=2)}


@pytest.mark.parametrize(
    ("parent", "child"),
    [({}, ONE), (dict.fromkeys(THREAD_LIMITS, ""), ONE), (DISTINCT, DISTINCT)],
    ids=["unset", "empty", "set"],
)
def test_a_child_gets_the_parent_thread_limits_or_one_and_no_apm_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent: dict[str, str],
    child: dict[str, str],
) -> None:
    for name in THREAD_LIMITS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(APM_INJECTION, "true")
    for name, value in parent.items():
        monkeypatch.setenv(name, value)
    sandbox = Sandbox(tmp_path, (), ("bash",))
    executable(sandbox.scripts / "probe.sh", PROBE)
    done = sandbox.call("probe.sh", *THREAD_LIMITS, APM_INJECTION)
    assert done.returncode == 0, done.stderr
    seen = dict(line.split("=", 1) for line in done.stdout.splitlines())
    assert seen == {**child, APM_INJECTION: "false"}
