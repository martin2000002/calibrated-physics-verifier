"""One parse function of scripts/h200/_steps.sh run on a captured command output, as the H200
runners call it."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

STEPS_SH = Path(__file__).resolve().parents[2] / "scripts" / "h200" / "_steps.sh"
_SCRIPT = """set -euo pipefail
source "$1"
stop() { printf 'stop: %s\\n' "$*" >&2; exit 2; }
LAST_LOG="$2"
shift 2
"$@"
"""


def steps_read(log: Path, function: str, *args: str, env: Mapping[str, str] | None = None) -> str:
    """stdout of ``function args`` with ``LAST_LOG=log`` and ``env``; a ``stop`` fails the call."""
    result = subprocess.run(
        ["bash", "-c", _SCRIPT, "_", str(STEPS_SH), str(log), function, *args],
        env={"PATH": os.environ["PATH"], **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.removesuffix("\n")
