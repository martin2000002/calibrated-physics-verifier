"""Read-only view of one recorded chain directory of the H200 step runners.

``state.env`` holds ``KEY=%q`` lines appended by ``save`` (a later line wins, as when the script
sources it); ``steps.tsv`` holds ``n, name, exit, wall_s, log`` rows, and a step's last row is
its state, the rule of ``require_completed`` in ``scripts/h200/_steps.sh``.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from physverify.core.errors import UsageError

STATE_FILE = "state.env"
STEPS_FILE = "steps.tsv"
LOGS_DIR = "logs"
CPU = "cpu"
DEVICE_KEY = "CHAIN_DEVICE"
EVALUATION_KEY = "EVAL_JSON"
RECORD_KEY = "RECORD_SHA"
ABSTAIN_STEP = 7
ABSTAIN_CURVE = "abstain curve"
RUN_PREFIX = "run: "
COMMAND_PREFIX = "$ "


class ResultsInputError(UsageError):
    """The inputs of the results report are missing, unfinished or do not belong together."""


@dataclass(frozen=True)
class Missing:
    """Why a source could not be used; the section it feeds is reported as missing."""

    reason: str


@dataclass(frozen=True)
class ChainState:
    root: Path
    env: Mapping[str, str]
    last_codes: Mapping[int, str]

    @classmethod
    def load(cls, root: Path) -> ChainState:
        state, steps = root / STATE_FILE, root / STEPS_FILE
        for path in (state, steps):
            if not path.is_file():
                raise ResultsInputError(f"{root} is not an A8 v2 chain directory: no {path.name}")
        return cls(
            root=root,
            env=parse_state(state.read_text(encoding="utf-8")),
            last_codes=last_codes(steps.read_text(encoding="utf-8")),
        )

    @property
    def device(self) -> str:
        device = self.env.get(DEVICE_KEY)
        if not device:
            raise ResultsInputError(f"{self.root / STATE_FILE} records no {DEVICE_KEY}")
        return device

    @property
    def rehearsal(self) -> bool:
        """Only the CPU chain is reportable; a CUDA chain is a rehearsal."""
        return self.device != CPU

    def completed(self, step: int) -> bool:
        return self.last_codes.get(step) == "0"

    def require_steps(self, steps: Sequence[int]) -> None:
        for step in steps:
            if not self.completed(step):
                code = self.last_codes.get(step, "none")
                raise ResultsInputError(
                    f"step {step} of {self.root} has not completed (last exit {code!r})"
                )

    def required(self, key: str) -> str:
        value = self.env.get(key)
        if not value:
            raise ResultsInputError(f"{self.root / STATE_FILE} records no {key}")
        return value


def parse_state(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        words = shlex.split(line)
        if len(words) != 1 or "=" not in words[0]:
            raise ResultsInputError(f"unreadable {STATE_FILE} line: {line!r}")
        key, value = words[0].split("=", 1)
        found[key] = value
    return found


def last_codes(text: str) -> dict[int, str]:
    found: dict[int, str] = {}
    for line in text.splitlines():
        cells = line.split("\t")
        if len(cells) >= 3 and cells[0].isdigit():
            found[int(cells[0])] = cells[2]
    return found


def abstain_curve_run(state: ChainState) -> Path | Missing:
    """The run printed by step 7's ``abstain curve`` call (the historical log name of
    ``selective-risk curve``, ``typer.echo(f"run: {run_dir}")``). Exactly one candidate is
    accepted; nothing is inferred from dates."""
    found: set[str] = set()
    for log in sorted((state.root / LOGS_DIR).glob(f"{ABSTAIN_STEP}-*.log")):
        lines = log.read_text(encoding="utf-8").splitlines()
        if not lines or not _is_curve(lines[0]):
            continue
        found |= {line.removeprefix(RUN_PREFIX) for line in lines if line.startswith(RUN_PREFIX)}
    if len(found) != 1:
        what = "no" if not found else f"{len(found)} ambiguous"
        return Missing(f"{what} `run:` line of `{ABSTAIN_CURVE}` in {state.root / LOGS_DIR}")
    return Path(found.pop())


def _is_curve(first_line: str) -> bool:
    return first_line.startswith(COMMAND_PREFIX) and ABSTAIN_CURVE in first_line
