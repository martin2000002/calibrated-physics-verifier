"""The iteration a freeze adopts, and the seed of its fresh renders."""

from __future__ import annotations

import math
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from physverify.core.digests import text_sha256
from physverify.freeze.errors import QueryError
from physverify.freeze.files import read_json, readable_sha256

PROFILE = "precise"
DEVICE = "cpu"
ITERATION_ID = re.compile(r"^it(\d+)")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
J4_RULE = (
    "CPU iterations only; most n_met, then the highest AUROC_cv at full precision, then the "
    "later iteration (plan-congelado J4)"
)
ITERATION_FILES: Mapping[str, str] = {
    "DEVSCORE_SHA": "devscore.json",
    "CALIBRATION_SHA": "calibration.json",
    "PROFILE_SHA": "calibration/stage_errors.precise.json",
    "STAGE_SIGMA_SHA": "calibration/stage_sigma.json",
}
ITERATION_STEPS = range(1, 6)


@dataclass(frozen=True)
class Candidate:
    id: str
    dir: str
    number: int
    n_met: int
    auroc_cv: float | None
    commit: str
    devscore_sha256: str
    calibration_sha256: str

    def key(self) -> tuple[int, float, int]:
        return (self.n_met, -math.inf if self.auroc_cv is None else self.auroc_cv, self.number)


def state_env(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines written with ``printf %q``; a later line wins, as ``source`` does."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise QueryError(f"cannot read {path}: {exc}") from exc
    found: dict[str, str] = {}
    for line in lines:
        key, sep, raw = line.partition("=")
        if sep:
            words = shlex.split(raw)
            found[key] = words[0] if words else ""
    return found


def completed_steps(path: Path) -> set[int]:
    """The steps whose last row in ``steps.tsv`` has exit 0."""
    last: dict[int, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= 3 and fields[0].isdigit():
                last[int(fields[0])] = fields[2]
    return {step for step, code in last.items() if code == "0"}


def candidate(directory: Path) -> Candidate:
    """One iteration of run_perception_iteration.sh, refused unless CPU, complete and unchanged."""
    state = state_env(directory / "state.env")
    iteration = state.get("CHAIN_ITER_ID", "")
    match = ITERATION_ID.match(iteration)
    if match is None:
        raise QueryError(f"{directory}: state.env has no iteration id (CHAIN_ITER_ID)")
    if state.get("CHAIN_DEVICE") != DEVICE:
        raise QueryError(
            f"{iteration} ran on {state.get('CHAIN_DEVICE')!r}: only CPU iterations are eligible "
            "under the selection rule"
        )
    missing = [s for s in ITERATION_STEPS if s not in completed_steps(directory / "steps.tsv")]
    if missing:
        raise QueryError(f"{iteration}: steps {missing} are not completed in steps.tsv")
    for key, name in ITERATION_FILES.items():
        if readable_sha256(directory / name) != state.get(key):
            raise QueryError(f"{iteration}: {name} is not the file state.env records ({key})")
    devscore = read_json(directory / ITERATION_FILES["DEVSCORE_SHA"])
    selection = devscore.get("selection")
    if not isinstance(selection, dict):
        raise QueryError(f"{iteration}: devscore.json has no selection")
    auroc = selection.get("tiebreak_auroc_cv")
    return Candidate(
        id=iteration,
        dir=str(directory),
        number=int(match.group(1)),
        n_met=int(selection["n_met"]),
        auroc_cv=None if auroc is None else float(auroc),
        commit=str(devscore.get("commit", "")),
        devscore_sha256=state["DEVSCORE_SHA"],
        calibration_sha256=state["CALIBRATION_SHA"],
    )


def select(candidates: Sequence[Candidate]) -> dict[str, Any]:
    if not candidates:
        raise QueryError("no iteration to choose from")
    numbers = [c.number for c in candidates]
    if len(set(numbers)) != len(numbers):
        raise QueryError(f"two directories of one iteration: {[c.id for c in candidates]}")
    chosen = max(candidates, key=Candidate.key)
    return {"rule": J4_RULE, "candidates": [asdict(c) for c in candidates], "chosen": chosen.id}


def seed_of(commit: str, calibration_sha: str, exclusions_sha: str) -> int:
    """Amendment 2 section 2: the lowercase hex strings joined without separator, in UTF-8."""
    if not COMMIT.match(commit):
        raise QueryError(f"the commit must be 40 lowercase hex characters, got {commit!r}")
    for sha in (calibration_sha, exclusions_sha):
        if not SHA256.match(sha):
            raise QueryError(f"a sha256 must be 64 lowercase hex characters, got {sha!r}")
    return int(text_sha256("".join([commit, calibration_sha, exclusions_sha]))[:8], 16)


def chosen_candidate(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    chosen = [c for c in selection.get("candidates", []) if c.get("id") == selection.get("chosen")]
    if len(chosen) != 1:
        raise QueryError("the selection names no single chosen iteration")
    return dict(chosen[0])
