"""freeze.json: the frozen files, the diffs from the iteration, the decision record, the seed."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from physverify.calibration.sealed_ledger import frozen_files
from physverify.calibration.sigma_recalibration import STAGE_SIGMA_FILE
from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.core.errors import PhysverifyError
from physverify.freeze.errors import QueryError
from physverify.freeze.files import readable_sha256
from physverify.freeze.selection import COMMIT, DEVICE, PROFILE, seed_of
from physverify.selective_risk.settings import ABSTENTION_FILE
from physverify.verification.scope import check_scope

FREEZE_SCHEMA = "physverify/freeze"
FREEZE_VERSION = "1.0.0"
DIFF_PATHS: Mapping[str, tuple[str, ...]] = {
    "j3": tuple(f"src/physverify/{d}" for d in ("checks", "physics", "extraction", "verification")),
    "j13": ("src/physverify/extraction", "configs/extraction.toml"),
}
Git = Callable[[Sequence[str]], str]


@dataclass(frozen=True)
class ManifestInputs:
    selection: Mapping[str, Any]
    record_sha: str
    p2_commit: str
    v1_commit: str
    v1_calibration: Path


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


def frozen_paths(settings: Settings) -> dict[str, Path | None]:
    """What a look judges with (the ledger's freeze) plus the stage sigma and the abstention."""
    calibration_dir = settings.paths.calibration_dir
    found: dict[str, Path | None] = {"exclusions": None, **frozen_files(settings)}
    found["stage_sigma"] = calibration_dir / STAGE_SIGMA_FILE
    found["abstention"] = calibration_dir / ABSTENTION_FILE
    return found


def artifact_record(paths: Mapping[str, Path | None]) -> dict[str, dict[str, str | None]]:
    """``thresholds`` may be absent (null); it then has to stay absent."""
    record: dict[str, dict[str, str | None]] = {}
    for label, path in paths.items():
        present = path is not None and path.is_file()
        if not present and label != "thresholds":
            raise QueryError(f"the frozen {label} file {path} is missing")
        sha = readable_sha256(path) if present and path is not None else None
        record[label] = {"path": None if path is None else str(path), "sha256": sha}
    return record


def record_sha(settings: Settings, *, checks: Mapping[str, type[Check]]) -> str:
    """J12: the decision record's stable hash, the one run_human_agreement.sh's prereg_problem
    compares."""
    try:
        sha = check_scope(settings, checks).record_sha256
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc
    if sha is None:
        raise QueryError("there is no decision record (exclusions decide)")
    return sha


def chosen_candidate(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    chosen = [c for c in selection.get("candidates", []) if c.get("id") == selection.get("chosen")]
    if len(chosen) != 1:
        raise QueryError("the selection names no single chosen iteration")
    return dict(chosen[0])


def manifest(
    inputs: ManifestInputs,
    settings: Settings,
    git: Git,
    *,
    record: Callable[[], str],
    now: Callable[[], datetime],
) -> dict[str, Any]:
    """freeze.json (AC-8, AC-18); refused when J3's diff is not empty or the record differs.

    ``record`` gives the decision record's hash and ``now`` the UTC time of the freeze."""
    head = git(["rev-parse", "HEAD"]).strip()
    chosen = chosen_candidate(inputs.selection)
    j3 = diff_record(git, "j3", chosen["commit"], head)
    if not j3["empty"]:
        raise QueryError(f"J3: {chosen['id']} -> HEAD changes {j3['changed']}; refit, not adopt")
    if not COMMIT.match(inputs.v1_commit):
        raise QueryError(f"the v1 commit must be 40 lowercase hex characters: {inputs.v1_commit}")
    artifacts = artifact_record(frozen_paths(settings))
    if artifacts["calibration"]["sha256"] != chosen["calibration_sha256"]:
        raise QueryError(f"calibration.json is not the full calib fit of {chosen['id']}")
    computed = record()
    if inputs.record_sha != computed:
        raise QueryError(f"exclusions show printed {inputs.record_sha}, the record is {computed}")
    calibration = str(artifacts["calibration"]["sha256"])
    exclusions = str(artifacts["exclusions"]["sha256"])
    return {
        "schema": FREEZE_SCHEMA,
        "version": FREEZE_VERSION,
        "created_utc": now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": head,
        "configs_tree": git(["rev-parse", "HEAD:configs"]).strip(),
        "profile": PROFILE,
        "device": DEVICE,
        "calibration_dir": str(settings.paths.calibration_dir),
        "artifacts": artifacts,
        "record_sha": computed,
        "seed": seed_of(head, calibration, exclusions),
        "selection": dict(inputs.selection),
        "iteration": {"id": chosen["id"], "dir": chosen["dir"], "commit": chosen["commit"]},
        "diffs": {"j3": j3, "j13": diff_record(git, "j13", inputs.p2_commit, head)},
        "v1": {
            "commit": inputs.v1_commit,
            "calibration_sha256": readable_sha256(inputs.v1_calibration),
        },
    }


def field(data: Mapping[str, Any], key: str) -> str:
    value: Any = data
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise QueryError(f"no {key} in the JSON")
        value = value[part]
    if isinstance(value, (dict, list)) or value is None:
        raise QueryError(f"{key} is not a single value")
    return str(value)


def artifact_problems(frozen: Mapping[str, Any], settings: Settings) -> list[str]:
    """Every frozen file still has its sha256 at the path the settings give."""
    recorded = frozen.get("artifacts", {})
    problems = []
    for label, path in frozen_paths(settings).items():
        entry = recorded.get(label)
        if entry is None:
            problems.append(f"the manifest records no {label}")
            continue
        now = readable_sha256(path) if path is not None and path.is_file() else None
        if entry.get("path") != (None if path is None else str(path)):
            problems.append(f"{label} is read from {path}, the manifest froze {entry.get('path')}")
        elif now != entry.get("sha256"):
            problems.append(f"{label} {path} has sha256 {now}, frozen {entry.get('sha256')}")
    return problems
