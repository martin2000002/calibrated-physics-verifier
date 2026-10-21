"""Queries and records of the freeze (run_freeze.sh) and of the sealed final looks (run_final.sh,
run_h2.sh PHYSVERIFY_H2_FREEZE): plan-congelado F7 with Juicio J1, J3-J6, J12 and J13.

Exit codes: 0 ok, 2 a check failed or an input is unusable (message on stderr).

    python freeze_query.py select ITERATION_DIR...       J4 over the iterations' devscore.json
    python freeze_query.py seed COMMIT CALIBRATION_SHA EXCLUSIONS_SHA
    python freeze_query.py diff j3|j13 FROM               files changed from FROM to HEAD
    python freeze_query.py manifest --selection FILE --record-sha SHA --p2-commit C
                           --v1-commit C --v1-calibration PATH --out FREEZE_JSON
    python freeze_query.py field FREEZE_JSON KEY          one value (dotted key)
    python freeze_query.py artifacts FREEZE_JSON          the frozen files are unchanged
    python freeze_query.py render-env SET SEED            the dataset overrides of a fresh render
    python freeze_query.py plan SET SEED                  the render's plan.json is that render's
    python freeze_query.py look-state [--retry] SPEC...   J6: first look, or the one retry
    python freeze_query.py v1-state FREEZE_JSON SPEC...   the v1 arm follows the look, once

The checkout's settings (configs/, the environment, profile precise, device cpu) give the paths,
so every query sees what the physverify calls of the same script see.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = "precise"
DEVICE = "cpu"
FREEZE_SCHEMA = "physverify/freeze"
FREEZE_VERSION = "1.0.0"
ITERATION_ID = re.compile(r"^it(\d+)")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
J4_RULE = (
    "CPU iterations only; most n_met, then the highest AUROC_cv at full precision, then the "
    "later iteration (plan-congelado J4)"
)
DIFF_PATHS: Mapping[str, tuple[str, ...]] = {
    "j3": tuple(f"src/physverify/{d}" for d in ("checks", "physics", "extraction", "verification")),
    "j13": ("src/physverify/extraction", "configs/extraction.toml"),
}
ITERATION_FILES: Mapping[str, str] = {
    "DEVSCORE_SHA": "devscore.json",
    "CALIBRATION_SHA": "calibration.json",
    "PROFILE_SHA": "calibration/stage_errors.precise.json",
    "STAGE_SIGMA_SHA": "calibration/stage_sigma.json",
}
ITERATION_STEPS = range(1, 6)
FRESH_SUBSET = "full"
VERSION_ENV = "PHYSVERIFY__DATASET__VERSION"
Git = Callable[[Sequence[str]], str]


class QueryError(Exception):
    """A check failed or an input is unusable (exit 2)."""


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


def sha256_of(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise QueryError(f"cannot read {path}: {exc}") from exc


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
    """One iteration of run_iteration.sh, refused unless CPU, complete and unchanged."""
    state = state_env(directory / "state.env")
    iteration = state.get("CHAIN_ITER_ID", "")
    match = ITERATION_ID.match(iteration)
    if match is None:
        raise QueryError(f"{directory}: state.env has no iteration id (CHAIN_ITER_ID)")
    if state.get("CHAIN_DEVICE") != DEVICE:
        raise QueryError(
            f"{iteration} ran on {state.get('CHAIN_DEVICE')!r}: only CPU iterations are eligible "
            "(J4, amendment 2 section 7)"
        )
    missing = [s for s in ITERATION_STEPS if s not in completed_steps(directory / "steps.tsv")]
    if missing:
        raise QueryError(f"{iteration}: steps {missing} are not completed in steps.tsv")
    for key, name in ITERATION_FILES.items():
        if sha256_of(directory / name) != state.get(key):
            raise QueryError(f"{iteration}: {name} is not the file state.env records ({key})")
    devscore = _json(directory / ITERATION_FILES["DEVSCORE_SHA"])
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
    digest = hashlib.sha256("".join([commit, calibration_sha, exclusions_sha]).encode("utf-8"))
    return int(digest.hexdigest()[:8], 16)


def run_git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
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


def frozen_paths(settings: Any) -> dict[str, Path | None]:
    """What a look judges with (the ledger's freeze) plus the stage sigma and the abstention."""
    from physverify.cli.commands._sealed import frozen_files
    from physverify.cli.commands.calibrate import STAGE_SIGMA_FILE
    from physverify.verification.selective_risk import ABSTENTION_FILE

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
        sha = sha256_of(path) if present and path is not None else None
        record[label] = {"path": None if path is None else str(path), "sha256": sha}
    return record


def record_sha(settings: Any) -> str:
    """J12: the decision record's stable hash, the one run_h2.sh's prereg_problem compares."""
    from physverify.calibration.settings import check_registry
    from physverify.core.errors import PhysverifyError
    from physverify.verification.scope import check_scope

    try:
        sha = check_scope(settings, check_registry()).record_sha256
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc
    if sha is None:
        raise QueryError("there is no decision record (exclusions decide)")
    return sha


@dataclass(frozen=True)
class ManifestInputs:
    selection: Mapping[str, Any]
    record_sha: str
    p2_commit: str
    v1_commit: str
    v1_calibration: Path


def manifest(inputs: ManifestInputs, settings: Any, git: Git) -> dict[str, Any]:
    """freeze.json (AC-8, AC-18); refused when J3's diff is not empty or the record differs."""
    head = git(["rev-parse", "HEAD"]).strip()
    chosen = _chosen(inputs.selection)
    j3 = diff_record(git, "j3", chosen["commit"], head)
    if not j3["empty"]:
        raise QueryError(f"J3: {chosen['id']} -> HEAD changes {j3['changed']}; refit, not adopt")
    if not COMMIT.match(inputs.v1_commit):
        raise QueryError(f"the v1 commit must be 40 lowercase hex characters: {inputs.v1_commit}")
    artifacts = artifact_record(frozen_paths(settings))
    if artifacts["calibration"]["sha256"] != chosen["calibration_sha256"]:
        raise QueryError(f"calibration.json is not the full calib fit of {chosen['id']}")
    computed = record_sha(settings)
    if inputs.record_sha != computed:
        raise QueryError(f"exclusions show printed {inputs.record_sha}, the record is {computed}")
    calibration = str(artifacts["calibration"]["sha256"])
    exclusions = str(artifacts["exclusions"]["sha256"])
    return {
        "schema": FREEZE_SCHEMA,
        "version": FREEZE_VERSION,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
            "calibration_sha256": sha256_of(inputs.v1_calibration),
        },
    }


def _chosen(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    chosen = [c for c in selection.get("candidates", []) if c.get("id") == selection.get("chosen")]
    if len(chosen) != 1:
        raise QueryError("the selection names no single chosen iteration")
    return dict(chosen[0])


def field(data: Mapping[str, Any], key: str) -> str:
    value: Any = data
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise QueryError(f"no {key} in the freeze manifest")
        value = value[part]
    if isinstance(value, (dict, list)) or value is None:
        raise QueryError(f"{key} is not a single value")
    return str(value)


def artifact_problems(frozen: Mapping[str, Any], settings: Any) -> list[str]:
    """Every frozen file still has its sha256 at the path the settings give."""
    recorded = frozen.get("artifacts", {})
    problems = []
    for label, path in frozen_paths(settings).items():
        entry = recorded.get(label)
        if entry is None:
            problems.append(f"the manifest records no {label}")
            continue
        now = sha256_of(path) if path is not None and path.is_file() else None
        if entry.get("path") != (None if path is None else str(path)):
            problems.append(f"{label} is read from {path}, the manifest froze {entry.get('path')}")
        elif now != entry.get("sha256"):
            problems.append(f"{label} {path} has sha256 {now}, frozen {entry.get('sha256')}")
    return problems


def render_overrides(fresh: Any, dataset: str, seed: int) -> dict[str, str]:
    """The dataset section a fresh render reads: plan-congelado section 2.2."""
    if dataset == "kubric_fresh_ood":
        version, scenes = fresh.ood_version, fresh.ood_scenes_per_family
        extra = {"PHYSVERIFY__DATASET__HDRI_POOL": fresh.ood_hdri_pool}
    elif dataset in ("kubric_fresh", "kubric_fresh_blind"):
        version, scenes, extra = fresh.version, fresh.scenes_per_family, {}
    else:
        raise QueryError(f"{dataset} is not a fresh Kubric set")
    return {
        VERSION_ENV: version,
        "PHYSVERIFY__DATASET__SEED": str(seed),
        "PHYSVERIFY__DATASET__CALIB_FRACTION": "0.0",
        f"PHYSVERIFY__DATASET__SUBSETS__{FRESH_SUBSET.upper()}__SCENES_PER_FAMILY": str(scenes),
        **extra,
    }


def plan_lines(settings: Any, version: str, seed: int) -> list[str]:
    """J1: the stored plan is the render of ``seed`` (its content hash), with no calib clip."""
    from physverify.schema.clip import Split
    from physverify.synth.generate import load_plan
    from physverify.synth.settings import dataset_root, load_dataset_settings
    from physverify.synth.store import DatasetStore

    ds = load_dataset_settings(settings)
    if (ds.version, ds.seed) != (version, seed):
        raise QueryError(
            f"the render settings give {ds.version} seed {ds.seed}, not {version} {seed}"
        )
    store = DatasetStore(dataset_root(settings, ds))
    plan = load_plan(store)
    if plan is None:
        raise QueryError(f"no plan at {store.plan_path}: render first")
    if plan.version != version or plan.config_hash != ds.content_hash():
        raise QueryError(f"{store.plan_path} is not the render of {version} seed {seed}")
    calib = [s.scene_id for s in plan.scenes if s.split is Split.CALIB]
    calib += [v.clip_id for v in plan.variants if v.split is Split.CALIB]
    if calib:
        raise QueryError(f"{store.plan_path} plans {len(calib)} calib scenes or clips")
    return [
        f"plan {store.plan_path}: version {version}, seed {seed}, {len(plan.scenes)} scenes, "
        f"{len(plan.variants)} clips, 0 calib"
    ]


def look_state(entries: Sequence[Any], specs: Sequence[str], *, retry: bool) -> str:
    """J6: a set is looked at once; one retry only when that look wrote no evaluation."""
    from physverify.calibration.report import EVALUATION_MD
    from physverify.calibration.sealed import earlier
    from physverify.cli.commands.calibrate import EVALUATION_FILE

    found = earlier(entries, specs)
    if not retry:
        if found:
            raise QueryError(
                f"{','.join(specs)} already has {len(found)} ledger lines: " + _lines(found)
            )
        return "first"
    if len(found) != 1:
        raise QueryError(f"--retry needs exactly one earlier look, found {len(found)}")
    run_dir = found[0].run_dir
    if run_dir is None:
        raise QueryError("the earlier look recorded no run directory")
    written = [name for name in (EVALUATION_FILE, EVALUATION_MD) if (Path(run_dir) / name).exists()]
    if written:
        raise QueryError(f"the earlier look wrote {written} in {run_dir}: no retry")
    return "retry"


def v1_state(entries: Sequence[Any], frozen: Mapping[str, Any], specs: Sequence[str]) -> str:
    """The v1 arm repeats a look of the freeze, once."""
    from physverify.calibration.sealed import earlier

    found = earlier(entries, specs)
    looks = [e for e in found if e.commit == frozen["commit"]]
    arms = [e for e in found if e.commit == frozen["v1"]["commit"]]
    if not looks:
        raise QueryError(f"{','.join(specs)} has no look of the freeze yet: run look first")
    if arms:
        raise QueryError(f"the v1 arm already looked at {','.join(specs)}: " + _lines(arms))
    return "v1arm"


def _lines(entries: Sequence[Any]) -> str:
    return "; ".join(f"{e.utc} {e.commit[:12]} {e.command} run {e.run_dir}" for e in entries)


def ledger_entries(settings: Any) -> tuple[Any, ...]:
    """The ledger the guard of every look appends to (fails closed when it is missing)."""
    from physverify.calibration.sealed import ledger_path, read_entries, sealed_settings
    from physverify.core.errors import PhysverifyError

    home = Path(os.environ.get("HOME") or Path.home())
    try:
        sealed = sealed_settings(settings)
        return read_entries(ledger_path(sealed, home, settings.repo_root))
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QueryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise QueryError(f"{path} is not a JSON object")
    return data


def _settings(extra: Mapping[str, str] | None = None) -> Any:
    from physverify.config.loader import load_settings

    environ = {**os.environ, **(extra or {})}
    return load_settings(REPO_ROOT / "configs", environ=environ, profile=PROFILE, device=DEVICE)


def _fresh(settings: Any) -> Any:
    from physverify.datasets.kubric_fresh import fresh_settings

    return fresh_settings(settings)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select").add_argument("dirs", type=Path, nargs="+")
    seed = sub.add_parser("seed")
    for name in ("commit", "calibration_sha", "exclusions_sha"):
        seed.add_argument(name)
    diff = sub.add_parser("diff")
    diff.add_argument("name", choices=sorted(DIFF_PATHS))
    diff.add_argument("start")
    made = sub.add_parser("manifest")
    made.add_argument("--selection", type=Path, required=True)
    for name in ("--record-sha", "--p2-commit", "--v1-commit"):
        made.add_argument(name, required=True)
    made.add_argument("--v1-calibration", type=Path, required=True)
    made.add_argument("--out", type=Path, required=True)
    one = sub.add_parser("field")
    one.add_argument("freeze", type=Path)
    one.add_argument("key")
    sub.add_parser("artifacts").add_argument("freeze", type=Path)
    for name in ("render-env", "plan"):
        render = sub.add_parser(name)
        render.add_argument("dataset")
        render.add_argument("seed", type=int)
    state = sub.add_parser("look-state")
    state.add_argument("--retry", action="store_true")
    state.add_argument("specs", nargs="+")
    arm = sub.add_parser("v1-state")
    arm.add_argument("freeze", type=Path)
    arm.add_argument("specs", nargs="+")
    return parser


def _write_manifest(args: argparse.Namespace) -> list[str]:
    inputs = ManifestInputs(
        selection=_json(args.selection),
        record_sha=args.record_sha,
        p2_commit=args.p2_commit,
        v1_commit=args.v1_commit,
        v1_calibration=args.v1_calibration,
    )
    data = manifest(inputs, _settings(), run_git)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    j13 = "empty" if data["diffs"]["j13"]["empty"] else f"changes {data['diffs']['j13']['changed']}"
    return [
        f"freeze: {args.out} sha256 {sha256_of(args.out)}",
        f"commit {data['commit']}, configs tree {data['configs_tree']}, seed {data['seed']}",
        f"J13 diff from P2 {args.p2_commit}: {j13}",
    ]


def _render_env(args: argparse.Namespace) -> list[str]:
    overrides = render_overrides(_fresh(_settings()), args.dataset, args.seed)
    return [f"{key}={value}" for key, value in overrides.items()]


def _plan(args: argparse.Namespace) -> list[str]:
    overrides = render_overrides(_fresh(_settings()), args.dataset, args.seed)
    version = overrides[VERSION_ENV]
    return plan_lines(_settings(overrides), version, args.seed)


def _artifacts(args: argparse.Namespace) -> list[str]:
    problems = artifact_problems(_json(args.freeze), _settings())
    if problems:
        raise QueryError("the frozen files changed:\n" + "\n".join(problems))
    return [f"frozen files unchanged ({args.freeze})"]


def _output(args: argparse.Namespace) -> list[str]:
    commands: dict[str, Callable[[], list[str]]] = {
        "select": lambda: [
            json.dumps(select([candidate(d) for d in args.dirs]), indent=2, sort_keys=True)
        ],
        "seed": lambda: [str(seed_of(args.commit, args.calibration_sha, args.exclusions_sha))],
        "diff": lambda: changed(run_git, args.name, args.start),
        "manifest": lambda: _write_manifest(args),
        "field": lambda: [field(_json(args.freeze), args.key)],
        "artifacts": lambda: _artifacts(args),
        "render-env": lambda: _render_env(args),
        "plan": lambda: _plan(args),
        "look-state": lambda: [
            look_state(ledger_entries(_settings()), args.specs, retry=args.retry)
        ],
        "v1-state": lambda: [v1_state(ledger_entries(_settings()), _json(args.freeze), args.specs)],
    }
    return commands[args.command]()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        lines = _output(args)
    except QueryError as exc:
        print(f"freeze_query: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
