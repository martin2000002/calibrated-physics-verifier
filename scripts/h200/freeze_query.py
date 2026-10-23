"""Queries and records of the freeze (run_freeze.sh) and of the sealed final looks (run_final.sh,
run_human_agreement.sh PHYSVERIFY_AGREEMENT_FREEZE).

Exit codes: 0 ok, 2 a check failed or an input is unusable (message on stderr).

    python freeze_query.py select ITERATION_DIR...       the selection rule over their devscore.json
    python freeze_query.py chosen SELECTION_JSON KEY      one value of the chosen iteration
    python freeze_query.py seed COMMIT CALIBRATION_SHA EXCLUSIONS_SHA
    python freeze_query.py diff j3|j13 FROM               files changed from FROM to HEAD
    python freeze_query.py manifest --selection FILE --record-sha SHA --p2-commit C
                           --v1-commit C --v1-calibration PATH --out FREEZE_JSON
    python freeze_query.py field FREEZE_JSON KEY          one value (dotted key)
    python freeze_query.py artifacts FREEZE_JSON          the frozen files are unchanged
    python freeze_query.py render-env SET SEED            the dataset overrides of a fresh render
    python freeze_query.py plan SET SEED                  the render's plan.json is that render's
    python freeze_query.py look-state [--retry] SPEC...   the first look, or the one retry
    python freeze_query.py v1-state FREEZE_JSON SPEC...   the v1 arm follows the look, once
    python freeze_query.py ledger                         the sealed ledger and its line count

The checkout's settings (configs/, the environment, profile precise, device cpu) give the paths,
so every query sees what the physverify calls of the same script see.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from physverify.freeze.diffs import DIFF_PATHS, changed, run_git
from physverify.freeze.errors import QueryError
from physverify.freeze.files import field, read_json, readable_sha256
from physverify.freeze.selection import (
    DEVICE,
    PROFILE,
    candidate,
    chosen_candidate,
    seed_of,
    select,
)

if TYPE_CHECKING:
    from physverify.config.models import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
_git = functools.partial(run_git, REPO_ROOT)


def _settings(extra: dict[str, str] | None = None) -> Settings:
    from physverify.config.loader import load_settings

    environ = {**os.environ, **(extra or {})}
    return load_settings(REPO_ROOT / "configs", environ=environ, profile=PROFILE, device=DEVICE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select").add_argument("dirs", type=Path, nargs="+")
    chosen = sub.add_parser("chosen")
    chosen.add_argument("selection", type=Path)
    chosen.add_argument("key")
    seed = sub.add_parser("seed")
    for name in ("commit", "calibration_sha", "exclusions_sha"):
        seed.add_argument(name)
    diff = sub.add_parser("diff")
    diff.add_argument("name", choices=sorted(DIFF_PATHS))
    diff.add_argument("start")
    made = sub.add_parser("manifest")
    for name in ("--selection", "--record-sha", "--p2-commit", "--v1-commit"):
        made.add_argument(name, required=True, type=Path if name == "--selection" else str)
    for name in ("--v1-calibration", "--out"):
        made.add_argument(name, type=Path, required=True)
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
    sub.add_parser("ledger")
    return parser


def _write_manifest(args: argparse.Namespace) -> list[str]:
    from physverify.app import plugins
    from physverify.freeze.manifest import ManifestInputs, manifest, record_sha

    inputs = ManifestInputs(
        selection=read_json(args.selection),
        record_sha=args.record_sha,
        p2_commit=args.p2_commit,
        v1_commit=args.v1_commit,
        v1_calibration=args.v1_calibration,
    )
    settings = _settings()
    record = functools.partial(record_sha, settings, checks=lambda: plugins.build("checks"))
    data = manifest(inputs, settings, _git, record=record, now=lambda: datetime.now(UTC))
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    j13 = "empty" if data["diffs"]["j13"]["empty"] else f"changes {data['diffs']['j13']['changed']}"
    return [
        f"freeze: {args.out} sha256 {readable_sha256(args.out)}",
        f"commit {data['commit']}, configs tree {data['configs_tree']}, seed {data['seed']}",
        f"j13 diff from the early-extraction commit {args.p2_commit}: {j13}",
    ]


def _render(args: argparse.Namespace) -> list[str]:
    from physverify.datasets.kubric_fresh import fresh_settings
    from physverify.freeze.fresh_render import VERSION_ENV, plan_lines, render_overrides

    overrides = render_overrides(fresh_settings(_settings()), args.dataset, args.seed)
    if args.command == "render-env":
        return [f"{key}={value}" for key, value in overrides.items()]
    return plan_lines(_settings(overrides), overrides[VERSION_ENV], args.seed)


def _artifacts(args: argparse.Namespace) -> list[str]:
    from physverify.freeze.manifest import artifact_problems

    problems = artifact_problems(read_json(args.freeze), _settings())
    if problems:
        raise QueryError("the frozen files changed:\n" + "\n".join(problems))
    return [f"frozen files unchanged ({args.freeze})"]


def _ledger(args: argparse.Namespace) -> list[str]:
    from physverify.freeze.ledger import ledger_entries, ledger_line, look_state, v1_state

    settings, home = _settings(), Path(os.environ.get("HOME") or Path.home())
    if args.command == "ledger":
        return [ledger_line(settings, home)]
    entries = ledger_entries(settings, home)
    if args.command == "look-state":
        return [look_state(entries, args.specs, retry=args.retry)]
    return [v1_state(entries, read_json(args.freeze), args.specs)]


def _output(args: argparse.Namespace) -> list[str]:
    commands: dict[str, Callable[[], list[str]]] = {
        "select": lambda: [
            json.dumps(select([candidate(d) for d in args.dirs]), indent=2, sort_keys=True)
        ],
        "chosen": lambda: [field(chosen_candidate(read_json(args.selection)), args.key)],
        "seed": lambda: [str(seed_of(args.commit, args.calibration_sha, args.exclusions_sha))],
        "diff": lambda: changed(_git, args.name, args.start),
        "manifest": lambda: _write_manifest(args),
        "field": lambda: [field(read_json(args.freeze), args.key)],
        "artifacts": lambda: _artifacts(args),
        "render-env": lambda: _render(args),
        "plan": lambda: _render(args),
    }
    return commands.get(args.command, lambda: _ledger(args))()


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
