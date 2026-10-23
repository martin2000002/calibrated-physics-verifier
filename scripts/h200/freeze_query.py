"""Queries and records of the freeze (run_freeze.sh) and of the sealed final looks (run_final.sh,
run_human_agreement.sh PHYSVERIFY_AGREEMENT_FREEZE): plan-congelado F7 with Juicio J1, J3-J6, J12
and J13.

Exit codes: 0 ok, 2 a check failed or an input is unusable (message on stderr).

    python freeze_query.py select ITERATION_DIR...       J4 over the iterations' devscore.json
    python freeze_query.py chosen SELECTION_JSON KEY      one value of the chosen iteration
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

from physverify.app import plugins
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.datasets.kubric_fresh import fresh_settings
from physverify.freeze.arguments import query_parser
from physverify.freeze.errors import QueryError
from physverify.freeze.files import read_json, readable_sha256
from physverify.freeze.fresh_render import VERSION_ENV, plan_lines, render_overrides
from physverify.freeze.ledger import ledger_entries, ledger_line, look_state, v1_state
from physverify.freeze.manifest import (
    ManifestInputs,
    artifact_problems,
    changed,
    chosen_candidate,
    field,
    manifest,
    record_sha,
    run_git,
)
from physverify.freeze.selection import DEVICE, PROFILE, candidate, seed_of, select

REPO_ROOT = Path(__file__).resolve().parents[2]
git = functools.partial(run_git, REPO_ROOT)


def _settings(extra: dict[str, str] | None = None) -> Settings:
    environ = {**os.environ, **(extra or {})}
    return load_settings(REPO_ROOT / "configs", environ=environ, profile=PROFILE, device=DEVICE)


def _home() -> Path:
    return Path(os.environ.get("HOME") or Path.home())


def _write_manifest(args: argparse.Namespace) -> list[str]:
    inputs = ManifestInputs(
        selection=read_json(args.selection),
        record_sha=args.record_sha,
        p2_commit=args.p2_commit,
        v1_commit=args.v1_commit,
        v1_calibration=args.v1_calibration,
    )
    settings = _settings()
    record = functools.partial(record_sha, settings, checks=plugins.build("checks"))
    data = manifest(inputs, settings, git, record=record, now=lambda: datetime.now(UTC))
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    j13 = "empty" if data["diffs"]["j13"]["empty"] else f"changes {data['diffs']['j13']['changed']}"
    return [
        f"freeze: {args.out} sha256 {readable_sha256(args.out)}",
        f"commit {data['commit']}, configs tree {data['configs_tree']}, seed {data['seed']}",
        f"J13 diff from P2 {args.p2_commit}: {j13}",
    ]


def _overrides(args: argparse.Namespace) -> dict[str, str]:
    return render_overrides(fresh_settings(_settings()), args.dataset, args.seed)


def _plan(args: argparse.Namespace) -> list[str]:
    overrides = _overrides(args)
    return plan_lines(_settings(overrides), overrides[VERSION_ENV], args.seed)


def _artifacts(args: argparse.Namespace) -> list[str]:
    problems = artifact_problems(read_json(args.freeze), _settings())
    if problems:
        raise QueryError("the frozen files changed:\n" + "\n".join(problems))
    return [f"frozen files unchanged ({args.freeze})"]


def _output(args: argparse.Namespace) -> list[str]:
    commands: dict[str, Callable[[], list[str]]] = {
        "select": lambda: [
            json.dumps(select([candidate(d) for d in args.dirs]), indent=2, sort_keys=True)
        ],
        "chosen": lambda: [field(chosen_candidate(read_json(args.selection)), args.key)],
        "seed": lambda: [str(seed_of(args.commit, args.calibration_sha, args.exclusions_sha))],
        "diff": lambda: changed(git, args.name, args.start),
        "manifest": lambda: _write_manifest(args),
        "field": lambda: [field(read_json(args.freeze), args.key)],
        "artifacts": lambda: _artifacts(args),
        "render-env": lambda: [f"{key}={value}" for key, value in _overrides(args).items()],
        "plan": lambda: _plan(args),
        "look-state": lambda: [
            look_state(ledger_entries(_settings(), _home()), args.specs, retry=args.retry)
        ],
        "v1-state": lambda: [
            v1_state(ledger_entries(_settings(), _home()), read_json(args.freeze), args.specs)
        ],
        "ledger": lambda: [ledger_line(_settings(), _home())],
    }
    return commands[args.command]()


def main(argv: Sequence[str] | None = None) -> int:
    args = query_parser((__doc__ or "").splitlines()[0]).parse_args(argv)
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
