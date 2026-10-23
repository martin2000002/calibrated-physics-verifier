"""The arguments of scripts/h200/freeze_query.py, one subcommand per query."""

from __future__ import annotations

import argparse
from pathlib import Path

from physverify.freeze.manifest import DIFF_PATHS


def query_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
    sub.add_parser("ledger")
    return parser
