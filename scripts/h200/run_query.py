"""Read-only queries the H200 scripts ask of the checkout: clip selections, the checks on an
evaluation and the input errors of a verify run.

Exit codes: 0 ok, 2 a check failed or an input is unusable (message on stderr).

    python run_query.py [--profile P] [--set K=V]... clips DATASET SUBSET [--split calib|eval]
    python run_query.py [--set K=V]... evaluation EVALUATION_JSON
    python run_query.py run-errors RUN_DIR
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from physverify.app.queries import (
    PROFILE,
    QueryError,
    allowed_real_sets,
    dataset_clip_ids,
    read_json,
    run_errors,
)
from physverify.app.queries import query_settings as checkout_settings

if TYPE_CHECKING:
    from physverify.app.queries import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_HEADING = "## H1 primary"


def evaluation_lines(path: Path, allowed: Sequence[Mapping[str, int]]) -> list[str]:
    """The real sets and the primary-view section of ``path``; refused without the primary view or
    with a real set rule 1.5 does not allow."""
    data = read_json(path)
    sizes = {ref["name"]: ref["n_clips"] for ref in data.get("real_sets", [])}
    if sizes not in [dict(a) for a in allowed]:
        raise QueryError(f"{path}: real_sets {sizes} not in the allowed sets {list(allowed)}")
    if data.get("h1_primary") is None:
        raise QueryError(
            f"{path} has no h1_primary: the evaluation predates that section, rerun from 6"
        )
    lines = ["Real sets: " + ", ".join(f"{k} {v}" for k, v in sorted(sizes.items())), ""]
    return lines + primary_section(path.with_suffix(".md"))


def primary_section(markdown: Path) -> list[str]:
    if not markdown.is_file():
        raise QueryError(f"no {markdown} next to the evaluation")
    lines = markdown.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(PRIMARY_HEADING)), None)
    if start is None:
        raise QueryError(f"{markdown} has no '{PRIMARY_HEADING}' section")
    return lines[start:]


def query_settings(overrides: Sequence[str], profile: str = PROFILE) -> Settings:
    """The checkout's settings; the profile also fixes the synthetic dataset version."""
    return checkout_settings(REPO_ROOT / "configs", overrides, profile)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--profile", default=PROFILE)
    sub = parser.add_subparsers(dest="command", required=True)
    clips = sub.add_parser("clips")
    clips.add_argument("dataset")
    clips.add_argument("subset")
    clips.add_argument("--split", choices=("calib", "eval"))
    sub.add_parser("evaluation").add_argument("path", type=Path)
    sub.add_parser("run-errors").add_argument("run_dir", type=Path)
    return parser


def _lines(args: argparse.Namespace) -> list[str]:
    commands: dict[str, Callable[[], list[str]]] = {
        "clips": lambda: dataset_clip_ids(
            query_settings(args.overrides, args.profile), args.dataset, args.subset, args.split
        ),
        "evaluation": lambda: evaluation_lines(
            args.path, allowed_real_sets(query_settings(args.overrides))
        ),
        "run-errors": lambda: run_errors(args.run_dir),
    }
    return commands[args.command]()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        lines = _lines(args)
    except QueryError as exc:
        print(f"run_query: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
