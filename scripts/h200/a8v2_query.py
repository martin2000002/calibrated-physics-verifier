"""Read-only queries the H200 scripts ask of the checkout: clip selections, the checks on an
evaluation and the input errors of a verify run.

Exit codes: 0 ok, 2 a check failed or an input is unusable (message on stderr).

    python a8v2_query.py [--profile P] [--set K=V]... clips DATASET SUBSET [--split calib|eval]
    python a8v2_query.py [--set K=V]... evaluation EVALUATION_JSON
    python a8v2_query.py run-errors RUN_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = "precise"
DEVICE = "cpu"
H1_HEADING = "## H1 primary"
ERROR_KEY = "error"
SHOWN_ERRORS = 10


class QueryError(Exception):
    """A check failed or an input is unusable (exit 2)."""


class Clip(Protocol):
    @property
    def clip_id(self) -> str: ...

    @property
    def split(self) -> Any: ...


class Source(Protocol):
    def clips(self, subset: str | None = None) -> Sequence[Clip]: ...


def clip_ids(source: Source, subset: str, split: str | None) -> list[str]:
    clips = source.clips(subset)
    return sorted(c.clip_id for c in clips if split is None or c.split.value == split)


def evaluation_lines(path: Path, allowed: Sequence[Mapping[str, int]]) -> list[str]:
    """The real sets and the H1 section of ``path``; refused without the primary view or with a
    real set rule 1.5 does not allow."""
    data = _json(path)
    sizes = {ref["name"]: ref["n_clips"] for ref in data.get("real_sets", [])}
    if sizes not in [dict(a) for a in allowed]:
        raise QueryError(f"{path}: real_sets {sizes} not in the allowed sets {list(allowed)}")
    if data.get("h1_primary") is None:
        raise QueryError(f"{path} has no h1_primary: the evaluation predates WP-J2, rerun from 6")
    lines = ["Real sets: " + ", ".join(f"{k} {v}" for k, v in sorted(sizes.items())), ""]
    return lines + h1_section(path.with_suffix(".md"))


def h1_section(markdown: Path) -> list[str]:
    if not markdown.is_file():
        raise QueryError(f"no {markdown} next to the evaluation")
    lines = markdown.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(H1_HEADING)), None)
    if start is None:
        raise QueryError(f"{markdown} has no '{H1_HEADING}' section")
    return lines[start:]


def run_errors(run_dir: Path) -> list[str]:
    """Refused when a check row of the run carries an input or check error: verify exits 0 on an
    uncached clip and only marks its rows abstained, with the error in ``details``."""
    from physverify.schema.checks import CheckResult
    from physverify.schema.io import read_jsonl
    from physverify.verification.runner import CHECKS_FILE

    path = run_dir / CHECKS_FILE
    try:
        rows = read_jsonl(CheckResult, path)
    except (OSError, ValueError) as exc:
        raise QueryError(f"cannot read {path}: {exc}") from exc
    errors = [
        f"{r.clip_id} {r.check_id}: {r.details[ERROR_KEY] or '(no message)'}"
        for r in rows
        if ERROR_KEY in r.details
    ]
    if errors:
        more = len(errors) - SHOWN_ERRORS
        shown = errors[:SHOWN_ERRORS] + ([f"... and {more} more"] if more > 0 else [])
        raise QueryError(f"{len(errors)} check rows of {path} failed:\n" + "\n".join(shown))
    return [f"{path}: {len(rows)} check rows, none with an error"]


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QueryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise QueryError(f"{path} is not a JSON object")
    return data


def _settings(overrides: Sequence[str], profile: str = PROFILE) -> Any:
    """The checkout's settings; the profile also fixes the synthetic dataset version."""
    from physverify.config.loader import load_settings

    return load_settings(
        REPO_ROOT / "configs",
        environ=os.environ,
        profile=profile,
        device=DEVICE,
        overrides=overrides,
    )


def _source(settings: Any, dataset: str) -> Source:
    from physverify.app import plugins

    registry = plugins.build("datasets")
    if dataset not in registry:
        raise QueryError(f"unknown dataset {dataset!r}; known: {sorted(registry)}")
    source: Source = registry[dataset](settings)
    return source


def _allowed_sets(settings: Any) -> Sequence[Mapping[str, int]]:
    from physverify.app import plugins
    from physverify.calibration.exclusions import SECTION, exclusion_rules

    rules = exclusion_rules(settings.sections.get(SECTION, {}), plugins.build("checks"))
    return list(rules.real.allowed_sets)


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
        "clips": lambda: clip_ids(
            _source(_settings(args.overrides, args.profile), args.dataset), args.subset, args.split
        ),
        "evaluation": lambda: evaluation_lines(args.path, _allowed_sets(_settings(args.overrides))),
        "run-errors": lambda: run_errors(args.run_dir),
    }
    return commands[args.command]()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        lines = _lines(args)
    except QueryError as exc:
        print(f"a8v2_query: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
