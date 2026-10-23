"""Read-only queries run_human_agreement.sh asks of the checkout: a field of the
pre-registration, how many test comparisons are logged, the input gaps of a subset and a clip ready
to preflight. None of them reads a label.

Exit codes: 0 ok, 2 an input is unusable (message on stderr).

    python agreement_query.py prereg PATH FIELD   the field's value (empty for null)
    python agreement_query.py tested              logged test comparisons of the configured freeze
    python agreement_query.py log-path            the comparison log of the configured freeze
    python agreement_query.py gaps SUBSET         rows without an object query or a video
    python agreement_query.py ready SUBSET        the first clip with a video and an object query
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from run_query import query_settings

from physverify.app.queries import (
    QueryError,
    comparison_log,
    logged_tests,
    preregistration_field,
    ready_clip,
    subset_gaps,
)

if TYPE_CHECKING:
    from physverify.app.queries import InputGaps, Settings

SHOWN_CAPTIONS = 20


def prereg_field(path: Path, field: str) -> list[str]:
    value = preregistration_field(path, field)
    return ["" if value is None else value]


def tested(settings: Settings) -> list[str]:
    return [str(logged_tests(settings))]


def log_path(settings: Settings) -> list[str]:
    return [str(comparison_log(settings))]


def gap_lines(gaps: InputGaps) -> list[str]:
    lines = [
        f"- no object query: {gaps.no_query} rows ({gaps.no_query_with_video} with video)",
        f"- no video in the source: {gaps.no_video} rows",
    ]
    for title, groups in (("part", gaps.by_part), ("generator", gaps.by_generator)):
        for key, counts in sorted(groups.items()):
            pairs = ", ".join(f"{name} {n}" for name, n in sorted(counts.items()))
            lines.append(f"- {title} {key}: {pairs}")
    shown = gaps.no_query_captions[:SHOWN_CAPTIONS]
    more = len(gaps.no_query_captions) - len(shown)
    lines += [f"- no query: {caption}" for caption in shown]
    return lines + ([f"- ... and {more} more captions"] if more > 0 else [])


def gaps(settings: Settings, subset: str) -> list[str]:
    return gap_lines(subset_gaps(settings, subset))


def ready(settings: Settings, subset: str) -> list[str]:
    return [ready_clip(settings, subset)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prereg = sub.add_parser("prereg")
    prereg.add_argument("path", type=Path)
    prereg.add_argument("field")
    sub.add_parser("tested")
    sub.add_parser("log-path")
    sub.add_parser("gaps").add_argument("subset")
    sub.add_parser("ready").add_argument("subset")
    return parser


def _lines(args: argparse.Namespace) -> list[str]:
    commands: dict[str, Callable[[], list[str]]] = {
        "prereg": lambda: prereg_field(args.path, args.field),
        "tested": lambda: tested(query_settings(())),
        "log-path": lambda: log_path(query_settings(())),
        "gaps": lambda: gaps(query_settings(()), args.subset),
        "ready": lambda: ready(query_settings(()), args.subset),
    }
    return commands[args.command]()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        lines = _lines(args)
    except QueryError as exc:
        print(f"agreement_query: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
