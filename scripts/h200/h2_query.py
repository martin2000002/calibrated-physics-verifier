"""Read-only queries run_h2.sh asks of the checkout: a field of the pre-registration, how many
test comparisons are logged, the input gaps of a subset and a clip ready to preflight. None of
them reads a label.

Exit codes: 0 ok, 2 an input is unusable (message on stderr).

    python h2_query.py prereg PATH FIELD   the field's value (empty for null)
    python h2_query.py tested              logged test comparisons of the configured freeze
    python h2_query.py log-path            the comparison log of the configured freeze
    python h2_query.py gaps SUBSET         rows without an object query or a video
    python h2_query.py ready SUBSET        the first clip with a video and an object query
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from a8v2_query import QueryError, _settings

SHOWN_CAPTIONS = 20


def prereg_field(path: Path, field: str) -> list[str]:
    from physverify.core.errors import PhysverifyError
    from physverify.human_agreement.prereg import Preregistration, load_preregistration

    if field not in Preregistration.model_fields:
        raise QueryError(f"no field {field!r} in a pre-registration")
    try:
        value = getattr(load_preregistration(path), field)
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc
    return ["" if value is None else str(value)]


def _preregistration(settings: Any) -> Path:
    from physverify.human_agreement.settings import agreement_settings, preregistration_path

    path: Path = preregistration_path(settings, agreement_settings(settings))
    return path


def tested(settings: Any) -> list[str]:
    from physverify.human_agreement.prereg import logged_comparisons

    return [str(len(logged_comparisons(_preregistration(settings))))]


def log_path(settings: Any) -> list[str]:
    from physverify.human_agreement.prereg import comparison_log_path

    return [str(comparison_log_path(_preregistration(settings)))]


def gap_lines(gaps: Any) -> list[str]:
    """``gaps`` is an ``human_agreement.coverage.InputGaps``."""
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


def _dataset(settings: Any) -> Any:
    from physverify.datasets.videophy import VideophyDataset

    return VideophyDataset(settings)


def _rows(dataset: Any, subset: str) -> Any:
    from physverify.core.errors import PhysverifyError

    try:
        return dataset.rows(subset)
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc


def gaps(settings: Any, subset: str) -> list[str]:
    from physverify.human_agreement.coverage import dataset_gaps

    dataset = _dataset(settings)
    return gap_lines(dataset_gaps(dataset, _rows(dataset, subset)))


def ready(settings: Any, subset: str) -> list[str]:
    from physverify.human_agreement.coverage import dataset_tests

    dataset = _dataset(settings)
    has_query, has_video = dataset_tests(dataset)
    for row in sorted(_rows(dataset, subset), key=lambda r: r.clip_id):
        if has_video(row) and has_query(row):
            return [row.clip_id]
    raise QueryError(f"no clip of {subset} has both a video and an object query")


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
        "tested": lambda: tested(_settings(())),
        "log-path": lambda: log_path(_settings(())),
        "gaps": lambda: gaps(_settings(()), args.subset),
        "ready": lambda: ready(_settings(()), args.subset),
    }
    return commands[args.command]()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        lines = _lines(args)
    except QueryError as exc:
        print(f"h2_query: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
