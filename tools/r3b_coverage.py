"""Coverage of a scoped check: clips with an in-scope event over clips with an event.

    PYTHONPATH=src python tools/r3b_coverage.py checks.jsonl [more.jsonl ...] [--roles roles.csv]

Each input holds serialized ``CheckResult`` records (a ``verify`` run's ``checks.jsonl``). A clip
has an event when the check returned a result for it, and is in scope when one of those results is
not ``NOT_APPLICABLE``. ``roles.csv`` (columns ``clip_id,role``) splits the counts by role;
without it every clip is ``all``. Only reasons and ``details`` are read, never verdicts against
a threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.io import read_jsonl

DEFAULT_CHECK: Final = "R3b"
ALL_ROLES: Final = "all"
PREDICATES_KEY: Final = "not_applicable_predicates"


@dataclass
class Cell:
    clips_with_event: int = 0
    clips_in_scope: int = 0
    failed: Counter[str] = field(default_factory=Counter)

    def add(self, found: Sequence[CheckResult]) -> None:
        self.clips_with_event += 1
        self.clips_in_scope += any(map(in_scope, found))
        self.failed.update(p for r in found if not in_scope(r) for p in r.details[PREDICATES_KEY])

    def summary(self) -> dict[str, object]:
        return {
            "clips_with_event": self.clips_with_event,
            "clips_in_scope": self.clips_in_scope,
            "coverage": self.clips_in_scope / self.clips_with_event,
            "failed": dict(sorted(self.failed.items())),
        }


def in_scope(result: CheckResult) -> bool:
    return result.abstain_reason is not AbstainReason.NOT_APPLICABLE


def coverage(
    results: Iterable[CheckResult], roles: Mapping[str, str], check_id: str = DEFAULT_CHECK
) -> dict[str, dict[str, dict[str, object]]]:
    """``{input level: {role: summary}}``, one clip counted once per level."""
    clips: dict[tuple[str, str], list[CheckResult]] = defaultdict(list)
    for result in results:
        if result.check_id == check_id:
            clips[(result.input_level.value, result.clip_id)].append(result)
    cells: dict[str, dict[str, Cell]] = defaultdict(lambda: defaultdict(Cell))
    for (level, clip_id), found in clips.items():
        cells[level][roles.get(clip_id, ALL_ROLES)].add(found)
    return {
        level: {role: cell.summary() for role, cell in sorted(by_role.items())}
        for level, by_role in sorted(cells.items())
    }


def read_roles(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {record["clip_id"]: record["role"] for record in csv.DictReader(handle)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("checks", nargs="+", type=Path)
    parser.add_argument("--roles", type=Path)
    parser.add_argument("--check", default=DEFAULT_CHECK)
    args = parser.parse_args(argv)
    results = [r for path in args.checks for r in read_jsonl(CheckResult, path)]
    print(json.dumps(coverage(results, read_roles(args.roles), args.check), indent=2))


if __name__ == "__main__":
    main()
