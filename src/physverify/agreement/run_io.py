"""Reading a ``verify`` run back: ``run.json``, ``table.csv`` and ``checks.jsonl``."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from physverify.core.errors import MissingInputError
from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.schema.io import read_jsonl
from physverify.verification.runner import (
    CHECKS_FILE,
    TABLE_FILE,
    THRESHOLDS_KEY,
    RowStatus,
    TableRow,
)

RUN_FILE = "run.json"


@dataclass(frozen=True)
class VerifyRun:
    """One clip's evidence is ``rows[clip_id]`` (one per check) and ``results[clip_id]``."""

    path: Path
    run: Mapping[str, Any]
    rows: Mapping[str, tuple[TableRow, ...]]
    results: Mapping[str, tuple[CheckResult, ...]]

    @property
    def profile(self) -> str:
        return str(self.run.get("profile", ""))

    @property
    def device(self) -> str:
        return str(self.run.get("device", ""))

    def thresholds_labels(self) -> frozenset[str]:
        return frozenset(
            r.provenance[THRESHOLDS_KEY]
            for clip in self.results.values()
            for r in clip
            if THRESHOLDS_KEY in r.provenance
        )


def _optional(text: str) -> str | None:
    return text if text != "" else None


def _number(text: str) -> float | None:
    return float(text) if text != "" else None


def parse_row(record: Mapping[str, str]) -> TableRow:
    verdict, reason = _optional(record["verdict"]), _optional(record["abstain_reason"])
    magnitude = _number(record["magnitude"])
    sigma = _number(record["sigma"])
    return TableRow(
        clip_id=record["clip_id"],
        check_id=record["check_id"],
        status=RowStatus(record["status"]),
        magnitude=magnitude,
        sigma=math.nan if sigma is None else sigma,
        verdict=None if verdict is None else Verdict(verdict),
        abstain_reason=None if reason is None else AbstainReason(reason),
        n_subjects=int(record["n_subjects"]),
        n_abstained=int(record["n_abstained"]),
        unit=record["unit"],
        row_step=int(record["row_step"]),
    )


def read_table(path: Path) -> list[TableRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = [f.name for f in fields(TableRow)]
        if reader.fieldnames != expected:
            raise MissingInputError(f"{path}: columns {reader.fieldnames} are not {expected}")
        return [parse_row(record) for record in reader]


def _grouped[T](items: Sequence[T], key: str) -> dict[str, tuple[T, ...]]:
    groups: dict[str, list[T]] = defaultdict(list)
    for item in items:
        groups[getattr(item, key)].append(item)
    return {name: tuple(group) for name, group in groups.items()}


def load_run(run_dir: Path) -> VerifyRun:
    paths = [run_dir / name for name in (RUN_FILE, TABLE_FILE, CHECKS_FILE)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise MissingInputError(f"not a finished verify run: missing {', '.join(missing)}")
    run = json.loads(paths[0].read_text(encoding="utf-8"))
    rows = _grouped(read_table(paths[1]), "clip_id")
    results = _grouped(read_jsonl(CheckResult, paths[2]), "clip_id")
    return VerifyRun(path=run_dir, run=run, rows=rows, results=results)
