"""Where the outputs of a finished run are, and their typed readers.

The run records each output's path in ``run/state.env``; the selective-risk curve is the last
``runs/<run>/abstention_report.json`` (run directories are named by start time), the one the final
run reads. A figure that needs another output adds an ``Output`` member and a reader here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import cast

from physverify.baselines.records import PartRows
from physverify.baselines.settings import VlmComparisonSettings
from physverify.calibration.artifact import EvaluationReport
from physverify.core.errors import PhysverifyError, UsageError
from physverify.core.jsonio import read_model
from physverify.diagnosis.report import AttributionReport
from physverify.reporting.chain import STATE_FILE, parse_state
from physverify.reporting.figures._base import MalformedInputError, NoDataError, Output
from physverify.reporting.final_rows import Criteria, load_evaluation
from physverify.selective_risk.records import CurveRecord
from physverify.selective_risk.settings import REPORT_FILE, AbstentionSettings

RUN_DIR = "run"
CURVE_PATTERN = f"runs/*/{REPORT_FILE}"


@dataclass(frozen=True)
class Located:
    paths: Mapping[Output, Path]
    gaps: Mapping[Output, str]
    """Why an output has no path."""


def locate(final_dir: Path) -> Located:
    if not final_dir.is_dir():
        raise UsageError(f"{final_dir} is not a directory")
    state = final_dir / RUN_DIR / STATE_FILE
    env = parse_state(state.read_text(encoding="utf-8")) if state.is_file() else None
    found = {
        key: _curve(final_dir) if key is Output.CURVE else _recorded(env, key, state)
        for key in Output
    }
    return Located(
        {k: v for k, v in found.items() if isinstance(v, Path)},
        {k: v for k, v in found.items() if isinstance(v, str)},
    )


def _recorded(env: Mapping[str, str] | None, key: Output, state: Path) -> Path | str:
    if env is None:
        return f"no {state}"
    value = env.get(key.value)
    if not value:
        return f"{state} records no {key.value}"
    path = Path(value)
    return path if path.is_file() else f"{key.value}={value} is not a file"


def _curve(final_dir: Path) -> Path | str:
    found = sorted(final_dir.glob(CURVE_PATTERN))
    return found[-1] if found else f"no {CURVE_PATTERN} under {final_dir}"


@dataclass(frozen=True)
class Malformed:
    """A run output that failed to parse, kept as the reason."""

    reason: str


@dataclass(frozen=True)
class Records:
    """The located outputs with their readers, the settings the closing tables judge with and
    those the selective-risk curve was computed with.

    Each output is parsed on first use and kept, a malformed one as its reason, so the figures of
    one render share a single read of it."""

    located: Located
    comparison: VlmComparisonSettings
    excluded: frozenset[str] = frozenset()
    criteria: Criteria = field(default_factory=Criteria)
    abstention: AbstentionSettings = field(default_factory=AbstentionSettings)
    _parsed: dict[Output, object] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def gap(self, key: Output) -> str | None:
        if key in self.located.paths:
            return None
        return self.located.gaps.get(key, f"{key.value} is not located")

    def path(self, key: Output) -> Path:
        gap = self.gap(key)
        if gap is not None:
            raise NoDataError(gap)
        return self.located.paths[key]

    def kubric(self) -> EvaluationReport:
        return self._once(Output.KUBRIC_EVAL, load_evaluation)

    def real(self) -> EvaluationReport:
        return self._once(Output.REAL_EVAL, load_evaluation)

    def attribution(self) -> AttributionReport:
        return self._once(Output.ATTRIBUTION, partial(read_model, AttributionReport))

    def curve(self) -> CurveRecord:
        return self._once(Output.CURVE, partial(read_model, CurveRecord))

    def kubric_rows(self) -> PartRows:
        return self._once(Output.VLM_KUBRIC, partial(read_model, PartRows))

    def _once[M](self, key: Output, read: Callable[[Path], M]) -> M:
        if key not in self._parsed:
            self._parsed[key] = _parse(key, read, self.path(key))
        found = self._parsed[key]
        if isinstance(found, Malformed):
            raise MalformedInputError(found.reason)
        return cast(M, found)


def _parse[M](key: Output, read: Callable[[Path], M], path: Path) -> M | Malformed:
    try:
        return read(path)
    except (OSError, ValueError, PhysverifyError) as error:
        return Malformed(f"{key.value} cannot be read: {error}")
