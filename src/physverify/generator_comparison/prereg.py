"""Pre-registration of the generator comparison: written by ``compare-generators freeze`` before any
eval score or judge row exists; ``score``, ``judge`` and ``compare`` recompute every identity field
and refuse when one changed. A refreeze archives the replaced file, so the chain back to the first
freeze is kept. Once eval data exist, any refreeze needs a declared deviation and makes the verdict
post hoc."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pydantic import Field, ValidationError

from physverify.core.errors import ConfigError, PhysverifyError, UsageError
from physverify.core.models import FrozenModel
from physverify.finetune.records import read_record, write_record
from physverify.finetune.stop_rule import StopRuleState
from physverify.generator_comparison.layout import ComparisonLayout
from physverify.generator_comparison.margins import MarginDerivation
from physverify.generator_comparison.records import Usability
from physverify.generator_comparison.verdict import RuleStatus
from physverify.human_agreement.prereg import archive_replaced, changed_fields, walk_chain

EVAL_DATA_DEVIATION: Final = "ran after eval data existed"
JUDGE_STAMPS: Final = "judge_stamps"
FREEZE_HINT: Final = "run `physverify compare-generators freeze`"


class Mde(FrozenModel):
    """Minimum detectable effect at the frozen n; the sd come from dev (estimated)."""

    n: int
    alpha: float
    power: float
    verifier_sd: float | None
    verifier_mde: float | None
    pc_sd: float | None
    pc_mde: float | None
    pc_underpowered: bool | None


class CandidateSnapshot(FrozenModel):
    n: int
    sha256: str


class ComparisonPreregistration(FrozenModel):
    commit: str
    section_hashes: dict[str, str]
    prompts_sha256: str | None
    eval_counts: dict[str, int]
    selection_run_id: str
    adapter: str | None
    adapter_tag: str | None
    calibration_artifact_sha256: str | None
    rule: str
    scorer_version: str
    rule_label: str | None
    rule_status: RuleStatus
    calibration_dataset: dict[str, Any] | None
    contaminated: bool | None
    judge_stamps: dict[str, str]
    gate_sha256: str | None
    margins: dict[str, float]
    margin_derivations: dict[str, MarginDerivation]
    margin_source_sha256: str | None
    bootstrap_seed: int
    mde: Mde
    usability: Usability | None
    scipy_version: str
    eval_candidates_at_freeze: CandidateSnapshot
    stop_rule: StopRuleState
    created_utc: str
    exclusion_record_sha256: str | None = None
    """The exclusion decision record the scorer's excluded checks came from (None: none)."""
    secondary_tests: list[str] = Field(default_factory=list)
    """The sensitivities ``analyse`` reports (``analysis.SECONDARY_TESTS``); frozen."""
    stop_rule_deviation: str | None = None
    """The text that let a fired pilot stop rule through: the generator comparison then has no
    verdict."""
    replaces_sha256: str | None = None
    deviations: list[str] = Field(default_factory=list)
    post_hoc: bool = False


AUDIT_FIELDS: Final = frozenset(
    {
        "commit",
        "created_utc",
        "replaces_sha256",
        "deviations",
        "post_hoc",
        "eval_candidates_at_freeze",
        "stop_rule_deviation",
    }
)
"""Recorded and reported, never refused on: a code fix after the freeze is shown, not blocked."""
H3_FIELDS: Final = tuple(f for f in ComparisonPreregistration.model_fields if f not in AUDIT_FIELDS)
REFREEZE_MUTABLE: Final = frozenset(
    {
        "calibration_artifact_sha256",
        "rule",
        "scorer_version",
        "rule_label",
        "rule_status",
        "calibration_dataset",
        "contaminated",
        "mde",
        "stop_rule",
        "exclusion_record_sha256",
    }
)


def identity(record: ComparisonPreregistration) -> dict[str, object]:
    return record.model_dump(include=set(H3_FIELDS))


def stamp_problem(
    scorer_version: str, label: str | None, prereg: ComparisonPreregistration
) -> str | None:
    """Why a scorer stamp (version, thresholds label) is not the frozen one."""
    if (scorer_version, label) == (prereg.scorer_version, prereg.rule_label):
        return None
    return (
        f"scorer version {scorer_version!r} with label {label!r} is not the frozen "
        f"{prereg.scorer_version!r} with {prereg.rule_label!r}"
    )


def assert_complete(current: Mapping[str, object]) -> None:
    """``changed_fields`` skips names missing from ``current``; a missing name is a bug here."""
    missing = sorted(set(H3_FIELDS) - current.keys())
    if missing:
        raise PhysverifyError(
            f"the generator-comparison state lacks pre-registered fields {missing}"
        )


def require_unchanged(frozen: ComparisonPreregistration, current: Mapping[str, object]) -> None:
    assert_complete(current)
    changed = changed_fields(frozen, current, fields=H3_FIELDS)
    if changed:
        raise ConfigError(
            "the generator-comparison pre-registration no longer holds "
            f"({', '.join(changed)} changed); "
            f"{FREEZE_HINT} --force with a declared deviation, or restore the frozen state"
        )


@dataclass(frozen=True)
class FreezeRequest:
    force: bool
    deviation: str | None
    eval_data: bool
    gate_exists: bool


def mutable(gate_exists: bool) -> frozenset[str]:
    """Judge stamps may change only while the gate has not run."""
    return REFREEZE_MUTABLE if gate_exists else REFREEZE_MUTABLE | {JUDGE_STAMPS}


def _deviations(
    old: ComparisonPreregistration, changed: list[str], request: FreezeRequest
) -> list[str]:
    added = [] if request.deviation is None else [f"{request.deviation} (changed: {changed})"]
    late = [f"{EVAL_DATA_DEVIATION} (changed: {changed})"] if request.eval_data else []
    return [*old.deviations, *added, *late]


def plan_freeze(
    old: ComparisonPreregistration | None,
    old_sha256: str | None,
    current: ComparisonPreregistration,
    request: FreezeRequest,
) -> ComparisonPreregistration | None:
    """The record to write, or None when the frozen state still holds."""
    state = identity(current)
    assert_complete(state)
    _require_known_provenance(current)
    if old is None:
        if request.eval_data:
            raise UsageError("eval scores or judge rows already exist; the freeze is not blind")
        return current
    changed = changed_fields(old, state, fields=H3_FIELDS)
    if not changed:
        return None
    if not request.force:
        raise UsageError(f"the pre-registered state changed ({changed}); refreeze with --force")
    outside = [name for name in changed if name not in mutable(request.gate_exists)]
    if outside and request.eval_data:
        raise UsageError(f"{outside} cannot change once eval data exist")
    if request.eval_data and request.deviation is None:
        raise UsageError(
            f"{changed} changed after eval data exist: the verdict becomes post hoc; "
            "refreeze with --declare-deviation TEXT"
        )
    if outside and request.deviation is None:
        raise UsageError(f"{outside} changed: refreeze with --declare-deviation TEXT")
    return current.model_copy(
        update={
            "replaces_sha256": old_sha256,
            "deviations": _deviations(old, changed, request),
            "post_hoc": old.post_hoc or request.eval_data,
        }
    )


def _require_known_provenance(current: ComparisonPreregistration) -> None:
    """Unknown provenance is never taken as clean."""
    if current.rule_status == "calibrated" and current.contaminated is None:
        raise UsageError(
            "the calibration dataset of the calibrated rule is unknown (artefact missing or "
            "unreadable), so contamination cannot be ruled out; the freeze is refused"
        )


def load_prereg(path: Path) -> ComparisonPreregistration:
    if not path.is_file():
        raise UsageError(f"{path} is missing; {FREEZE_HINT}")
    try:
        return read_record(path, ComparisonPreregistration)
    except ValidationError as exc:
        raise ConfigError(f"{path} is invalid:\n{exc}") from exc


def optional_prereg(path: Path) -> ComparisonPreregistration | None:
    return load_prereg(path) if path.is_file() else None


def write_prereg(layout: ComparisonLayout, record: ComparisonPreregistration) -> None:
    """Archives the file it replaces under its sha256, then writes the new one."""
    archive_replaced(layout.prereg, layout.replaced)
    write_record(layout.prereg, record)


def chain(
    layout: ComparisonLayout, record: ComparisonPreregistration
) -> list[ComparisonPreregistration]:
    """The record and every archived one it replaces, newest first (stops at a missing link)."""
    return walk_chain(record, lambda sha: optional_prereg(layout.replaced(sha)))


def chain_shas(
    layout: ComparisonLayout, record: ComparisonPreregistration, sha256: str
) -> set[str]:
    """sha256 of ``record``'s file and of every version it replaced."""
    return {sha256} | {r.replaces_sha256 for r in chain(layout, record) if r.replaces_sha256}


def first_created(layout: ComparisonLayout, record: ComparisonPreregistration) -> str:
    return chain(layout, record)[-1].created_utc
