"""Pre-registration of H3 (ADR 0013 D9): written by ``h3 freeze`` before any eval score or judge
row exists; ``score``, ``judge`` and ``compare`` recompute every identity field and refuse when one
changed. A refreeze archives the replaced file, so the chain back to the first freeze is kept.
Once eval data exist, any refreeze needs a declared deviation and makes the verdict post hoc.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pydantic import Field, ValidationError

from physverify.agreement.prereg import archive_replaced, changed_fields, walk_chain
from physverify.core.errors import ConfigError, PhysverifyError, UsageError
from physverify.core.models import FrozenModel
from physverify.evaluation.layout import H3Layout
from physverify.evaluation.margins import MarginDerivation
from physverify.evaluation.records import Usability
from physverify.evaluation.verdict import RuleStatus
from physverify.finetune.records import read_record, write_record
from physverify.finetune.stop_rule import StopRuleState

EVAL_DATA_DEVIATION: Final = "ran after eval data existed"
JUDGE_STAMPS: Final = "judge_stamps"
FREEZE_HINT: Final = "run `physverify h3 freeze`"


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


class H3Preregistration(FrozenModel):
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
    """The ADR 0015 decision record the scorer's excluded checks came from (None: none)."""
    secondary_tests: list[str] = Field(default_factory=list)
    """The sensitivities ``analyse`` reports (``analysis.SECONDARY_TESTS``); frozen."""
    stop_rule_deviation: str | None = None
    """The text that let a fired pilot stop rule through (ADR 0012): H3 then has no verdict."""
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
H3_FIELDS: Final = tuple(f for f in H3Preregistration.model_fields if f not in AUDIT_FIELDS)
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


def identity(record: H3Preregistration) -> dict[str, object]:
    return record.model_dump(include=set(H3_FIELDS))


def stamp_problem(scorer_version: str, label: str | None, prereg: H3Preregistration) -> str | None:
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
        raise PhysverifyError(f"the H3 state lacks pre-registered fields {missing}")


def require_unchanged(frozen: H3Preregistration, current: Mapping[str, object]) -> None:
    assert_complete(current)
    changed = changed_fields(frozen, current, fields=H3_FIELDS)
    if changed:
        raise ConfigError(
            f"the H3 pre-registration no longer holds ({', '.join(changed)} changed); "
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


def _deviations(old: H3Preregistration, changed: list[str], request: FreezeRequest) -> list[str]:
    added = [] if request.deviation is None else [f"{request.deviation} (changed: {changed})"]
    late = [f"{EVAL_DATA_DEVIATION} (changed: {changed})"] if request.eval_data else []
    return [*old.deviations, *added, *late]


def plan_freeze(
    old: H3Preregistration | None,
    old_sha256: str | None,
    current: H3Preregistration,
    request: FreezeRequest,
) -> H3Preregistration | None:
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


def _require_known_provenance(current: H3Preregistration) -> None:
    """Unknown provenance is never taken as clean (ADR 0013 D9)."""
    if current.rule_status == "calibrated" and current.contaminated is None:
        raise UsageError(
            "the calibration dataset of the calibrated rule is unknown (artefact missing or "
            "unreadable), so contamination cannot be ruled out; the freeze is refused"
        )


def load_prereg(path: Path) -> H3Preregistration:
    if not path.is_file():
        raise UsageError(f"{path} is missing; {FREEZE_HINT}")
    try:
        return read_record(path, H3Preregistration)
    except ValidationError as exc:
        raise ConfigError(f"{path} is invalid:\n{exc}") from exc


def optional_prereg(path: Path) -> H3Preregistration | None:
    return load_prereg(path) if path.is_file() else None


def write_prereg(layout: H3Layout, record: H3Preregistration) -> None:
    """Archives the file it replaces under its sha256, then writes the new one."""
    archive_replaced(layout.prereg, layout.replaced)
    write_record(layout.prereg, record)


def chain(layout: H3Layout, record: H3Preregistration) -> list[H3Preregistration]:
    """The record and every archived one it replaces, newest first (stops at a missing link)."""
    return walk_chain(record, lambda sha: optional_prereg(layout.replaced(sha)))


def chain_shas(layout: H3Layout, record: H3Preregistration, sha256: str) -> set[str]:
    """sha256 of ``record``'s file and of every version it replaced."""
    return {sha256} | {r.replaces_sha256 for r in chain(layout, record) if r.replaces_sha256}


def first_created(layout: H3Layout, record: H3Preregistration) -> str:
    return chain(layout, record)[-1].created_utc
