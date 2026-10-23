"""The exclusion decision record ``exclusions.json`` (ADR 0015 items 1.1, 1.2 and 1.5).

Written by ``exclusions decide``; H2, H3 and the scorer read it only through ``[check_scope]``.
It imports no consumer and holds no timestamp nor absolute path, so the same inputs give the same
bytes, and ``sha256`` (``stable_hash`` of the record without it) binds what was decided.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, ValidationError

from physverify.core.errors import UsageError
from physverify.core.models import SCHEMA_KEY, ArtifactModel
from physverify.schema.io import stable_hash

RECORD_SCHEMA: Final = "physverify.exclusions"
RECORD_VERSION: Final = "1.0.0"
SHA_FIELD: Final = "sha256"

EvidenceValue = float | int | str | bool | None


class ExclusionInputError(UsageError):
    """The inputs of a decision do not belong together, or a record does not hold."""


class RuleEvidence(ArtifactModel):
    """One rule applied to one check. ``decided`` False: the rule could not decide (it then
    never excludes); ``inputs`` are the numbers it read, flat ``name`` or ``dataset.name``."""

    rule: str
    check_id: str
    excluded: bool
    decided: bool
    reason: str
    inputs: dict[str, EvidenceValue] = Field(default_factory=dict)


class InputHashes(ArtifactModel):
    """What the decision read: content hashes and run ids, never paths."""

    artifact_sha256: str
    artifact_run_id: str
    evaluation_sha256: str
    evaluation_run_id: str
    r2_base_run: str
    r2_base_report_sha256: str
    r2_tiny_run: str
    r2_tiny_report_sha256: str
    rules_sha256: str


class ExclusionRecord(ArtifactModel):
    schema_: Literal["physverify.exclusions"] = Field(default=RECORD_SCHEMA, alias=SCHEMA_KEY)
    version: str = RECORD_VERSION
    rules_version: str
    consumed_level: str
    consumed_profile: str
    checks_hash: str
    inputs: InputHashes
    evidence: tuple[RuleEvidence, ...]
    exclude_checks: tuple[str, ...]
    h3_families_without_target: tuple[str, ...]
    replaces_sha256: str | None = None
    sha256: str = ""

    def payload(self) -> dict[str, Any]:
        """The JSON content without ``sha256`` (what the hash covers)."""
        data: dict[str, Any] = json.loads(self.to_json())
        data.pop(SHA_FIELD)
        return data

    def sealed(self) -> ExclusionRecord:
        return self.model_copy(update={SHA_FIELD: stable_hash(self.payload())})


def parse_record(text: str, label: str) -> ExclusionRecord:
    try:
        record = ExclusionRecord.model_validate_json(text)
    except ValidationError as exc:
        raise ExclusionInputError(f"{label} is not an exclusion record:\n{exc}") from exc
    if record.version != RECORD_VERSION:
        raise ExclusionInputError(
            f"{label} has version {record.version!r}; this code reads {RECORD_VERSION}"
        )
    if record.sealed().sha256 != record.sha256:
        raise ExclusionInputError(f"{label}: its sha256 does not match its content (edited?)")
    return record


def load_record(path: Path, expected_checks_hash: str) -> ExclusionRecord:
    """The record at ``path``; refused when edited or decided with other check parameters."""
    record = parse_record(path.read_text(encoding="utf-8"), str(path))
    if record.checks_hash != expected_checks_hash:
        raise ExclusionInputError(
            f"{path} was decided with other check parameters "
            f"({record.checks_hash[:12]} != {expected_checks_hash[:12]}): decide again"
        )
    return record
