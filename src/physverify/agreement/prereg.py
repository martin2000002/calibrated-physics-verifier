"""Pre-registration of the test comparison (ADR 0011 D10, Juicio changes 2 and 8).

Written by ``agreement freeze`` from the dev run, before any test run is compared. ``compare``
on the test subset recomputes every frozen hash and refuses to run when one changed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from physverify.core.errors import ConfigError, MissingInputError
from physverify.diagnosis.atomic_io import write_text_atomic
from physverify.extraction.weights import sha256_of

FROZEN_FIELDS = (
    "commit",
    "agreement_config_sha256",
    "subset_sha256",
    "queries_sha256",
    "calibration_artifact_sha256",
    "rule",
    "profile",
)


class Preregistration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    created_utc: str
    commit: str
    agreement_config_sha256: str
    subset_sha256: dict[str, str | None]
    queries_sha256: str | None
    calibration_artifact_sha256: str | None
    rule: str
    rule_label: str
    profile: str
    device: str
    dev_run: str
    dev_subset: str
    test_subset: str
    n_dev: int
    dev_majority: int | None
    dev_cut: float | None
    dev_tuned_cut: float | None
    kubric_threshold: float | None
    replaces_sha256: str | None = None


def file_sha256(path: Path) -> str | None:
    return sha256_of(path) if path.is_file() else None


def write_preregistration(path: Path, record: Preregistration) -> None:
    write_text_atomic(path, record.model_dump_json(indent=2) + "\n")


def load_preregistration(path: Path) -> Preregistration:
    if not path.is_file():
        raise MissingInputError(
            f"{path} is missing; run `physverify agreement freeze` on the dev run"
        )
    try:
        return Preregistration.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ConfigError(f"{path} is invalid:\n{exc}") from exc


def changed_fields(frozen: Preregistration, current: Mapping[str, object]) -> list[str]:
    """Names of the frozen fields whose current value differs."""
    stored = frozen.model_dump()
    return [name for name in FROZEN_FIELDS if name in current and stored[name] != current[name]]


def require_unchanged(frozen: Preregistration, current: Mapping[str, object]) -> None:
    changed = changed_fields(frozen, current)
    if changed:
        detail = ", ".join(f"{n}: {frozen.model_dump()[n]!r} -> {current[n]!r}" for n in changed)
        raise ConfigError(
            f"the pre-registration no longer holds ({detail}); the test is not comparable"
        )
