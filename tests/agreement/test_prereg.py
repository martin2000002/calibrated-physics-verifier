"""The pre-registration file and its frozen fields (ADR 0011 D10, C5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from physverify.agreement.prereg import (
    FROZEN_FIELDS,
    Preregistration,
    changed_fields,
    load_preregistration,
    require_unchanged,
    write_preregistration,
)
from physverify.core.errors import ConfigError, MissingInputError

RECORD = Preregistration(
    created_utc="2026-10-17T00:00:00Z",
    commit="abc",
    agreement_config_sha256="c" * 64,
    subset_sha256={"v1_test": "s" * 64, "v2_test": None},
    queries_sha256=None,
    calibration_artifact_sha256=None,
    rule="provisional_any_fail",
    rule_label="provisional_any_fail/provisional",
    profile="precise",
    device="cuda",
    dev_run="runs/dev",
    dev_subset="h2_dev",
    test_subset="h2_test",
    n_dev=3,
    dev_majority=0,
    dev_cut=None,
    dev_tuned_cut=0.5,
    kubric_threshold=None,
)


def _current(**changes: object) -> dict[str, object]:
    stored = RECORD.model_dump()
    return {**{name: stored[name] for name in FROZEN_FIELDS}, **changes}


def test_an_unchanged_state_passes_and_a_round_trip_is_identical(tmp_path: Path) -> None:
    require_unchanged(RECORD, _current())
    path = tmp_path / "preregistration.json"
    write_preregistration(path, RECORD)
    assert load_preregistration(path) == RECORD


def test_every_changed_frozen_field_is_named() -> None:
    current = _current(commit="def", profile="fast", dev_cut=9.0)
    assert changed_fields(RECORD, current) == ["commit", "profile"]
    with pytest.raises(ConfigError, match="commit: 'abc' -> 'def'"):
        require_unchanged(RECORD, current)


def test_a_missing_or_invalid_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(MissingInputError, match="agreement freeze"):
        load_preregistration(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"commit": 1}', encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid"):
        load_preregistration(bad)
