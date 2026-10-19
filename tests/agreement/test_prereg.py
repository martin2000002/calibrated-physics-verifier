"""The pre-registration file and its frozen fields (ADR 0011 D10, C5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from physverify.agreement.prereg import (
    FROZEN_FIELDS,
    LoggedComparison,
    Preregistration,
    changed_fields,
    file_sha256,
    frozen_before,
    load_preregistration,
    log_comparison,
    logged_comparisons,
    plan_refreeze,
    preregistration_chain,
    replaced_path,
    require_unchanged,
    write_preregistration,
)
from physverify.core.errors import ConfigError, MissingInputError, UsageError

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


class OtherRecord(BaseModel):
    seed: int
    commit: str
    note: str


OTHER = OtherRecord(seed=1, commit="abc", note="free")
OTHER_FIELDS = ("seed", "commit")


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


def test_another_model_is_checked_on_its_own_fields() -> None:
    current = {"seed": 2, "commit": "abc", "note": "edited", "rule": "other"}
    assert changed_fields(OTHER, current, fields=OTHER_FIELDS) == ["seed"]
    with pytest.raises(ConfigError, match="seed: 1 -> 2"):
        require_unchanged(OTHER, current, fields=OTHER_FIELDS)


def test_fields_outside_the_list_or_absent_from_the_state_are_ignored() -> None:
    require_unchanged(OTHER, {"commit": "abc", "note": "edited"}, fields=OTHER_FIELDS)
    assert changed_fields(OTHER, {"note": "edited"}, fields=OTHER_FIELDS) == []


def test_a_missing_or_invalid_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(MissingInputError, match="agreement freeze"):
        load_preregistration(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"commit": 1}', encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid"):
        load_preregistration(bad)


def test_the_rule_label_is_frozen() -> None:
    assert "rule_label" in FROZEN_FIELDS
    assert changed_fields(RECORD, _current(rule_label="calibrated:x")) == ["rule_label"]


def _plan(old: Preregistration | None, current: Preregistration, **kwargs: object):
    options: dict[str, object] = {"force": False, "deviation": None, "tested": False, **kwargs}
    return plan_refreeze(old, None if old is None else "a" * 64, current, **options)  # type: ignore[arg-type]


def test_a_first_freeze_is_written_and_an_unchanged_one_is_kept() -> None:
    first = _plan(None, RECORD)
    assert first is not None and first.replaces_sha256 is None and not first.post_hoc
    assert _plan(RECORD, RECORD) is None


def test_a_changed_state_needs_force_and_records_what_it_replaces() -> None:
    other = RECORD.model_copy(update={"profile": "fast"})
    with pytest.raises(UsageError, match="profile"):
        _plan(RECORD, other)
    forced = _plan(RECORD, other, force=True)
    assert forced is not None and forced.replaces_sha256 == "a" * 64
    assert not forced.post_hoc and forced.deviations == []


def test_a_refreeze_after_a_test_comparison_must_declare_a_deviation() -> None:
    other = RECORD.model_copy(update={"rule_label": "calibrated:x"})
    with pytest.raises(UsageError, match="--declare-deviation"):
        _plan(RECORD, other, force=True, tested=True)
    late = _plan(RECORD, other, force=True, tested=True, deviation="new thresholds")
    assert late is not None and late.post_hoc
    assert late.deviations == ["new thresholds (changed: ['rule_label'])"]
    later = _plan(late, other.model_copy(update={"profile": "fast"}), force=True)
    assert later is not None and later.post_hoc and later.deviations == late.deviations


def test_a_replaced_file_is_archived_and_the_chain_walks_back(tmp_path: Path) -> None:
    path = tmp_path / "preregistration.json"
    write_preregistration(path, RECORD)
    first_sha = file_sha256(path)
    assert first_sha is not None
    second = RECORD.model_copy(update={"replaces_sha256": first_sha, "profile": "fast"})
    write_preregistration(path, second)
    assert load_preregistration(replaced_path(path, first_sha)) == RECORD
    assert preregistration_chain(path, second) == [second, RECORD]


def test_the_comparison_log_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "preregistration.json"
    assert logged_comparisons(path) == []
    entries = [
        LoggedComparison(prereg_sha256="a", run=f"runs/{n}", compared_utc="2026-10-18T00:00:00")
        for n in range(2)
    ]
    for entry in entries:
        log_comparison(path, entry)
    assert logged_comparisons(path) == entries


def test_a_freeze_is_blind_only_before_the_earliest_eval_file() -> None:
    assert frozen_before("2026-10-17T00:00:00+00:00", "2026-10-18T00:00:00+00:00") is True
    assert frozen_before("2026-10-19T00:00:00+00:00", "2026-10-18T00:00:00+00:00") is False
    assert frozen_before("2026-10-19T00:00:00+00:00", None) is None
