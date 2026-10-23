"""``[check_scope]`` merges the exclusion decision record: the one place the human-agreement
study and the generator comparison read it."""

from __future__ import annotations

import sys
from pathlib import Path

CALIBRATION_TESTS = str(Path(__file__).resolve().parent.parent / "calibration")
if CALIBRATION_TESTS not in sys.path:
    sys.path.insert(0, CALIBRATION_TESTS)

import pytest  # noqa: E402
from exclusion_cases import write_record  # noqa: E402
from wiring import active_checks_hash, checks  # noqa: E402

from physverify.config.loader import load_settings  # noqa: E402
from physverify.config.models import Settings  # noqa: E402
from physverify.core.errors import ConfigError, UsageError  # noqa: E402
from physverify.verification.scope import (  # noqa: E402
    CheckScope,
    check_scope,
    require_record_for,
)


@pytest.fixture
def make(config_dir: Path, environ: dict[str, str], tmp_path: Path):  # type: ignore[no-untyped-def]
    env = {**environ, "PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")}

    def build(*overrides: str) -> Settings:
        return load_settings(config_dir, environ=env, overrides=overrides)

    return build


def _record_path(settings: Settings) -> Path:
    return settings.paths.calibration_dir / "exclusions.json"


def test_without_a_record_the_scope_is_the_toml(make) -> None:  # type: ignore[no-untyped-def]
    scope = check_scope(make(), checks())
    assert scope.exclude_checks == () and scope.record is None and scope.record_sha256 is None


def test_the_record_ids_are_merged_and_its_sha_exposed(make) -> None:  # type: ignore[no-untyped-def]
    settings = make()
    found = write_record(
        _record_path(settings), ("R5a", "R1"), checks_hash=active_checks_hash(settings)
    )
    scope = check_scope(settings, checks())
    assert scope.exclude_checks == ("R1", "R5a")
    assert scope.record_sha256 == found.sha256
    assert scope.excluded(time_dependent=True) == {"R1", "R5a", "R5b", "R4_jump"}


def test_the_time_dependent_checks_are_the_ones_the_checks_declare(make) -> None:  # type: ignore[no-untyped-def]
    declared = {check_id for check_id, check in checks().items() if check.time_dependent}
    assert set(check_scope(make(), checks()).time_dependent_checks) == declared
    assert set(CheckScope().time_dependent_checks) == declared
    with pytest.raises(ConfigError, match="time_dependent"):
        check_scope(make('check_scope.time_dependent_checks=["R5b"]'), checks())


def test_the_sha_cannot_be_set_from_the_toml(make) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ConfigError):
        check_scope(make("check_scope.record_sha256=abc"), checks())


def test_a_required_record_that_is_missing_is_a_config_error(make) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ConfigError, match="exclusions decide"):
        check_scope(make("check_scope.require_decision_record=true"), checks())


def test_a_stale_or_edited_record_is_a_config_error(make) -> None:  # type: ignore[no-untyped-def]
    settings = make()
    write_record(_record_path(settings), ("R1",), checks_hash="other")
    with pytest.raises(ConfigError, match="other check parameters"):
        check_scope(settings, checks())


def test_manual_ids_next_to_a_record_need_a_declared_deviation(make) -> None:  # type: ignore[no-untyped-def]
    settings = make('check_scope.exclude_checks=["R3a"]')
    write_record(_record_path(settings), ("R1",), checks_hash=active_checks_hash(settings))
    with pytest.raises(ConfigError, match="manual_deviation"):
        check_scope(settings, checks())
    declared = make('check_scope.exclude_checks=["R3a"]', "check_scope.manual_deviation=why")
    assert check_scope(declared, checks()).exclude_checks == ("R1", "R3a")


def test_a_record_id_is_validated_against_the_registry(make) -> None:  # type: ignore[no-untyped-def]
    settings = make()
    write_record(_record_path(settings), ("R99",), checks_hash=active_checks_hash(settings))
    with pytest.raises(ConfigError, match="unknown checks"):
        check_scope(settings, checks())


def test_a_calibrated_rule_requires_a_record_of_its_artifact(make) -> None:  # type: ignore[no-untyped-def]
    settings = make()
    bare = check_scope(settings, checks())
    require_record_for(bare, calibrated=False, artifact_sha256=None)
    with pytest.raises(UsageError, match="exclusions decide"):
        require_record_for(bare, calibrated=True, artifact_sha256="a")
    write_record(
        _record_path(settings), (), artifact_sha256="a", checks_hash=active_checks_hash(settings)
    )
    scope = check_scope(settings, checks())
    require_record_for(scope, calibrated=True, artifact_sha256="a")
    with pytest.raises(UsageError, match="decide again"):
        require_record_for(scope, calibrated=True, artifact_sha256="b")
