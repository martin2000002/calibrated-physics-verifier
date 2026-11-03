"""``extract run --strict-profile``: a missing or stale error profile is an error, not a warning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from cli_runner import CliRunner
from fakes import FAKE
from settings_factory import SettingsFactory
from support import entry, plan_overrides, profile_of
from synthetic_clip import SyntheticFixtureDataset
from test_cli_extract import FAKE_PLAN, RUN, fixture_registries  # noqa: F401
from wiring import resolve_plan

from physverify.core.errors import ConfigError
from physverify.extraction._base import StageProvider
from physverify.extraction.cache import plan_params_hashes
from physverify.extraction.error_profile import require_current_profile
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.stages import Stage

Providers = Mapping[str, type[StageProvider]]


@pytest.fixture
def plan(
    make_settings: SettingsFactory, providers: Providers, fixture_dataset: SyntheticFixtureDataset
) -> ResolvedPlan:
    settings = make_settings(*plan_overrides(detect=FAKE, track=FAKE, depth=FAKE))
    return resolve_plan(settings, providers, dataset=fixture_dataset.name)


def _profile(plan: ResolvedPlan, depth_params: str | None = None) -> ErrorProfile:
    params = plan_params_hashes(plan.stages)
    lift = entry(
        Stage.LIFT,
        "backproject",
        "centroid_rel",
        0.1,
        conditions={"params_hash": params[Stage.LIFT]},
    )
    depth = entry(
        Stage.DEPTH, FAKE, "depth_rel", 0.1, conditions={"params_hash": params[Stage.DEPTH]}
    )
    if depth_params is not None:
        depth = replace(depth, conditions={"params_hash": depth_params})
    return profile_of(lift, depth)


def test_a_current_profile_passes(plan: ResolvedPlan) -> None:
    require_current_profile(plan, _profile(plan))


def test_a_stale_entry_names_its_stage(plan: ResolvedPlan) -> None:
    with pytest.raises(ConfigError, match=f"stale for depth/{FAKE} "):
        require_current_profile(plan, _profile(plan, depth_params="stale"))


def test_a_missing_profile_is_refused_when_a_provider_reads_one(plan: ResolvedPlan) -> None:
    with pytest.raises(ConfigError, match="no error profile is loaded"):
        require_current_profile(plan, None)


def _cli_profile(tmp_path: Path, plan: ResolvedPlan) -> list[str]:
    path = tmp_path / "stale_profile.json"
    _profile(plan, depth_params="stale").save_json(path)
    return ["--set", f"paths.error_profile={path}"]


def test_strict_profile_exits_two_and_writes_no_run(
    cli: CliRunner, tmp_path: Path, plan: ResolvedPlan
) -> None:
    result = cli(*FAKE_PLAN, *_cli_profile(tmp_path, plan), *RUN, "--strict-profile")
    assert result.code == 2
    assert f"depth/{FAKE}" in result.err
    assert "run: " not in result.out


def test_without_the_flag_a_stale_profile_only_warns(
    cli: CliRunner, tmp_path: Path, plan: ResolvedPlan
) -> None:
    result = cli(*FAKE_PLAN, *_cli_profile(tmp_path, plan), *RUN)
    assert result.code == 0, result.err
    assert "is ignored and its sigma is unknown" in result.err
