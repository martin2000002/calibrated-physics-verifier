"""plan-congelado J9: the sealed sets of the freeze sample exactly like their source dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from physverify.cli.context import PLUGIN_KINDS
from physverify.config.loader import known_profiles, load_settings, read_config_files
from physverify.extraction.plan import resolve_plan

SOURCE = {
    "kubric_fresh": "kubric_synthetic",
    "kubric_fresh_ood": "kubric_synthetic",
    "kubric_fresh_blind": "kubric_synthetic",
    "physics_iq_views": "physics_iq",
    "objectron_static": "physics_iq",
}


def _profiles(repo_root: Path) -> list[str]:
    base, _ = read_config_files(repo_root / "configs")
    return sorted(known_profiles(base))


@pytest.mark.parametrize(("dataset", "source"), sorted(SOURCE.items()))
def test_target_fps_follows_source(repo_root: Path, dataset: str, source: str) -> None:
    providers = PLUGIN_KINDS["providers"]()
    profiles = _profiles(repo_root)
    assert {"fast", "precise"} <= set(profiles)
    for profile in profiles:
        settings = load_settings(repo_root / "configs", environ={}, profile=profile)
        mine = resolve_plan(settings, providers, dataset=dataset).sampling
        theirs = resolve_plan(settings, providers, dataset=source).sampling
        assert mine == theirs, profile
        assert mine.target_fps is not None, profile
