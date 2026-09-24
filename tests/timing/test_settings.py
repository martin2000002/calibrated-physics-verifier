"""``configs/timing.toml`` under every profile, and the validators of the timing settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from physverify.config.models import Settings
from physverify.core.errors import ConfigError, UsageError
from physverify.datasets.local_video import QueryEntry
from physverify.timing.settings import (
    GenerationSettings,
    TargetRunSettings,
    TimingSettings,
    load_timing,
    run_settings,
    with_overrides,
)

SettingsFactory = Callable[..., Settings]

ORDER = ["wan21_generate", "extract_pipeline", "verify_checks", "kubric_render"]
PRECISE_REPETITIONS = {
    "wan21_generate": 3,
    "extract_pipeline": 3,
    "verify_checks": 5,
    "kubric_render": 3,
}


@pytest.fixture
def generation(make_settings: SettingsFactory) -> dict[str, Any]:
    return load_timing(make_settings()).generation.model_dump()


@pytest.mark.parametrize("profile", [None, "fast", "precise"])
def test_timing_toml_validates_in_every_profile(
    make_settings: SettingsFactory, profile: str | None
) -> None:
    timing = load_timing(make_settings(profile=profile))

    assert timing.order == ORDER
    assert timing.valid_profile == "precise"
    assert timing.total_measures == ["generate.b1.per_video_s", "extract.span_s", "verify.wall_s"]
    assert set(timing.targets) == set(ORDER)
    gen = timing.generation
    assert (gen.height, gen.width, gen.num_frames, gen.fps) == (480, 832, 81, 16)
    assert gen.batch_sizes == [1, 4]
    assert len(gen.prompts) == 5


def test_fast_profile_runs_each_target_once(make_settings: SettingsFactory) -> None:
    timing = load_timing(make_settings(profile="fast"))

    assert {name: run.repetitions for name, run in timing.targets.items()} == dict.fromkeys(
        ORDER, 1
    )


def test_precise_profile_keeps_the_committed_repetitions(make_settings: SettingsFactory) -> None:
    timing = load_timing(make_settings(profile="precise"))

    assert {name: run.repetitions for name, run in timing.targets.items()} == PRECISE_REPETITIONS
    assert timing.targets["extract_pipeline"].warmup_scope == "clip"
    assert timing.targets["kubric_render"].max_clips == 3


def test_default_profile_is_fast(make_settings: SettingsFactory) -> None:
    assert load_timing(make_settings()) == load_timing(make_settings(profile="fast"))


def test_prompt_queries_parse_counts(make_settings: SettingsFactory) -> None:
    prompts = {prompt.id: prompt for prompt in load_timing(make_settings()).generation.prompts}

    assert prompts["ball_collision"].queries == [QueryEntry(phrase="ball", expected_count=2)]


@pytest.mark.parametrize(
    "override",
    [
        "timing.generation.height=470",
        "timing.generation.width=830",
        "timing.generation.num_frames=80",
        "timing.bogus=1",
        "timing.targets.verify_checks.bogus=1",
        "timing.targets.verify_checks.repetitions=0",
        "timing.max_iqr_rel=0",
    ],
)
def test_invalid_config_is_a_config_error(make_settings: SettingsFactory, override: str) -> None:
    with pytest.raises(ConfigError, match=r"\[timing\] is invalid"):
        load_timing(make_settings(override))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("revision", "abc123"),
        ("revision", "G" * 40),
        ("sha256", "0" * 63),
        ("sha256", "Z" * 64),
        ("batch_sizes", [4, 8]),
        ("batch_sizes", [1, 1, 4]),
        ("batch_sizes", []),
        ("prompts", []),
        ("subset", "Timing-Wan"),
        ("guidance_scale", 0.0),
    ],
)
def test_generation_validators_reject(generation: dict[str, Any], key: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        GenerationSettings.model_validate({**generation, key: value})


def test_generation_accepts_the_committed_shape(generation: dict[str, Any]) -> None:
    valid = GenerationSettings.model_validate({**generation, "revision": "a" * 40})

    assert (valid.height, valid.width, valid.num_frames) == (480, 832, 81)


def test_prompt_ids_must_be_unique(generation: dict[str, Any]) -> None:
    prompts = generation["prompts"]

    with pytest.raises(ValidationError, match="prompt ids must be unique"):
        GenerationSettings.model_validate({**generation, "prompts": [prompts[0], prompts[0]]})


@pytest.mark.parametrize("key", ["order", "total_measures"])
def test_repeated_entries_are_rejected(make_settings: SettingsFactory, key: str) -> None:
    timing = load_timing(make_settings()).model_dump()
    entries = timing[key]

    with pytest.raises(ValidationError, match=f"{key} has repeated entries"):
        TimingSettings.model_validate({**timing, key: [*entries, entries[0]]})


def test_run_settings_falls_back_to_defaults(make_settings: SettingsFactory) -> None:
    timing = load_timing(make_settings())

    assert run_settings(timing, "not_configured") == TargetRunSettings()
    assert run_settings(timing, "verify_checks") is timing.targets["verify_checks"]


def test_with_overrides_replaces_only_given_values() -> None:
    run = TargetRunSettings(warmup=1, warmup_scope="target", repetitions=5, max_clips=3)

    assert with_overrides(run, warmup=None, repetitions=None) == run
    assert with_overrides(run, warmup=0, repetitions=None) == run.model_copy(update={"warmup": 0})
    assert with_overrides(run, warmup=None, repetitions=2).repetitions == 2


@pytest.mark.parametrize(("warmup", "repetitions"), [(-1, None), (None, 0)])
def test_with_overrides_rejects_invalid_values(warmup: int | None, repetitions: int | None) -> None:
    with pytest.raises(UsageError, match="--warmup/--repetitions"):
        with_overrides(TargetRunSettings(), warmup=warmup, repetitions=repetitions)
