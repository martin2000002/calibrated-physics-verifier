from __future__ import annotations

from pathlib import Path

import pytest

from physverify.config.loader import load_settings
from physverify.config.models import PathsSettings, SizePriorTable
from physverify.core.errors import ConfigError

BASE = """
[runtime]
threads = 2
device = "cpu"

[demo]
level = "file"
kept = true

[demo.profiles.precise]
level = "profile"
"""


def _write(directory: Path, name: str, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text, encoding="utf-8")


def _load(config: Path, **kwargs: object) -> tuple[str, int, str]:
    settings = load_settings(config, environ=kwargs.pop("environ", {}), **kwargs)  # type: ignore[arg-type]
    return settings.sections["demo"]["level"], settings.runtime.threads, settings.runtime.device


def test_precedence_level_by_level(tmp_path: Path) -> None:
    config = tmp_path / "configs"
    _write(config, "empty.toml", "")
    assert load_settings(config, environ={}).runtime.threads == 4
    _write(config, "base.toml", BASE)
    assert _load(config) == ("file", 2, "cpu")
    assert _load(config, profile="precise") == ("profile", 2, "cpu")
    _write(config, "local.toml", '[demo]\nlevel = "local"\n[runtime]\nthreads = 3\n')
    assert _load(config, profile="precise") == ("local", 3, "cpu")
    env = {"PHYSVERIFY__DEMO__LEVEL": "env", "PHYSVERIFY__RUNTIME__DEVICE": "cuda:1"}
    assert _load(config, environ=env) == ("env", 3, "cuda:1")
    assert _load(config, environ=env, device="cuda") == ("env", 3, "cuda")
    overrides = ("demo.level=set", "runtime.device=cpu", "runtime.threads=7")
    assert _load(config, environ=env, device="cuda", overrides=overrides) == ("set", 7, "cpu")
    settings = load_settings(config, environ=env, overrides=overrides)
    assert settings.sections["demo"]["kept"] is True
    assert any(source.endswith("local.toml") for source in settings.sources)
    assert "--set demo.level=set" in settings.sources


def test_repeated_top_level_table_raises(tmp_path: Path) -> None:
    _write(tmp_path, "a.toml", "[demo]\nx = 1\n")
    _write(tmp_path, "b.toml", "[demo]\ny = 2\n")
    with pytest.raises(ConfigError, match="both"):
        load_settings(tmp_path, environ={})


def test_local_files_are_not_globbed(tmp_path: Path) -> None:
    _write(tmp_path, "default.toml", "[runtime]\nthreads = 2\n")
    _write(tmp_path, "local.example.toml", "[runtime]\nthreads = 9\n")
    assert load_settings(tmp_path, environ={}).runtime.threads == 2


def test_tables_are_read_from_subdirectories(tmp_path: Path) -> None:
    _write(tmp_path, "default.toml", "[runtime]\nthreads = 2\n")
    _write(tmp_path / "studies", "demo.toml", '[demo]\nlevel = "nested"\n')
    settings = load_settings(tmp_path, environ={})
    assert settings.sections["demo"]["level"] == "nested"
    assert str(tmp_path.resolve() / "studies" / "demo.toml") in settings.sources


def test_local_files_in_subdirectories_are_read(tmp_path: Path) -> None:
    _write(tmp_path / "studies", "local.toml", '[demo]\nlevel = "nested"\n')
    assert load_settings(tmp_path, environ={}).sections["demo"]["level"] == "nested"


def test_a_table_repeated_across_subdirectories_names_both_files(tmp_path: Path) -> None:
    _write(tmp_path / "datasets", "a.toml", "[demo]\nx = 1\n")
    _write(tmp_path / "studies", "b.toml", "[demo]\ny = 2\n")
    with pytest.raises(ConfigError, match=r"both datasets/a\.toml and studies/b\.toml"):
        load_settings(tmp_path, environ={})


@pytest.mark.parametrize(
    "bad",
    [{"PHYSVERIFY__RUNTIME__DEVICE": "tpu"}, {"PHYSVERIFY__NOKEY": "1"}],
)
def test_invalid_values_raise(tmp_path: Path, bad: dict[str, str]) -> None:
    _write(tmp_path, "default.toml", "")
    with pytest.raises(ConfigError):
        load_settings(tmp_path, environ=bad)


def test_unknown_profile_and_bad_set_raise(tmp_path: Path) -> None:
    _write(tmp_path, "default.toml", "")
    with pytest.raises(ConfigError, match="profile"):
        load_settings(tmp_path, environ={}, profile="nope")
    with pytest.raises(ConfigError, match="--set"):
        load_settings(tmp_path, environ={}, overrides=("threads=3",))


def test_paths_derive_from_data_dir_and_expand_env(tmp_path: Path) -> None:
    _write(tmp_path / "configs", "default.toml", '[paths]\ndata_dir = "${BIG}/data"\n')
    settings = load_settings(tmp_path / "configs", environ={"BIG": "/mnt/big", "HOME": "/h"})
    assert settings.paths.cache_dir == Path("/mnt/big/data/cache")
    assert settings.paths.error_profile == Path("/mnt/big/data/calibration/stage_errors.fast.json")
    assert settings.paths.hf_home == Path("/h/.cache/huggingface")
    assert settings.repo_root == (tmp_path).resolve()


def test_relative_lock_is_anchored_at_repo_root(tmp_path: Path) -> None:
    _write(tmp_path / "configs", "default.toml", '[runtime]\nheavy_lock = "run/heavy.lock"\n')
    settings = load_settings(tmp_path / "configs", environ={})
    assert settings.runtime.heavy_lock == str(tmp_path.resolve() / "run" / "heavy.lock")


def test_repository_configs(config_dir: Path) -> None:
    fast = load_settings(config_dir, environ={})
    precise = load_settings(config_dir, environ={}, profile="precise")
    assert fast.profile == "fast" and fast.sections["extraction"]["plan"]["track"] == "edgetam"
    assert precise.sections["extraction"]["plan"]["track"] == "sam2_1_hiera_tiny"
    assert (
        precise.sections["extraction"]["plan"]["depth"] == "vda_small"
    )  # the chosen base chain is the CPU one
    assert precise.sections["extraction"]["sampling"]["max_side"] == 960
    assert precise.sections["extraction"]["sampling"]["target_fps"] == {
        "kubric_synthetic": 24, "physics_iq": 30, "pisa_real": 120,
        "kubric_fresh": 24, "kubric_fresh_ood": 24, "kubric_fresh_blind": 24,
        "physics_iq_views": 30, "objectron_static": 30,
    }  # fmt: skip
    assert fast.runtime.docker_image.startswith("kubricdockerhub/kubruntu@sha256:")
    priors = fast.section("size_priors", SizePriorTable)
    assert set(priors.root) >= {"cube", "ball", "cylinder"}
    assert fast.config_hash() != precise.config_hash()


def test_section_validation_error_is_a_config_error(config_dir: Path) -> None:
    settings = load_settings(config_dir, environ={}, overrides=("size_priors.cube.mean_m=-1",))
    with pytest.raises(ConfigError, match="size_priors"):
        settings.section("size_priors", SizePriorTable)


def test_the_error_profile_defaults_to_one_file_per_profile(config_dir: Path) -> None:
    fast = load_settings(config_dir, environ={})
    precise = load_settings(config_dir, environ={}, profile="precise")
    assert fast.paths.error_profile == fast.paths.calibration_dir / "stage_errors.fast.json"
    assert (
        precise.paths.error_profile == precise.paths.calibration_dir / "stage_errors.precise.json"
    )


def test_an_explicit_error_profile_wins_over_the_profile_default(config_dir: Path) -> None:
    by_env = load_settings(
        config_dir, environ={"PHYSVERIFY__PATHS__ERROR_PROFILE": "/x/env.json"}, profile="precise"
    )
    by_set = load_settings(
        config_dir, environ={}, profile="precise", overrides=("paths.error_profile=/x/set.json",)
    )
    assert by_env.paths.error_profile == Path("/x/env.json")
    assert by_set.paths.error_profile == Path("/x/set.json")


def test_direct_validation_without_a_profile_keeps_the_shared_name() -> None:
    paths = PathsSettings.model_validate({"data_dir": "/d"})
    assert paths.error_profile == Path("/d/calibration/stage_errors.json")
