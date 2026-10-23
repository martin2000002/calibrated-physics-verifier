"""freeze.json, written by scripts/h200/freeze_query.py and read back by the final looks: its
schema, version and key paths."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from physverify.config.loader import load_settings

HELPER = Path(__file__).resolve().parents[2] / "scripts" / "h200" / "freeze_query.py"
HEAD = "c" * 40
MANIFEST_FIELDS = (
    "schema",
    "version",
    "created_utc",
    "commit",
    "configs_tree",
    "profile",
    "device",
    "calibration_dir",
    "artifacts",
    "artifacts.exclusions",
    "artifacts.exclusions.path",
    "artifacts.exclusions.sha256",
    "artifacts.calibration",
    "artifacts.calibration.path",
    "artifacts.calibration.sha256",
    "artifacts.thresholds",
    "artifacts.thresholds.path",
    "artifacts.thresholds.sha256",
    "artifacts.error_profile",
    "artifacts.error_profile.path",
    "artifacts.error_profile.sha256",
    "artifacts.stage_sigma",
    "artifacts.stage_sigma.path",
    "artifacts.stage_sigma.sha256",
    "artifacts.abstention",
    "artifacts.abstention.path",
    "artifacts.abstention.sha256",
    "record_sha",
    "seed",
    "selection",
    "iteration",
    "iteration.id",
    "iteration.dir",
    "iteration.commit",
    "diffs",
    "diffs.j3",
    "diffs.j3.from",
    "diffs.j3.to",
    "diffs.j3.paths",
    "diffs.j3.changed",
    "diffs.j3.empty",
    "diffs.j13",
    "diffs.j13.from",
    "diffs.j13.to",
    "diffs.j13.paths",
    "diffs.j13.changed",
    "diffs.j13.empty",
    "v1",
    "v1.commit",
    "v1.calibration_sha256",
)
FROZEN = (
    "calibration.json",
    "exclusions.json",
    "stage_errors.precise.json",
    "stage_sigma.json",
    "abstention.json",
)


@pytest.fixture
def query() -> Iterator[ModuleType]:
    spec = importlib.util.spec_from_file_location("freeze_query", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


def _git(args: Sequence[str]) -> str:
    return "" if args[0] == "diff" else HEAD + "\n"


def _key_paths(data: Mapping[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in data.items():
        paths.append(prefix + key)
        if isinstance(value, Mapping) and key != "selection":
            paths += _key_paths(value, f"{prefix}{key}.")
    return paths


def test_the_freeze_manifest_keeps_its_schema_and_key_paths(
    query: ModuleType,
    tmp_path: Path,
    config_dir: Path,
    environ: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = {**environ, "PHYSVERIFY__PATHS__CALIBRATION_DIR": str(tmp_path / "calibration")}
    settings = load_settings(config_dir, environ=env, profile="precise", device="cpu")
    settings.paths.calibration_dir.mkdir(parents=True)
    for name in FROZEN:
        (settings.paths.calibration_dir / name).write_bytes(name.encode())
    monkeypatch.setattr(query, "record_sha", lambda _: "r" * 64)
    chosen = {
        "id": "it",
        "dir": str(tmp_path),
        "commit": HEAD,
        "calibration_sha256": hashlib.sha256(b"calibration.json").hexdigest(),
    }
    (tmp_path / "v1.json").write_bytes(b"v1")
    inputs = query.ManifestInputs(
        selection={"chosen": "it", "candidates": [chosen]},
        record_sha="r" * 64,
        p2_commit=HEAD,
        v1_commit=HEAD,
        v1_calibration=tmp_path / "v1.json",
    )
    data = query.manifest(inputs, settings, _git)
    assert (data["schema"], data["version"]) == ("physverify/freeze", "1.0.0")
    assert _key_paths(data) == list(MANIFEST_FIELDS)
