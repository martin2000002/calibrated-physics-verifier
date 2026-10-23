"""freeze.json, written by physverify.freeze through scripts/h200/freeze_query.py and read back
by the final looks: its schema, version, key paths and bytes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.freeze.diffs import DIFF_PATHS
from physverify.freeze.manifest import FREEZE_SCHEMA, FREEZE_VERSION, ManifestInputs, manifest
from physverify.freeze.selection import J4_RULE

REPO = Path(__file__).resolve().parents[2]
REFACTOR_BASE = "9eb5888"
J3_PACKAGES = ("checks", "physics", "extraction", "verification")
DIFF_SOURCES = {
    "j3": tuple(f"src/physverify/{package}/" for package in J3_PACKAGES),
    "j13": ("src/physverify/extraction/", "configs/extraction.toml"),
}
HEAD = "c" * 40
FREEZE_SHA256 = "9a3741804997cdb56c3c1282fabc5ba579086afc7eb1dd77aab8effc0521c23e"
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


def _git(args: Sequence[str]) -> str:
    return "" if args[0] == "diff" else HEAD + "\n"


def _key_paths(data: Mapping[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in data.items():
        paths.append(prefix + key)
        if isinstance(value, Mapping) and key != "selection":
            paths += _key_paths(value, f"{prefix}{key}.")
    return paths


def _frozen_settings(tmp_path: Path, config_dir: Path, environ: dict[str, str]) -> Settings:
    env = {**environ, "PHYSVERIFY__PATHS__CALIBRATION_DIR": str(tmp_path / "calibration")}
    settings = load_settings(config_dir, environ=env, profile="precise", device="cpu")
    settings.paths.calibration_dir.mkdir(parents=True)
    for name in FROZEN:
        (settings.paths.calibration_dir / name).write_bytes(name.encode())
    (tmp_path / "v1.json").write_bytes(b"v1")
    return settings


def _manifest(
    settings: Settings, tmp_path: Path, selection: Mapping[str, Any], **commits: str
) -> dict[str, Any]:
    inputs = ManifestInputs(
        selection=selection,
        record_sha="r" * 64,
        p2_commit=commits.get("p2_commit", HEAD),
        v1_commit=commits.get("v1_commit", HEAD),
        v1_calibration=tmp_path / "v1.json",
    )
    created = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    return manifest(inputs, settings, _git, record=lambda: "r" * 64, now=lambda: created)


def _chosen(tmp_path: Path, name: str) -> dict[str, str]:
    return {
        "id": name,
        "dir": str(tmp_path / name),
        "commit": HEAD,
        "calibration_sha256": hashlib.sha256(b"calibration.json").hexdigest(),
    }


def test_the_freeze_manifest_keeps_its_schema_and_key_paths(
    tmp_path: Path, config_dir: Path, environ: dict[str, str]
) -> None:
    settings = _frozen_settings(tmp_path, config_dir, environ)
    chosen = {**_chosen(tmp_path, "it"), "dir": str(tmp_path)}
    data = _manifest(settings, tmp_path, {"chosen": "it", "candidates": [chosen]})
    assert (data["schema"], data["version"]) == ("physverify/freeze", "1.0.0")
    assert (FREEZE_SCHEMA, FREEZE_VERSION) == ("physverify/freeze", "1.0.0")
    assert _key_paths(data) == list(MANIFEST_FIELDS)


def test_the_freeze_manifest_keeps_its_bytes(
    tmp_path: Path, config_dir: Path, environ: dict[str, str]
) -> None:
    """The bytes run_freeze.sh writes, with the run's paths masked, pinned before WP8 moved them."""
    settings = _frozen_settings(tmp_path, config_dir, environ)
    selection = {
        "rule": J4_RULE,
        "chosen": "it6-chosen",
        "candidates": [_chosen(tmp_path, "it6-chosen")],
    }
    data = _manifest(settings, tmp_path, selection, p2_commit="d" * 40, v1_commit="a" * 40)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    text = text.replace(str(tmp_path), "<tmp>").replace(str(settings.repo_root), "<repo>")
    assert hashlib.sha256(text.encode()).hexdigest() == FREEZE_SHA256


def _git_lines(*args: str) -> list[str]:
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"git history unavailable: {result.stderr.strip()}")
    return result.stdout.splitlines()


def _destinations(sources: tuple[str, ...]) -> set[str]:
    renames = {}
    for line in _git_lines("diff", "-M", "--name-status", REFACTOR_BASE, "HEAD"):
        status, *paths = line.split("\t")
        if status.startswith("R"):
            renames[paths[0]] = paths[1]
    files = _git_lines("ls-tree", "-r", "--name-only", REFACTOR_BASE, "--", *sources)
    return {renames.get(path, path) for path in files}


@pytest.mark.parametrize("name", sorted(DIFF_SOURCES))
def test_the_diff_paths_cover_exactly_where_the_pre_refactor_paths_moved(name: str) -> None:
    """Each diff covers the moved files, as whole directories where they stayed inside one."""
    sources = DIFF_SOURCES[name]
    directories = [source for source in sources if source.endswith("/")]
    expected = {
        next((d.rstrip("/") for d in directories if path.startswith(d)), path)
        for path in _destinations(sources)
    }
    assert set(DIFF_PATHS[name]) == expected
