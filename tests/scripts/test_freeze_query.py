"""scripts/h200/freeze_query.py: the checks of the freeze and of the sealed final looks
(plan-congelado F7, Juicio J1, J3, J4, J6, AC-8, AC-18)."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.sealed_ledger import LedgerEntry, append_entry
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.datasets.kubric_fresh import fresh_settings
from physverify.freeze.errors import QueryError
from physverify.freeze.fresh_render import VERSION_ENV, plan_lines, render_overrides
from physverify.freeze.ledger import ledger_entries, look_state, v1_state
from physverify.freeze.ledger import ledger_line as summary_line
from physverify.freeze.manifest import (
    DIFF_PATHS,
    Git,
    ManifestInputs,
    artifact_problems,
    field,
    manifest,
)
from physverify.freeze.selection import J4_RULE, candidate, seed_of, select
from physverify.schema.clip import Split
from physverify.synth.generate import load_plan, plan_subset, save_plan
from physverify.synth.settings import dataset_root, load_dataset_settings
from physverify.synth.store import DatasetStore

HELPER = Path(__file__).resolve().parents[2] / "scripts" / "h200" / "freeze_query.py"
HEAD = "c" * 40
V1 = "a" * 40


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def script() -> Iterator[ModuleType]:
    spec = importlib.util.spec_from_file_location("freeze_query", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


def test_the_most_n_met_wins_then_auroc_cv_at_full_precision(
    script: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    iteration_factory: Callable[..., Path],
) -> None:
    """J4."""
    dirs = [
        iteration_factory(tmp_path, "it0-baseline", 1, 0.99),
        iteration_factory(tmp_path, "it1-fixes", 3, 0.5861),
        iteration_factory(tmp_path, "it2-state", 3, 0.5862),
        iteration_factory(tmp_path, "it4-selfrel", 2, 0.9),
    ]
    found = select([candidate(d) for d in dirs])
    assert found["chosen"] == "it2-state"
    assert found["rule"] == J4_RULE
    assert [c["auroc_cv"] for c in found["candidates"]] == [0.99, 0.5861, 0.5862, 0.9]
    assert found["candidates"][2]["commit"] == HEAD
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(found))
    for key, value in (("id", "it2-state"), ("dir", str(dirs[2])), ("commit", HEAD)):
        assert script.main(["chosen", str(selection), key]) == 0
        assert capsys.readouterr().out == f"{value}\n"


def test_a_full_tie_goes_to_the_later_iteration_and_a_null_auroc_ranks_last(
    tmp_path: Path, iteration_factory: Callable[..., Path]
) -> None:
    tied = [
        iteration_factory(tmp_path, "it4-selfrel", 2, 0.586),
        iteration_factory(tmp_path, "it1-fixes", 2, 0.586),
        iteration_factory(tmp_path, "it2-state", 2, None),
    ]
    assert select([candidate(d) for d in tied])["chosen"] == "it4-selfrel"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"device": "cuda"}, "only CPU iterations"),
        ({"steps": (1, 2, 3, 4)}, r"steps \[5\]"),
    ],
)
def test_a_cuda_or_incomplete_iteration_is_refused(
    tmp_path: Path,
    change: Mapping[str, Any],
    message: str,
    iteration_factory: Callable[..., Path],
) -> None:
    directory = iteration_factory(tmp_path, "it3-arms", 9, 0.99, **change)
    with pytest.raises(QueryError, match=message):
        candidate(directory)


def test_a_file_that_moved_after_the_iteration_is_refused(
    tmp_path: Path, iteration_factory: Callable[..., Path]
) -> None:
    directory = iteration_factory(tmp_path, "it2-state", 3, 0.6)
    (directory / "calibration" / "stage_sigma.json").write_bytes(b"refit")
    with pytest.raises(QueryError, match="STAGE_SIGMA_SHA"):
        candidate(directory)


def test_two_directories_of_one_iteration_are_refused(
    tmp_path: Path, iteration_factory: Callable[..., Path]
) -> None:
    dirs = [
        iteration_factory(tmp_path / "a", "it2-state", 3, 0.6),
        iteration_factory(tmp_path / "b", "it2-x", 1, 0.6),
    ]
    with pytest.raises(QueryError, match="two directories"):
        select([candidate(d) for d in dirs])


def test_the_seed_is_the_preregistered_formula() -> None:
    """Amendment 2 section 2, by hand."""
    calibration, exclusions = "1" * 64, "2" * 64
    digest = hashlib.sha256((HEAD + calibration + exclusions).encode("utf-8")).hexdigest()
    assert seed_of(HEAD, calibration, exclusions) == int(digest[:8], 16)
    with pytest.raises(QueryError, match="40 lowercase"):
        seed_of(HEAD.upper(), calibration, exclusions)
    with pytest.raises(QueryError, match="64 lowercase"):
        seed_of(HEAD, calibration[:-1], exclusions)


def test_a_bad_input_exits_2_with_the_message(
    script: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert script.main(["seed", "abc", "1" * 64, "2" * 64]) == 2
    assert capsys.readouterr().err.startswith("freeze_query: the commit must be")


def _settings(config_dir: Path, environ: Mapping[str, str], tmp_path: Path) -> Settings:
    env = {
        **environ,
        "PHYSVERIFY__PATHS__CALIBRATION_DIR": str(tmp_path / "freeze" / "calibration"),
        "PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data"),
    }
    return load_settings(config_dir, environ=env, profile="precise", device="cpu")


def _frozen_files(settings: Settings) -> dict[str, bytes]:
    directory = settings.paths.calibration_dir
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "calibration.json": b"calibration of it2-state",
        "exclusions.json": b"record",
        "stage_errors.precise.json": b"profile",
        "stage_sigma.json": b"sigma",
        "abstention.json": b"abstention",
    }
    for name, data in files.items():
        (directory / name).write_bytes(data)
    return files


class FakeGit:
    def __init__(self, changed: Mapping[str, Sequence[str]]) -> None:
        self.changed = changed
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: Sequence[str]) -> str:
        self.calls.append(tuple(args))
        if args[0] == "rev-parse":
            return ("t" * 40 if args[1] == "HEAD:configs" else HEAD) + "\n"
        assert list(args[:4]) == ["diff", "--name-only", args[2], "HEAD"]
        return "".join(f"{p}\n" for p in self.changed.get(args[2], ()))


def _inputs(
    tmp_path: Path, iteration_factory: Callable[..., Path], **change: Any
) -> ManifestInputs:
    directory = iteration_factory(tmp_path / "iterations", "it2-state", 3, 0.6, commit="b" * 40)
    selection = select([candidate(directory)])
    v1 = tmp_path / "v1-calibration.json"
    v1.write_bytes(b"v1")
    fields = {
        "selection": selection,
        "record_sha": "r" * 64,
        "p2_commit": "d" * 40,
        "v1_commit": V1,
        "v1_calibration": v1,
    }
    return ManifestInputs(**{**fields, **change})


def _manifest(inputs: ManifestInputs, settings: Settings, git: Git) -> dict[str, Any]:
    return manifest(inputs, settings, git, record=lambda: "r" * 64, now=datetime.now)


def test_the_manifest_records_the_freeze(
    tmp_path: Path,
    config_dir: Path,
    environ: dict[str, str],
    iteration_factory: Callable[..., Path],
) -> None:
    """AC-8, AC-18: commit, configs tree, artifacts, record, seed, selection, J3/J13 diffs, v1."""
    settings = _settings(config_dir, environ, tmp_path)
    files = _frozen_files(settings)
    git = FakeGit({"d" * 40: ["configs/extraction.toml"]})
    data = _manifest(_inputs(tmp_path, iteration_factory), settings, git)
    assert (data["commit"], data["configs_tree"]) == (HEAD, "t" * 40)
    assert (data["profile"], data["device"]) == ("precise", "cpu")
    artifacts = data["artifacts"]
    assert artifacts["calibration"]["sha256"] == _sha(files["calibration.json"])
    assert artifacts["exclusions"]["sha256"] == _sha(files["exclusions.json"])
    assert artifacts["error_profile"]["sha256"] == _sha(files["stage_errors.precise.json"])
    assert artifacts["stage_sigma"]["sha256"] == _sha(files["stage_sigma.json"])
    assert artifacts["abstention"]["sha256"] == _sha(files["abstention.json"])
    assert artifacts["thresholds"]["sha256"] is None
    calibration, exclusions = _sha(files["calibration.json"]), _sha(files["exclusions.json"])
    assert data["seed"] == seed_of(HEAD, calibration, exclusions)
    assert data["record_sha"] == "r" * 64
    assert data["iteration"] == {
        "id": "it2-state",
        "dir": data["selection"]["candidates"][0]["dir"],
        "commit": "b" * 40,
    }
    assert data["diffs"]["j3"]["empty"] and data["diffs"]["j3"]["from"] == "b" * 40
    assert data["diffs"]["j13"]["changed"] == ["configs/extraction.toml"]
    assert data["v1"] == {"commit": V1, "calibration_sha256": _sha(b"v1")}
    diffs = [c for c in git.calls if c[0] == "diff"]
    assert {c[5:] for c in diffs} == {DIFF_PATHS["j3"], DIFF_PATHS["j13"]}
    assert "src/physverify/checks" in DIFF_PATHS["j3"]
    json.dumps(data)


def test_a_j3_change_record_mismatch_or_other_calibration_refuses_the_manifest(
    tmp_path: Path,
    config_dir: Path,
    environ: dict[str, str],
    iteration_factory: Callable[..., Path],
) -> None:
    """J3: a change under checks, physics, extraction or verification means refit, not adopt;
    J12: the record sha the script parsed must be the record's."""
    settings = _settings(config_dir, environ, tmp_path)
    _frozen_files(settings)
    inputs = _inputs(tmp_path, iteration_factory)
    moved = FakeGit({"b" * 40: ["src/physverify/checks/r1.py"]})
    with pytest.raises(QueryError, match="J3"):
        _manifest(inputs, settings, moved)
    with pytest.raises(QueryError, match="exclusions show printed"):
        _manifest(dataclasses.replace(inputs, record_sha="e" * 64), settings, FakeGit({}))
    (settings.paths.calibration_dir / "calibration.json").write_bytes(b"another fit")
    with pytest.raises(QueryError, match="not the full calib fit"):
        _manifest(inputs, settings, FakeGit({}))


def test_the_frozen_files_must_stay_as_frozen(
    tmp_path: Path,
    config_dir: Path,
    environ: dict[str, str],
    iteration_factory: Callable[..., Path],
) -> None:
    settings = _settings(config_dir, environ, tmp_path)
    _frozen_files(settings)
    inputs = _inputs(tmp_path, iteration_factory)
    data = json.loads(json.dumps(_manifest(inputs, settings, FakeGit({}))))
    assert artifact_problems(data, settings) == []
    settings.paths.thresholds.write_text("{}")
    (settings.paths.calibration_dir / "abstention.json").write_bytes(b"refit")
    problems = artifact_problems(data, settings)
    assert len(problems) == 2 and any("abstention" in p for p in problems)
    elsewhere = _settings(config_dir, environ, tmp_path / "other")
    assert all("frozen" in p for p in artifact_problems(data, elsewhere))


def test_a_field_is_one_value() -> None:
    data = {"seed": 7, "v1": {"commit": V1}, "artifacts": {}}
    assert field(data, "seed") == "7"
    assert field(data, "v1.commit") == V1
    with pytest.raises(QueryError):
        field(data, "artifacts")
    with pytest.raises(QueryError):
        field(data, "v1.missing")


def _render(
    config_dir: Path, environ: Mapping[str, str], tmp_path: Path, name: str
) -> tuple[dict[str, str], Settings]:
    base = _settings(config_dir, environ, tmp_path)
    overrides = render_overrides(fresh_settings(base), name, 12345)
    env = {**environ, **overrides, "PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")}
    return overrides, load_settings(config_dir, environ=env, profile="precise", device="cpu")


def test_the_render_settings_are_the_fresh_sets(
    tmp_path: Path, config_dir: Path, environ: dict[str, str]
) -> None:
    """Section 2.2: new seed, no calib scene, the set's version, scene count and backgrounds."""
    _, fresh = _render(config_dir, environ, tmp_path, "kubric_fresh")
    _, ood = _render(config_dir, environ, tmp_path, "kubric_fresh_ood")
    _, blind = _render(config_dir, environ, tmp_path, "kubric_fresh_blind")
    section = fresh_settings(fresh)
    ds, ood_ds = load_dataset_settings(fresh), load_dataset_settings(ood)
    assert (ds.version, ds.seed, ds.calib_fraction) == (section.version, 12345, 0.0)
    assert ds.subsets["full"].scenes_per_family == section.scenes_per_family
    assert ood_ds.version == section.ood_version
    assert ood_ds.subsets["full"].scenes_per_family == section.ood_scenes_per_family
    pools = load_dataset_settings(fresh).hdri_pools
    assert set(ood_ds.effective_hdri_ids()) == set(pools[section.ood_hdri_pool])
    assert ood_ds.content_hash() != ds.content_hash()
    assert load_dataset_settings(blind) == ds
    with pytest.raises(QueryError, match="not a fresh Kubric set"):
        render_overrides(section, "kubric_synthetic", 1)


def _stored_plan(settings: Settings) -> DatasetStore:
    ds = load_dataset_settings(settings)
    plan = plan_subset(ds, PLUGIN_KINDS["families"](), PLUGIN_KINDS["perturbations"](), "full")
    store = DatasetStore(dataset_root(settings, ds))
    save_plan(store, plan)
    return store


def test_the_plan_check_accepts_only_the_render_of_the_seed(
    tmp_path: Path, config_dir: Path, environ: dict[str, str]
) -> None:
    """J1: a plan of another seed (or none) under the fresh version stops before any look."""
    overrides, settings = _render(config_dir, environ, tmp_path, "kubric_fresh")
    version = overrides[VERSION_ENV]
    with pytest.raises(QueryError, match="render first"):
        plan_lines(settings, version, 12345)
    store = _stored_plan(settings)
    (line,) = plan_lines(settings, version, 12345)
    section = fresh_settings(settings)
    assert f"{section.scenes_per_family * 5} scenes" in line and "0 calib" in line
    with pytest.raises(QueryError, match="not v2-fresh 1"):
        plan_lines(settings, version, 1)
    other = {**overrides, "PHYSVERIFY__DATASET__SEED": "1"}
    reseeded = load_settings(
        config_dir,
        environ={**environ, **other, "PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")},
        profile="precise",
        device="cpu",
    )
    with pytest.raises(QueryError, match="is not the render"):
        plan_lines(reseeded, version, 1)
    plan = query_plan(store)
    calib = dataclasses.replace(plan.scenes[0], split=Split.CALIB)
    store.plan_path.write_text(
        json.dumps(dataclasses.replace(plan, scenes=(calib, *plan.scenes[1:])).to_json())
    )
    with pytest.raises(QueryError, match="1 calib"):
        plan_lines(settings, version, 12345)


def query_plan(store: DatasetStore) -> Any:
    plan = load_plan(store)
    assert plan is not None
    return plan


def ledger_line(sets: Sequence[str], commit: str, run_dir: Path | None) -> LedgerEntry:
    return LedgerEntry(
        utc="2026-10-30T00:00:00+00:00",
        commit=commit,
        command="calibrate evaluate",
        argv=("physverify",),
        sets=tuple(sets),
        sealed=tuple(sets),
        datasets=(),
        artifacts={},
        config_hash="h",
        freeze="f",
        repeat=False,
        run_dir=None if run_dir is None else str(run_dir),
        user="u",
        host="h",
    )


def test_a_set_is_looked_at_once_with_one_retry_when_nothing_was_written(tmp_path: Path) -> None:
    """J6."""
    spec = ("kubric_fresh",)
    other = [ledger_line(["kubric_synthetic:eval"], HEAD, tmp_path)]
    assert look_state(other, spec, retry=False) == "first"
    with pytest.raises(QueryError, match="exactly one earlier look"):
        look_state(other, spec, retry=True)
    run = tmp_path / "run"
    run.mkdir()
    looked = [*other, ledger_line(spec, HEAD, run)]
    with pytest.raises(QueryError, match="already has 1 ledger lines"):
        look_state(looked, spec, retry=False)
    assert look_state(looked, spec, retry=True) == "retry"
    (run / "evaluation.md").write_text("# partial")
    with pytest.raises(QueryError, match="no retry"):
        look_state(looked, spec, retry=True)
    twice = [*looked, ledger_line(spec, HEAD, tmp_path / "again")]
    with pytest.raises(QueryError, match="found 2"):
        look_state(twice, spec, retry=True)


def test_the_v1_arm_follows_the_look_once(tmp_path: Path) -> None:
    frozen = {"commit": HEAD, "v1": {"commit": V1}}
    spec = ("kubric_fresh",)
    with pytest.raises(QueryError, match="no look of the freeze"):
        v1_state([ledger_line(spec, V1, None)], frozen, spec)
    look = [ledger_line(spec, HEAD, tmp_path)]
    assert v1_state(look, frozen, spec) == "v1arm"
    with pytest.raises(QueryError, match="already looked"):
        v1_state([*look, ledger_line(spec, V1, None)], frozen, spec)


def test_a_missing_ledger_is_refused(
    tmp_path: Path,
    config_dir: Path,
    environ: dict[str, str],
) -> None:
    """AC-6: the ledger is created by hand; without it no look starts."""
    settings = _settings(config_dir, environ, tmp_path)
    home = Path(environ["HOME"])
    with pytest.raises(QueryError, match="no sealed ledger"):
        ledger_entries(settings, home)
    home.mkdir(parents=True)
    ledger = home / "sealed-ledger.txt"
    ledger.touch()
    assert ledger_entries(settings, home) == ()
    append_entry(ledger, ledger_line(["kubric_fresh"], HEAD, None))
    assert len(ledger_entries(settings, home)) == 1
    assert summary_line(settings, home) == f"ledger {ledger}: 1 lines"
