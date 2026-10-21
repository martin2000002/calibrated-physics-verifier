"""scripts/h200/freeze_query.py: the checks of the freeze and of the sealed final looks
(plan-congelado F7, Juicio J1, J3, J4, J6, AC-8, AC-18)."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import shlex
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from physverify.calibration.sealed import LedgerEntry, append_entry
from physverify.cli.context import PLUGIN_KINDS
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.datasets.kubric_fresh import fresh_settings
from physverify.schema.clip import Split
from physverify.synth.generate import load_plan, plan_subset, save_plan
from physverify.synth.settings import dataset_root, load_dataset_settings
from physverify.synth.store import DatasetStore

HELPER = Path(__file__).resolve().parents[2] / "scripts" / "h200" / "freeze_query.py"
HEAD = "c" * 40
V1 = "a" * 40
FILES = {
    "DEVSCORE_SHA": "devscore.json",
    "CALIBRATION_SHA": "calibration.json",
    "PROFILE_SHA": "calibration/stage_errors.precise.json",
    "STAGE_SIGMA_SHA": "calibration/stage_sigma.json",
}


@pytest.fixture(scope="module")
def query() -> Iterator[ModuleType]:
    """Registered while loading: its dataclasses resolve their annotations through sys.modules."""
    spec = importlib.util.spec_from_file_location("freeze_query", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iteration(
    root: Path,
    name: str,
    n_met: int,
    auroc: float | None,
    *,
    device: str = "cpu",
    steps: Sequence[int] = (1, 2, 3, 4, 5),
    commit: str = HEAD,
) -> Path:
    """A run_iteration.sh directory: state.env written with printf %q, steps.tsv, the files."""
    directory = root / f"{name} {device}"
    (directory / "calibration").mkdir(parents=True)
    devscore = {"commit": commit, "selection": {"n_met": n_met, "tiebreak_auroc_cv": auroc}}
    contents = {
        "devscore.json": json.dumps(devscore).encode(),
        "calibration.json": f"calibration of {name}".encode(),
        "calibration/stage_errors.precise.json": b"profile",
        "calibration/stage_sigma.json": b"sigma",
    }
    for relative, data in contents.items():
        (directory / relative).write_bytes(data)
    state = [("CHAIN_DEVICE", device), ("CHAIN_ITER_ID", name), ("CHAIN_DEVICE", device)]
    state += [(key, _sha(contents[relative])) for key, relative in FILES.items()]
    (directory / "state.env").write_text("".join(f"{k}={shlex.quote(v)}\n" for k, v in state))
    rows = [f"{n}\tstep\trunning\t0\t-" for n in steps] + [f"{n}\tstep\t0\t9\t-" for n in steps]
    (directory / "steps.tsv").write_text("\n".join(rows) + "\n")
    return directory


def test_the_most_n_met_wins_then_auroc_cv_at_full_precision(
    query: ModuleType, tmp_path: Path
) -> None:
    """J4."""
    dirs = [
        iteration(tmp_path, "it0-baseline", 1, 0.99),
        iteration(tmp_path, "it1-fixes", 3, 0.5861),
        iteration(tmp_path, "it2-state", 3, 0.5862),
        iteration(tmp_path, "it4-selfrel", 2, 0.9),
    ]
    found = query.select([query.candidate(d) for d in dirs])
    assert found["chosen"] == "it2-state"
    assert found["rule"] == query.J4_RULE
    assert [c["auroc_cv"] for c in found["candidates"]] == [0.99, 0.5861, 0.5862, 0.9]
    assert found["candidates"][2]["commit"] == HEAD


def test_a_full_tie_goes_to_the_later_iteration_and_a_null_auroc_ranks_last(
    query: ModuleType, tmp_path: Path
) -> None:
    tied = [
        iteration(tmp_path, "it4-selfrel", 2, 0.586),
        iteration(tmp_path, "it1-fixes", 2, 0.586),
        iteration(tmp_path, "it2-state", 2, None),
    ]
    assert query.select([query.candidate(d) for d in tied])["chosen"] == "it4-selfrel"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"device": "cuda"}, "only CPU iterations"),
        ({"steps": (1, 2, 3, 4)}, r"steps \[5\]"),
    ],
)
def test_a_cuda_or_incomplete_iteration_is_refused(
    query: ModuleType, tmp_path: Path, change: Mapping[str, Any], message: str
) -> None:
    directory = iteration(tmp_path, "it3-arms", 9, 0.99, **change)
    with pytest.raises(query.QueryError, match=message):
        query.candidate(directory)


def test_a_file_that_moved_after_the_iteration_is_refused(
    query: ModuleType, tmp_path: Path
) -> None:
    directory = iteration(tmp_path, "it2-state", 3, 0.6)
    (directory / "calibration" / "stage_sigma.json").write_bytes(b"refit")
    with pytest.raises(query.QueryError, match="STAGE_SIGMA_SHA"):
        query.candidate(directory)


def test_two_directories_of_one_iteration_are_refused(query: ModuleType, tmp_path: Path) -> None:
    dirs = [
        iteration(tmp_path / "a", "it2-state", 3, 0.6),
        iteration(tmp_path / "b", "it2-x", 1, 0.6),
    ]
    with pytest.raises(query.QueryError, match="two directories"):
        query.select([query.candidate(d) for d in dirs])


def test_the_seed_is_the_preregistered_formula(query: ModuleType) -> None:
    """Amendment 2 section 2, by hand."""
    calibration, exclusions = "1" * 64, "2" * 64
    digest = hashlib.sha256((HEAD + calibration + exclusions).encode("utf-8")).hexdigest()
    assert query.seed_of(HEAD, calibration, exclusions) == int(digest[:8], 16)
    with pytest.raises(query.QueryError, match="40 lowercase"):
        query.seed_of(HEAD.upper(), calibration, exclusions)
    with pytest.raises(query.QueryError, match="64 lowercase"):
        query.seed_of(HEAD, calibration[:-1], exclusions)


def test_a_bad_input_exits_2_with_the_message(
    query: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert query.main(["seed", "abc", "1" * 64, "2" * 64]) == 2
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


def _inputs(query: ModuleType, tmp_path: Path, **change: Any) -> Any:
    directory = iteration(tmp_path / "iterations", "it2-state", 3, 0.6, commit="b" * 40)
    selection = query.select([query.candidate(directory)])
    v1 = tmp_path / "v1-calibration.json"
    v1.write_bytes(b"v1")
    fields = {
        "selection": selection,
        "record_sha": "r" * 64,
        "p2_commit": "d" * 40,
        "v1_commit": V1,
        "v1_calibration": v1,
    }
    return query.ManifestInputs(**{**fields, **change})


def test_the_manifest_records_the_freeze(
    query: ModuleType,
    tmp_path: Path,
    config_dir: Path,
    environ: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-8, AC-18: commit, configs tree, artifacts, record, seed, selection, J3/J13 diffs, v1."""
    settings = _settings(config_dir, environ, tmp_path)
    files = _frozen_files(settings)
    monkeypatch.setattr(query, "record_sha", lambda _: "r" * 64)
    git = FakeGit({"d" * 40: ["configs/extraction.toml"]})
    data = query.manifest(_inputs(query, tmp_path), settings, git)
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
    assert data["seed"] == query.seed_of(HEAD, calibration, exclusions)
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
    assert {c[5:] for c in diffs} == {query.DIFF_PATHS["j3"], query.DIFF_PATHS["j13"]}
    assert "src/physverify/checks" in query.DIFF_PATHS["j3"]
    json.dumps(data)


def test_a_j3_change_record_mismatch_or_other_calibration_refuses_the_manifest(
    query: ModuleType,
    tmp_path: Path,
    config_dir: Path,
    environ: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """J3: a change under checks, physics, extraction or verification means refit, not adopt;
    J12: the record sha the script parsed must be the record's."""
    settings = _settings(config_dir, environ, tmp_path)
    _frozen_files(settings)
    monkeypatch.setattr(query, "record_sha", lambda _: "r" * 64)
    inputs = _inputs(query, tmp_path)
    moved = FakeGit({"b" * 40: ["src/physverify/checks/r1.py"]})
    with pytest.raises(query.QueryError, match="J3"):
        query.manifest(inputs, settings, moved)
    with pytest.raises(query.QueryError, match="exclusions show printed"):
        query.manifest(dataclasses.replace(inputs, record_sha="e" * 64), settings, FakeGit({}))
    (settings.paths.calibration_dir / "calibration.json").write_bytes(b"another fit")
    with pytest.raises(query.QueryError, match="not the full calib fit"):
        query.manifest(inputs, settings, FakeGit({}))


def test_the_frozen_files_must_stay_as_frozen(
    query: ModuleType,
    tmp_path: Path,
    config_dir: Path,
    environ: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(config_dir, environ, tmp_path)
    _frozen_files(settings)
    monkeypatch.setattr(query, "record_sha", lambda _: "r" * 64)
    data = json.loads(json.dumps(query.manifest(_inputs(query, tmp_path), settings, FakeGit({}))))
    assert query.artifact_problems(data, settings) == []
    settings.paths.thresholds.write_text("{}")
    (settings.paths.calibration_dir / "abstention.json").write_bytes(b"refit")
    problems = query.artifact_problems(data, settings)
    assert len(problems) == 2 and any("abstention" in p for p in problems)
    elsewhere = _settings(config_dir, environ, tmp_path / "other")
    assert all("frozen" in p for p in query.artifact_problems(data, elsewhere))


def test_a_field_is_one_value(query: ModuleType) -> None:
    data = {"seed": 7, "v1": {"commit": V1}, "artifacts": {}}
    assert query.field(data, "seed") == "7"
    assert query.field(data, "v1.commit") == V1
    with pytest.raises(query.QueryError):
        query.field(data, "artifacts")
    with pytest.raises(query.QueryError):
        query.field(data, "v1.missing")


def _render(
    query: ModuleType, config_dir: Path, environ: Mapping[str, str], tmp_path: Path, name: str
) -> tuple[dict[str, str], Settings]:
    base = _settings(config_dir, environ, tmp_path)
    overrides = query.render_overrides(fresh_settings(base), name, 12345)
    env = {**environ, **overrides, "PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")}
    return overrides, load_settings(config_dir, environ=env, profile="precise", device="cpu")


def test_the_render_settings_are_the_fresh_sets(
    query: ModuleType, tmp_path: Path, config_dir: Path, environ: dict[str, str]
) -> None:
    """Section 2.2: new seed, no calib scene, the set's version, scene count and backgrounds."""
    _, fresh = _render(query, config_dir, environ, tmp_path, "kubric_fresh")
    _, ood = _render(query, config_dir, environ, tmp_path, "kubric_fresh_ood")
    _, blind = _render(query, config_dir, environ, tmp_path, "kubric_fresh_blind")
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
    with pytest.raises(query.QueryError, match="not a fresh Kubric set"):
        query.render_overrides(section, "kubric_synthetic", 1)


def _stored_plan(settings: Settings) -> DatasetStore:
    ds = load_dataset_settings(settings)
    plan = plan_subset(ds, PLUGIN_KINDS["families"](), PLUGIN_KINDS["perturbations"](), "full")
    store = DatasetStore(dataset_root(settings, ds))
    save_plan(store, plan)
    return store


def test_the_plan_check_accepts_only_the_render_of_the_seed(
    query: ModuleType, tmp_path: Path, config_dir: Path, environ: dict[str, str]
) -> None:
    """J1: a plan of another seed (or none) under the fresh version stops before any look."""
    overrides, settings = _render(query, config_dir, environ, tmp_path, "kubric_fresh")
    version = overrides[query.VERSION_ENV]
    with pytest.raises(query.QueryError, match="render first"):
        query.plan_lines(settings, version, 12345)
    store = _stored_plan(settings)
    (line,) = query.plan_lines(settings, version, 12345)
    section = fresh_settings(settings)
    assert f"{section.scenes_per_family * 5} scenes" in line and "0 calib" in line
    with pytest.raises(query.QueryError, match="not v2-fresh 1"):
        query.plan_lines(settings, version, 1)
    other = {**overrides, "PHYSVERIFY__DATASET__SEED": "1"}
    reseeded = load_settings(
        config_dir,
        environ={**environ, **other, "PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")},
        profile="precise",
        device="cpu",
    )
    with pytest.raises(query.QueryError, match="is not the render"):
        query.plan_lines(reseeded, version, 1)
    plan = query_plan(store)
    calib = dataclasses.replace(plan.scenes[0], split=Split.CALIB)
    store.plan_path.write_text(
        json.dumps(dataclasses.replace(plan, scenes=(calib, *plan.scenes[1:])).to_json())
    )
    with pytest.raises(query.QueryError, match="1 calib"):
        query.plan_lines(settings, version, 12345)


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


def test_a_set_is_looked_at_once_with_one_retry_when_nothing_was_written(
    query: ModuleType, tmp_path: Path
) -> None:
    """J6."""
    spec = ("kubric_fresh",)
    other = [ledger_line(["kubric_synthetic:eval"], HEAD, tmp_path)]
    assert query.look_state(other, spec, retry=False) == "first"
    with pytest.raises(query.QueryError, match="exactly one earlier look"):
        query.look_state(other, spec, retry=True)
    run = tmp_path / "run"
    run.mkdir()
    looked = [*other, ledger_line(spec, HEAD, run)]
    with pytest.raises(query.QueryError, match="already has 1 ledger lines"):
        query.look_state(looked, spec, retry=False)
    assert query.look_state(looked, spec, retry=True) == "retry"
    (run / "evaluation.md").write_text("# partial")
    with pytest.raises(query.QueryError, match="no retry"):
        query.look_state(looked, spec, retry=True)
    twice = [*looked, ledger_line(spec, HEAD, tmp_path / "again")]
    with pytest.raises(query.QueryError, match="found 2"):
        query.look_state(twice, spec, retry=True)


def test_the_v1_arm_follows_the_look_once(query: ModuleType, tmp_path: Path) -> None:
    frozen = {"commit": HEAD, "v1": {"commit": V1}}
    spec = ("kubric_fresh",)
    with pytest.raises(query.QueryError, match="no look of the freeze"):
        query.v1_state([ledger_line(spec, V1, None)], frozen, spec)
    look = [ledger_line(spec, HEAD, tmp_path)]
    assert query.v1_state(look, frozen, spec) == "v1arm"
    with pytest.raises(query.QueryError, match="already looked"):
        query.v1_state([*look, ledger_line(spec, V1, None)], frozen, spec)


def test_a_missing_ledger_is_refused(
    query: ModuleType,
    tmp_path: Path,
    config_dir: Path,
    environ: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-6: the ledger is created by hand; without it no look starts."""
    settings = _settings(config_dir, environ, tmp_path)
    home = Path(environ["HOME"])
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(query.QueryError, match="no sealed ledger"):
        query.ledger_entries(settings)
    home.mkdir(parents=True)
    ledger = home / "sealed-ledger.txt"
    ledger.touch()
    assert query.ledger_entries(settings) == ()
    append_entry(ledger, ledger_line(["kubric_fresh"], HEAD, None))
    assert len(query.ledger_entries(settings)) == 1
