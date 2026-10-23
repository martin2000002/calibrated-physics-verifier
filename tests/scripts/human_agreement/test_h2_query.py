"""scripts/h200/agreement_query.py: the read-only answers run_human_agreement.sh asks of the
checkout."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from physverify.human_agreement.prereg import (
    LoggedComparison,
    Preregistration,
    log_comparison,
    write_preregistration,
)

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "h200"
RECORD = Preregistration(
    created_utc="2026-10-20T00:00:00Z",
    commit="abc",
    agreement_config_sha256="c" * 64,
    subset_sha256={},
    queries_sha256=None,
    calibration_artifact_sha256="a" * 64,
    rule="calibrated",
    rule_label="calibrated/cal-1",
    profile="precise",
    device="cpu",
    dev_run="runs/dev",
    dev_subset="h2_dev",
    test_subset="h2_test",
    n_dev=3,
    dev_majority=0,
    dev_cut=0.25,
    dev_tuned_cut=None,
    kubric_threshold=None,
)


@pytest.fixture(scope="module")
def query() -> ModuleType:
    """Loaded like the script runs: its directory first on the path (it reuses run_query)."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "agreement_query", SCRIPTS / "agreement_query.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


def test_a_field_of_the_preregistration_and_null_as_empty(query, tmp_path: Path) -> None:
    path = tmp_path / "preregistration.json"
    write_preregistration(path, RECORD)
    assert query.prereg_field(path, "rule") == ["calibrated"]
    assert query.prereg_field(path, "dev_cut") == ["0.25"]
    assert query.prereg_field(path, "calibration_gate") == [""]
    with pytest.raises(query.QueryError, match="no field"):
        query.prereg_field(path, "nonsense")


def test_a_missing_preregistration_exits_2(query, tmp_path: Path, capsys) -> None:
    assert query.main(["prereg", str(tmp_path / "none.json"), "rule"]) == 2
    assert "agreement freeze" in capsys.readouterr().err


def test_tested_counts_the_log_of_the_configured_preregistration(query, settings) -> None:
    assert query.tested(settings) == ["0"]
    path = settings.paths.data_dir / "agreement" / "preregistration.json"
    log_comparison(path, LoggedComparison(prereg_sha256="p", run="r", compared_utc="t"))
    assert query.tested(settings) == ["1"]
    (log,) = query.log_path(settings)
    assert Path(log).read_text(encoding="utf-8").count("\n") == 1


def test_gaps_and_the_first_clip_ready_to_preflight(query, world) -> None:
    world.keep_all()
    with pytest.raises(query.QueryError, match="both a video and an object query"):
        query.ready(world.settings, "h2_test")
    box = next(r for r in world.dataset.rows("h2_test") if r.caption == world.BOX)
    video = world.dataset.video_file(box)
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    assert query.ready(world.settings, "h2_test") == [box.clip_id]
    rows = len(world.dataset.rows("h2_test"))
    lines = query.gaps(world.settings, "h2_test")
    assert lines[:2] == [
        "- no object query: 0 rows (0 with video)",
        f"- no video in the source: {rows - 1} rows",
    ]
    assert "- generator v2:gen3: no_query 0, no_video 2" in lines
