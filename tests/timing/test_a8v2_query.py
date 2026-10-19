"""scripts/h200/a8v2_query.py: the read-only checks the H200 scripts ask of the checkout."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from check_rows import UNCACHED, write_check_rows

from physverify.schema.clip import ClipManifest
from physverify.synth.migrate import COMPLETED_KEY, RERENDER_KEY

HELPER = Path(__file__).resolve().parents[2] / "scripts" / "h200" / "a8v2_query.py"
ALLOWED = ({"physics_iq": 54, "pisa_real": 360}, {"physics_iq": 54, "pisa_real": 40})


@pytest.fixture(scope="module")
def query() -> ModuleType:
    spec = importlib.util.spec_from_file_location("a8v2_query", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _evaluation(tmp_path: Path, pisa: int, primary: object) -> Path:
    sets = [{"name": "physics_iq", "n_clips": 54}, {"name": "pisa_real", "n_clips": pisa}]
    path = _write(tmp_path / "evaluation.json", {"real_sets": sets, "h1_primary": primary})
    path.with_suffix(".md").write_text("# Evaluation\n\n## H1 primary (scenes s000)\n\n| x |\n")
    return path


class _Source:
    def __init__(self, clips: Sequence[ClipManifest]) -> None:
        self._clips = clips

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return self._clips


def test_clip_ids_filter_by_split(
    query: ModuleType, clip_factory: Callable[..., ClipManifest]
) -> None:
    source = _Source([clip_factory("b"), clip_factory("a")])
    assert query.clip_ids(source, "full", None) == ["a", "b"]
    assert query.clip_ids(source, "full", "eval") == ["a", "b"]
    assert query.clip_ids(source, "full", "calib") == []


def test_rerendered_needs_a_completed_manifest(query: ModuleType, tmp_path: Path) -> None:
    missing = tmp_path / "pair-collision.json"
    assert query.rerendered(missing, missing_ok=True) == []
    with pytest.raises(query.QueryError, match="run step 0"):
        query.rerendered(missing, missing_ok=False)
    _write(missing, {RERENDER_KEY: ["c2", "c1"]})
    with pytest.raises(query.QueryError, match=COMPLETED_KEY):
        query.rerendered(missing, missing_ok=True)
    _write(missing, {RERENDER_KEY: ["c2", "c1"], COMPLETED_KEY: "2026-10-19T00:00:00Z"})
    assert query.rerendered(missing, missing_ok=False) == ["c2", "c1"]


def test_an_allowed_evaluation_prints_its_real_sets_and_h1_section(
    query: ModuleType, tmp_path: Path
) -> None:
    lines = query.evaluation_lines(_evaluation(tmp_path, 40, {"scenes": []}), ALLOWED)
    assert lines[0] == "Real sets: physics_iq 54, pisa_real 40"
    assert lines[2].startswith("## H1 primary") and lines[-1] == "| x |"


def test_an_evaluation_without_the_primary_view_or_with_another_set_is_refused(
    query: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(query.QueryError, match="rerun from 6"):
        query.evaluation_lines(_evaluation(tmp_path, 40, None), ALLOWED)
    with pytest.raises(query.QueryError, match="not in the allowed sets"):
        query.evaluation_lines(_evaluation(tmp_path, 41, {"scenes": []}), ALLOWED)


def test_the_allowed_sets_are_the_preregistered_ones(
    query: ModuleType, make_settings: Callable[..., Any]
) -> None:
    sets = [dict(s) for s in query._allowed_sets(make_settings())]
    assert sets == [{"physics_iq": 54, "pisa_real": n} for n in (360, 40, 361)]


def test_gravity_keeps_the_calib_lift_rows(query: ModuleType, tmp_path: Path) -> None:
    row = {"stage": "lift", "metric": "gravity_angle", "provider": "p", "n_clips": 3,
           "values": {"gravity_angle_deg": 1.25}}  # fmt: skip
    rows = [{**row, "split": "calib"}, {**row, "split": "eval"}, {**row, "metric": "other"}]
    path = _write(tmp_path / "stages.json", {"profile_split": "calib", "rows": rows})
    lines = query.gravity_lines(path)
    assert lines[-1] == "| p | 3 | gravity_angle_deg 1.25 |"
    assert len(lines) == 5
    _write(path, {"rows": rows[2:]})
    with pytest.raises(query.QueryError, match="no calib row"):
        query.gravity_lines(path)


def test_main_exits_2_with_the_reason(
    query: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert query.main(["gravity", str(tmp_path / "absent.json")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_null_gravity_values_print_as_nan(query: ModuleType, tmp_path: Path) -> None:
    row = {"stage": "lift", "metric": "gravity_angle", "provider": "p", "n_clips": 0,
           "split": "calib", "values": {"gravity_angle_deg": None}}  # fmt: skip
    path = _write(tmp_path / "stages.json", {"rows": [row]})
    assert query.gravity_lines(path)[-1] == "| p | 0 | gravity_angle_deg nan |"


def test_run_errors_passes_a_run_without_errors(query: ModuleType, tmp_path: Path) -> None:
    write_check_rows(tmp_path / "checks.jsonl")
    assert query.run_errors(tmp_path)[0].endswith("1 check rows, none with an error")


@pytest.mark.parametrize("error", [UNCACHED, ""])
def test_run_errors_refuses_any_row_with_an_error(
    query: ModuleType, tmp_path: Path, error: str
) -> None:
    write_check_rows(tmp_path / "checks.jsonl", error=error)
    with pytest.raises(query.QueryError, match="1 check rows .* failed") as raised:
        query.run_errors(tmp_path)
    assert f"clip R1: {error or '(no message)'}" in str(raised.value)


def test_run_errors_refuses_a_run_without_checks(query: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(query.QueryError, match="cannot read"):
        query.run_errors(tmp_path)
    (tmp_path / "checks.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(query.QueryError, match="cannot read"):
        query.run_errors(tmp_path)
