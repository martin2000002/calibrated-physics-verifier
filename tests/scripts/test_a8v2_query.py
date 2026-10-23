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


def test_main_exits_2_with_the_reason(
    query: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert query.main(["run-errors", str(tmp_path / "absent")]) == 2
    assert "cannot read" in capsys.readouterr().err


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
