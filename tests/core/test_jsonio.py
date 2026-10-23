"""JSON files read with one error shape, as the caller's error type."""

from __future__ import annotations

from pathlib import Path

import pytest

from physverify.core.errors import PhysverifyError, UsageError
from physverify.core.jsonio import loads, read_json, read_model
from physverify.core.models import FrozenModel


class Point(FrozenModel):
    x: int


def test_read_json_returns_the_value(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text('{"x": [1, 2]}', encoding="utf-8")
    assert read_json(path) == {"x": [1, 2]}


def test_a_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(PhysverifyError, match=r"cannot read .*absent\.json"):
        read_json(tmp_path / "absent.json")


def test_invalid_json_raises_the_callers_error(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match=r"b\.json is not valid JSON"):
        loads("{", tmp_path / "b.json", UsageError)


def test_read_model_validates(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text('{"x": 3}', encoding="utf-8")
    assert read_model(Point, path) == Point(x=3)
    path.write_text('{"x": "three"}', encoding="utf-8")
    with pytest.raises(PhysverifyError, match="is not a valid Point"):
        read_model(Point, path)
