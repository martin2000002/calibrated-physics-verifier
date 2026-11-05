from pathlib import Path

import numpy as np
import pytest

from physverify.core.feature_table import FeatureTable, FeatureTableError


def test_round_trip_keeps_rows_columns_and_meta(tmp_path: Path) -> None:
    table = FeatureTable.from_rows({"b": [3.0, 4.0], "a": [1.0, 2.0]}, ("x", "y"), {"k": "v"})
    path = tmp_path / "t.npz"
    table.save(path)
    loaded = FeatureTable.load(path)
    assert loaded.keys == ("a", "b")
    assert loaded.columns == ("x", "y")
    assert loaded.meta == {"k": "v"}
    np.testing.assert_array_equal(loaded.row("b"), [3.0, 4.0])
    assert loaded.row("missing") is None


def test_rejects_mismatched_shapes_duplicate_keys_and_other_columns() -> None:
    with pytest.raises(FeatureTableError):
        FeatureTable(("a",), ("x", "y"), np.zeros((1, 3)))
    with pytest.raises(FeatureTableError):
        FeatureTable(("a", "a"), ("x",), np.zeros((2, 1)))
    with pytest.raises(FeatureTableError):
        FeatureTable(("a",), ("x",), np.zeros((1, 1))).require_columns(("y",))
