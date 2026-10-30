"""DepthMap.focal_px: optional, positive, round-trips through JSON and NPZ, absent in old files."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
from records import depth_map

from physverify.schema.errors import SchemaError
from physverify.schema.perception import DepthMap


def test_focal_defaults_to_unknown() -> None:
    record = depth_map()
    assert record.focal_px is None
    assert math.isnan(record.focal_sigma_px)


@pytest.mark.parametrize("suffix", ["json", "npz"])
def test_focal_round_trips(tmp_path: Path, suffix: str) -> None:
    record = replace(depth_map(), focal_px=412.5, focal_sigma_px=20.0)
    path = tmp_path / f"depth.{suffix}"
    getattr(record, f"save_{suffix}")(path)
    loaded = getattr(DepthMap, f"load_{suffix}")(path)
    assert loaded == record
    assert loaded.focal_px == pytest.approx(412.5)


def test_records_written_before_the_field_still_load() -> None:
    data = depth_map().to_dict()
    del data["focal_px"], data["focal_sigma_px"]
    loaded = DepthMap.from_dict(data)
    assert loaded.focal_px is None
    assert math.isnan(loaded.focal_sigma_px)


@pytest.mark.parametrize("focal", [0.0, -3.0, math.nan, math.inf])
def test_rejects_non_positive_focal(focal: float) -> None:
    with pytest.raises(SchemaError, match="focal_px"):
        replace(depth_map(), focal_px=focal)


def test_rejects_negative_focal_sigma() -> None:
    with pytest.raises(SchemaError, match="focal_sigma_px"):
        replace(depth_map(), focal_px=100.0, focal_sigma_px=-1.0)
