"""Optional estimates added for depth models that also give the camera: DepthMap.focal_px and
CameraModel.gravity_sigma_deg. Unknown by default, round-trip, absent in older files."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
from records import camera_model, depth_map

from physverify.schema.errors import SchemaError
from physverify.schema.perception import CameraModel, DepthMap


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


def test_gravity_sigma_defaults_to_unknown_and_round_trips(tmp_path: Path) -> None:
    assert math.isnan(camera_model().gravity_sigma_deg)
    record = replace(camera_model(), gravity_sigma_deg=1.5)
    record.save_json(tmp_path / "camera.json")
    assert CameraModel.load_json(tmp_path / "camera.json") == record


def test_cameras_written_before_the_gravity_sigma_still_load() -> None:
    data = camera_model().to_dict()
    del data["gravity_sigma_deg"]
    assert math.isnan(CameraModel.from_dict(data).gravity_sigma_deg)


def test_rejects_a_negative_gravity_sigma() -> None:
    with pytest.raises(SchemaError, match="gravity_sigma_deg"):
        replace(camera_model(), gravity_sigma_deg=-1.0)
