from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import records

from physverify.schema.clip import FrameSampling
from physverify.schema.contacts import ContactRecord
from physverify.schema.errors import SchemaError, SchemaVersionError
from physverify.schema.io import Record, read_jsonl, stable_hash, values_equal, write_jsonl
from physverify.schema.states import ObjectStateTable

REQUIRED = {
    "ObjectStateTable", "ContactRecord", "ContactTable", "ScaleEstimate", "ErrorProfile", "IdMap",
    "FrameSampling", "ClipManifest", "SceneSpec", "PerturbationRecord", "CheckResult",
    "Detections", "MaskTrack", "DepthMap", "CameraModel", "StageArtifactMeta",
}  # fmt: skip


def test_every_required_record_has_a_sample() -> None:
    assert set(records.SAMPLES) >= REQUIRED


@pytest.mark.parametrize("name", sorted(records.SAMPLES))
@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_round_trip_is_exact(name: str, suffix: str, tmp_path: Path) -> None:
    record = records.SAMPLES[name]()
    path = tmp_path / f"{name}{suffix}"
    if suffix == ".json":
        record.save_json(path)
        loaded = type(record).load_json(path)
    else:
        record.save_npz(path)
        loaded = type(record).load_npz(path)
    assert loaded == record
    assert stable_hash(loaded.to_dict()) == stable_hash(record.to_dict())


def test_nan_survives_and_compares_equal(tmp_path: Path) -> None:
    record = records.contact()
    record.save_json(tmp_path / "c.json")
    loaded = ContactRecord.load_json(tmp_path / "c.json")
    assert loaded.point_distances is not None and math.isnan(loaded.point_distances[2])
    assert math.isnan(loaded.confidence)
    assert loaded.points.shape == (4, 3)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    items = [records.manifest(), records.manifest()]
    write_jsonl(tmp_path / "m.jsonl", items)
    assert read_jsonl(type(items[0]), tmp_path / "m.jsonl") == items


def test_values_equal_is_dtype_and_shape_strict() -> None:
    assert values_equal(np.array([np.nan]), np.array([np.nan]))
    assert not values_equal(np.array([1.0]), np.array([1.0], dtype=np.float32))
    assert not values_equal(np.zeros((2, 1)), np.zeros((1, 2)))
    assert not values_equal((1, 2), [1, 2])


def test_wrong_shape_is_rejected() -> None:
    with pytest.raises(SchemaError, match="points"):
        ContactRecord(
            object_a="a",
            object_b="b",
            t_start=0.0,
            t_end=1.0,
            kind=records.ContactKind.IMPACT,
            normal=np.array([0.0, 0.0, 1.0]),
            points=np.zeros((2, 2)),
            manifold=records.ManifoldKind.MEASURED,
        )


def test_inconsistent_dimensions_are_rejected() -> None:
    table = records.states()
    import dataclasses

    with pytest.raises(SchemaError, match="elsewhere"):
        dataclasses.replace(table, visible=np.ones((3, 3), dtype=bool))


def test_wrong_dtype_is_rejected() -> None:
    import dataclasses

    with pytest.raises(SchemaError, match="dtype"):
        dataclasses.replace(records.states(), t=np.zeros(3, dtype=np.float32))


def test_too_many_manifold_points_are_rejected() -> None:
    import dataclasses

    with pytest.raises(SchemaError, match="manifold"):
        dataclasses.replace(
            records.contact(), points=np.zeros((9, 3)), point_normals=None, point_distances=None
        )


def test_mismatched_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    records.sampling().save_json(path)
    raw = json.loads(path.read_text())
    raw["schema_version"] = "0.9"
    path.write_text(json.dumps(raw))
    with pytest.raises(SchemaVersionError):
        FrameSampling.load_json(path)


def test_wrong_record_type_is_rejected(tmp_path: Path) -> None:
    records.sampling().save_json(tmp_path / "s.json")
    with pytest.raises(SchemaError, match="expected a ClipManifest"):
        records.manifest().load_json(tmp_path / "s.json")


def test_pickled_arrays_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "states.npz"
    records.states().save_npz(path)
    with np.load(path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays["labels"] = np.array([{"evil": 1}, None], dtype=object)
    np.savez(path, **arrays)
    with pytest.raises(SchemaError, match="pickled"):
        ObjectStateTable.load_npz(path)


def test_object_arrays_are_never_written(tmp_path: Path) -> None:
    table = records.states()
    object.__setattr__(table, "labels", np.array(["a", None], dtype=object))
    with pytest.raises(SchemaError, match="object arrays"):
        table.save_npz(tmp_path / "x.npz")


def test_records_are_frozen_and_unhashable() -> None:
    record: Record = records.scale()
    with pytest.raises(AttributeError):
        record.sigma = 1.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(record)
