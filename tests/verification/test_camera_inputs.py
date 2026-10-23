"""it4 judgement J6: the checks receive the camera the states were lifted with.

Perceived: the CAMERA output of the same plan that produced KINEMATICS, loaded from the cache
(never computed: the lock refuses heavy stages); with LIFT ground truth and CAMERA perceived there
is no such camera. Oracle: the dataset's camera at the native resolution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic_clip import SyntheticFixtureDataset
from test_inputs import loader_for

from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.cache import StageCache
from physverify.schema.checks import InputLevel
from physverify.schema.perception import CameraModel, IntrinsicsSource
from physverify.schema.stages import STAGE_ORDER, Stage, StageArtifactMeta
from physverify.verification.inputs import (
    OracleSource,
    PerceivedSource,
    lift_camera,
    native_sampling,
)


@pytest.fixture
def settings(config_dir: Path, environ: dict[str, str]) -> Settings:
    return load_settings(config_dir, environ=environ, profile=None, overrides=())


def test_the_oracle_camera_is_the_dataset_camera_at_the_native_size() -> None:
    dataset = SyntheticFixtureDataset()
    (clip,) = dataset.clips()
    (item,) = OracleSource(dataset).load([clip])
    assert item.inputs is not None and item.inputs.camera is not None
    store = dataset.ground_truth(clip)
    assert store is not None
    truth = store.camera(native_sampling(clip))
    assert item.inputs.camera.sampling.size == (clip.width, clip.height)
    assert np.array_equal(item.inputs.camera.K, truth.K)


def test_the_ground_truth_chain_gives_the_ground_truth_camera(settings: Settings) -> None:
    dataset = SyntheticFixtureDataset()
    loader, lock = loader_for(settings, STAGE_ORDER, dataset.name)
    (item,) = PerceivedSource(dataset, loader, InputLevel.MIXED).load(dataset.clips())
    assert lock.acquisitions == 0
    assert item.inputs is not None and item.inputs.camera is not None
    assert item.inputs.camera.source is IntrinsicsSource.GROUND_TRUTH


def lift_upstream_camera(settings: Settings, dataset: str, clip_id: str) -> CameraModel:
    """The CAMERA output whose key the cached LIFT entry records as its upstream."""
    cache = StageCache(settings.paths.cache_dir)
    metas = [StageArtifactMeta.load_json(path) for path in cache.metas(dataset, clip_id)]
    (lift,) = [m for m in metas if m.stage is Stage.LIFT]
    key = lift.upstream[Stage.CAMERA]
    (camera_meta,) = [m for m in metas if m.stage is Stage.CAMERA and m.cache_key == key]
    output = cache.load(dataset, clip_id, Stage.CAMERA, camera_meta.provider, key)
    assert isinstance(output, CameraModel)
    return output


def test_a_perceived_camera_is_the_one_lift_used(settings: Settings) -> None:
    dataset = SyntheticFixtureDataset()
    (clip,) = dataset.clips()
    truth = (Stage.DETECT, Stage.TRACK, Stage.DEPTH)
    loader, lock = loader_for(settings, truth, dataset.name)
    (item,) = PerceivedSource(dataset, loader, InputLevel.MIXED).load([clip])
    assert lock.acquisitions == 0
    assert item.inputs is not None and item.inputs.camera is not None
    assert item.inputs.provenance[Stage.CAMERA.value] != GROUND_TRUTH
    used = lift_upstream_camera(settings, dataset.name, clip.clip_id)
    assert np.array_equal(item.inputs.camera.K, used.K)
    assert item.inputs.camera.source is used.source


def test_states_lifted_from_truth_have_no_perceived_camera() -> None:
    dataset = SyntheticFixtureDataset()
    (clip,) = dataset.clips()
    store = dataset.ground_truth(clip)
    assert store is not None
    outputs = {Stage.CAMERA: store.camera(native_sampling(clip))}
    names = dict.fromkeys(STAGE_ORDER, "perceived")
    assert lift_camera(outputs, names) is outputs[Stage.CAMERA]
    assert lift_camera(outputs, {**names, Stage.LIFT: GROUND_TRUTH}) is None
    both = {**names, Stage.LIFT: GROUND_TRUTH, Stage.CAMERA: GROUND_TRUTH}
    assert lift_camera(outputs, both) is outputs[Stage.CAMERA]
