"""it2 J1: HDRI backgrounds as dataset settings, the preregistered pools outside OpenPano and the
clean calib subset rendered with them."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from physverify.config.loader import load_settings
from physverify.core.cache_keys import params_hash
from physverify.kubric_worker.build_scene import hdri_for
from physverify.synth.generate import plan_subset
from physverify.synth.hdri import halves, outside_ids, panorama_name, training_names
from physverify.synth.plan import NONE
from physverify.synth.render import render_job
from physverify.synth.settings import (
    LEGACY_HDRI_IDS,
    NON_CONTENT_KEYS,
    DatasetSettings,
    load_dataset_settings,
)
from physverify.synth.store import DatasetStore

NEW_KEYS = ("hdri_ids", "hdri_pool", "hdri_pools")
DEV = "openpano_ood_dev"
RESERVED = "openpano_ood_reserved"


def old_content_hash(ds: DatasetSettings) -> str:
    """The content hash of the code before ``hdri_ids`` existed."""
    dumped = ds.model_dump(mode="json")
    skipped = ("render", "subsets", "version", *NEW_KEYS)
    return params_hash({k: v for k, v in dumped.items() if k not in skipped})


@pytest.mark.parametrize("profile", ["fast", "precise"])
def test_the_legacy_background_keeps_every_planned_hash(config_dir: Path, profile: str) -> None:
    ds = load_dataset_settings(load_settings(config_dir, environ={}, profile=profile))
    assert ds.effective_hdri_ids() == LEGACY_HDRI_IDS
    assert ds.content_hash() == old_content_hash(ds)
    assert set(NON_CONTENT_KEYS) >= {"hdri_pool", "hdri_pools"}


def test_other_backgrounds_are_other_content_and_a_pool_is_its_list(ds: DatasetSettings) -> None:
    pool = ds.hdri_pools[DEV]
    by_pool = ds.model_copy(update={"hdri_pool": DEV})
    by_ids = DatasetSettings.model_validate({**ds.model_dump(), "hdri_ids": pool})
    assert by_pool.effective_hdri_ids() == pool
    assert by_pool.content_hash() == by_ids.content_hash() != ds.content_hash()


@pytest.mark.parametrize(
    "update", [{"hdri_pool": "missing"}, {"hdri_pool": DEV, "hdri_ids": ["shudu_lake"]}]
)
def test_an_unknown_pool_or_both_keys_are_refused(
    ds: DatasetSettings, update: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        DatasetSettings.model_validate({**ds.model_dump(), **update})


def test_the_preregistered_pools_are_two_disjoint_halves_ordered_by_sha256(
    ds: DatasetSettings,
) -> None:
    dev, reserved = ds.hdri_pools[DEV], ds.hdri_pools[RESERVED]
    assert len(dev) == len(reserved) == 6
    assert not set(dev) & set(reserved) and LEGACY_HDRI_IDS[0] not in dev + reserved
    digests = [hashlib.sha256(i.encode()).hexdigest() for i in dev + reserved]
    assert digests == sorted(digests)


def test_the_split_rule_on_a_small_listing() -> None:
    listing = [
        "openpano/ 0",
        "openpano/panoramas/train/abandoned_parking_16k.jpg 10",
        "openpano/panoramas/val/bergen_8k.jpg 10",
        "openpano/panoramas/test/011_hdrmaps_com_free.jpg 10",
        "openpano/panoramas/.DS_Store 6",
    ]
    assert (
        panorama_name("openpano/panoramas/train/abandoned_parking_16k.jpg") == "abandoned_parking"
    )
    seen = training_names(listing)
    assert seen == {"abandoned_parking", "bergen", "011_hdrmaps_com_free"}
    ids = outside_ids(["abandoned_parking", "bergen", "lot_01", "orbita", "shudu_lake"], seen)
    assert sorted(ids) == ["lot_01", "orbita", "shudu_lake"]
    dev, reserved = halves(ids)
    assert (len(dev), len(reserved)) == (1, 2) and dev + reserved == ids


def test_calib_clean_is_the_clean_clips_of_the_full_calib_scenes(
    ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    full = plan_subset(ds, families, perturbations, "full")
    clean = plan_subset(ds, families, perturbations, "calib_clean")
    expected = {
        v.clip_id for v in full.variants if v.perturbation == NONE and v.split.value == "calib"
    }
    assert {v.clip_id for v in clean.variants} == expected and len(expected) == 100
    assert {s.scene_id: s.index for s in clean.scenes} == {
        s.scene_id: s.index for s in full.scenes if s.split.value == "calib"
    }


def test_the_render_job_carries_the_effective_backgrounds(
    ds: DatasetSettings, families: Any, perturbations: Any, tmp_path: Path
) -> None:
    dev = ds.model_copy(update={"hdri_pool": DEV})
    plan = plan_subset(dev, families, perturbations, "calib_clean")
    clip_id = plan.variants[0].clip_id

    class Runner:
        def container_path(self, path: Path) -> str:
            return str(path)

    job = render_job(clip_id, plan, DatasetStore(tmp_path), Runner(), dev)  # type: ignore[arg-type]
    assert job["hdri_ids"] == list(dev.hdri_pools[DEV])


def test_the_worker_picks_the_background_by_seed() -> None:
    ids = ("a", "b", "c")
    assert [hdri_for(seed, ids) for seed in (0, 1, 5)] == ["a", "b", "c"]
    assert hdri_for(7, LEGACY_HDRI_IDS) == "abandoned_parking"
    with pytest.raises(ValueError, match="hdri_ids"):
        hdri_for(0, ())
