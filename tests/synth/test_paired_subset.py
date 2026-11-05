"""The ``train_h`` recipe: per scene the clean clip, a positive and its decoy at one level shared by
both; a larger subset extends the same scene stream; and the seed audit finds every planned seed
in a manifest that already holds it."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from physverify.schema.clip import Split
from physverify.synth.generate import plan_subset
from physverify.synth.plan import NONE
from physverify.synth.seed_audit import collisions, manifest_seeds, planned_seeds
from physverify.synth.seeds import labelled_seed
from physverify.synth.settings import DatasetSettings

SUBSET = "train_h"


def resized(ds: DatasetSettings, scenes: int) -> DatasetSettings:
    sub = ds.subsets[SUBSET].model_copy(update={"scenes_per_family": scenes})
    return ds.model_copy(update={"subsets": {**ds.subsets, SUBSET: sub}})


def test_each_scene_pairs_a_positive_and_its_decoy_at_one_level(
    ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan = plan_subset(ds, families, perturbations, SUBSET)
    by_scene: dict[str, list[Any]] = {}
    for v in plan.variants:
        by_scene.setdefault(v.scene_id, []).append(v)
    assert len(by_scene) == 2 * ds.subsets[SUBSET].scenes_per_family
    for variants in by_scene.values():
        names = sorted(v.perturbation for v in variants)
        assert names[0] == NONE and len(names) == 3 and names[2] == f"{names[1]}_decoy"
        assert len({v.level for v in variants if v.perturbation != NONE}) == 1
    levels = Counter(v.level for v in plan.variants if v.perturbation != NONE)
    assert set(levels) == {1, 2, 3, 4}


def training(ds: DatasetSettings) -> DatasetSettings:
    """The settings ``_train_h.sh`` exports: no calib scene, every scene split=train."""
    return DatasetSettings.model_validate(
        ds.model_dump() | {"calib_fraction": 0.0, "train_split": True}
    )


def test_train_split_labels_every_scene_train_and_changes_nothing_else(
    ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    """Same scenes, seeds and content hash as the eval-labelled plan; only the split differs."""
    eval_ds = ds.model_copy(update={"calib_fraction": 0.0})
    labelled = plan_subset(eval_ds, families, perturbations, SUBSET)
    trained = plan_subset(training(ds), families, perturbations, SUBSET)
    assert {v.split for v in trained.variants} == {s.split for s in trained.scenes} == {Split.TRAIN}
    assert {v.split for v in labelled.variants} == {Split.EVAL}
    assert [replace_split(v) for v in trained.variants] == [
        replace_split(v) for v in labelled.variants
    ]
    assert training(ds).content_hash() == eval_ds.content_hash() == trained.config_hash


def replace_split(item: Any) -> Any:
    return replace(item, split=Split.EVAL)


def test_train_split_with_calib_scenes_is_refused(ds: DatasetSettings) -> None:
    with pytest.raises(ValidationError, match="calib_fraction = 0"):
        DatasetSettings.model_validate(ds.model_dump() | {"train_split": True})


def test_more_scenes_extend_the_same_stream(
    ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    small = plan_subset(resized(ds, 6), families, perturbations, SUBSET)
    large = plan_subset(resized(ds, 15), families, perturbations, SUBSET)
    assert set(small.variants) <= set(large.variants)


def test_the_labelled_seed_is_the_sha256_prefix() -> None:
    label = "some label"
    assert labelled_seed(label) == int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def test_the_seed_audit_finds_a_planned_seed(
    ds: DatasetSettings, families: Any, perturbations: Any, tmp_path: Path
) -> None:
    small = resized(ds, 3)
    planned = planned_seeds(
        plan_subset(small, families, perturbations, SUBSET), perturbations, small
    )
    shifted = small.model_copy(update={"seed": small.seed + 1})
    other = planned_seeds(
        plan_subset(shifted, families, perturbations, SUBSET), perturbations, shifted
    )
    clean = tmp_path / "clean.jsonl"
    hit = tmp_path / "hit.jsonl"
    row = {"type": "ClipManifest", "data": {"perturbation": {"seed": 0}}}
    clean.write_text(json.dumps(row) + "\n", encoding="utf-8")
    seed = sorted(planned)[1]
    row["data"]["perturbation"]["seed"] = seed  # type: ignore[index]
    hit.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
    assert list(manifest_seeds(hit)) == [seed]
    assert collisions(planned, [clean]) == []
    (found,) = collisions(planned, [clean, hit])
    assert (found.seed, found.clip_id, found.manifest) == (seed, planned[seed], str(hit))
    assert not set(planned) & set(other)
