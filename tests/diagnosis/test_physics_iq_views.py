"""``physics_iq_views``: the other views of the ``[physics_iq]`` clips
under their own name, raw directory and manifest; the ``center`` plugin is unchanged."""

from __future__ import annotations

import json
from collections.abc import Callable

import cv2
import numpy as np
import pytest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.models import Settings
from physverify.core.download import MANIFEST_FILE, read_manifest
from physverify.core.errors import MissingInputError
from physverify.datasets.physics_iq import (
    FULL,
    LOCAL,
    MASKS,
    TESTING,
    PhysicsIQDataset,
    PhysicsIQSettings,
    expected_ids,
    select_files,
)
from physverify.datasets.physics_iq_views import PhysicsIQViewsDataset, PhysicsIQViewsSettings

from .test_physics_iq import BUCKET, LISTING, SIZE, FakeBucket, object_name

SettingsFactory = Callable[..., Settings]
VIEWS = ("left", "right")
ALL_VIEWS = ("center", *VIEWS)


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    return make_settings(
        f'physics_iq.bucket_url="{BUCKET}"', f'physics_iq.listing_url="{LISTING}"', fakes=False
    )


@pytest.fixture
def section(settings: Settings) -> PhysicsIQSettings:
    return settings.section("physics_iq", PhysicsIQSettings)


@pytest.fixture(scope="module")
def mp4_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("mp4") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 16.0, SIZE)
    for i in range(6):
        writer.write(np.full((SIZE[1], SIZE[0], 3), 30 * i, np.uint8))
    writer.release()
    return path.read_bytes()


def bucket_of(section: PhysicsIQSettings, data: bytes) -> FakeBucket:
    """Every view of the local scenarios, as the real bucket lists them."""
    return FakeBucket(
        {
            object_name(scenario, kind, index=i + 1, view=view): data
            for i, scenario in enumerate(section.local)
            for kind in (TESTING, MASKS)
            for view in ALL_VIEWS
        }
    )


def test_the_plugin_is_registered_with_the_configured_views(settings: Settings) -> None:
    assert PLUGIN_KINDS["datasets"]()["physics_iq_views"] is PhysicsIQViewsDataset
    assert settings.section("physics_iq_views", PhysicsIQViewsSettings).views == VIEWS


def test_fetch_downloads_only_the_views_into_its_own_root(
    settings: Settings, section: PhysicsIQSettings, mp4_bytes: bytes
) -> None:
    bucket = bucket_of(section, mp4_bytes)
    dataset = PhysicsIQViewsDataset(settings, client=bucket)
    dest = settings.paths.raw_dir / "physics_iq_views"
    dataset.fetch(LOCAL, dest)
    manifest = read_manifest(dest / MANIFEST_FILE)
    assert manifest is not None and manifest.dataset == "physics_iq_views"
    assert manifest.clips == tuple(f"{s}__{v}_take1" for s in section.local for v in VIEWS)
    assert len(manifest.files) == 2 * len(section.local) * len(VIEWS)
    assert all("perspective-center" not in name for name, _ in bucket.ranges)
    assert not (settings.paths.raw_dir / "physics_iq").exists()


def test_clips_share_scenarios_annotations_and_static_flags(
    settings: Settings, section: PhysicsIQSettings, mp4_bytes: bytes
) -> None:
    table = {"balls-collide": [{"phrase": "ball", "expected_count": 2}]}
    path = settings.paths.annotations_dir / "physics_iq.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table), encoding="utf-8")
    dataset = PhysicsIQViewsDataset(settings, client=bucket_of(section, mp4_bytes))
    dataset.fetch(LOCAL, settings.paths.raw_dir / dataset.name)
    clips = dataset.clips(LOCAL)
    assert {c.dataset for c in clips} == {"physics_iq_views"}
    assert [c.scene_id for c in clips] == [s for s in section.local for _ in VIEWS]
    assert all(c.video_path.startswith("raw/physics_iq_views/") for c in clips)
    collide = [c for c in clips if c.scene_id == "balls-collide"]
    assert [c.object_queries[0].expected_count for c in collide] == [2, 2]
    static = {c.scene_id for c in clips if dataset.is_static(c)}
    assert static == {"stable-blocks"}


def test_the_center_plugin_ignores_the_other_views(
    settings: Settings, section: PhysicsIQSettings, mp4_bytes: bytes
) -> None:
    dataset = PhysicsIQDataset(settings, client=bucket_of(section, mp4_bytes))
    dataset.fetch(LOCAL, settings.paths.raw_dir / dataset.name)
    assert [c.clip_id for c in dataset.clips(LOCAL)] == [
        f"{s}__center_take1" for s in section.local
    ]


def test_precise_full_selection_is_the_108_preregistered_ids(
    make_settings: SettingsFactory,
) -> None:
    precise = make_settings(fakes=False, profile="precise")
    s = precise.section("physics_iq", PhysicsIQSettings)
    views = precise.section("physics_iq_views", PhysicsIQViewsSettings).views
    ids = expected_ids(s, FULL, views)
    assert len(ids) == len(set(ids)) == 108
    assert {i.split("__")[1].split("_")[0] for i in ids} == set(VIEWS)
    names = [
        object_name(scenario, kind, fps=s.fps_file, view=view, take=take)
        for scenario in (*s.full_dynamic, *s.full_static)
        for take in s.takes
        for view in ALL_VIEWS
        for kind in (TESTING, MASKS)
    ]
    assert tuple(f.clip_id for f in select_files(names, s, FULL, views)) == ids


def test_a_missing_view_names_its_own_fetch_command(settings: Settings) -> None:
    with pytest.raises(MissingInputError, match="download fetch physics_iq_views"):
        PhysicsIQViewsDataset(settings).clips(LOCAL)


def test_both_real_controls_are_downloadable_by_name_only(settings: Settings) -> None:
    from physverify.cli.commands.download import fetch_targets
    from physverify.diagnosis.settings import diagnosis_settings

    real = diagnosis_settings(settings).real
    assert real.sealed_controls == ("physics_iq_views", "objectron_static")
    assert not set(real.sealed_controls) & set(real.datasets)
    registry = dict.fromkeys((*real.datasets, *real.sealed_controls), object)
    assert len(fetch_targets(registry, real.datasets, None, real.sealed_controls)) == 2
    assert fetch_targets(registry, real.datasets, "physics_iq_views", real.sealed_controls)


def test_raw_dirs_do_not_collide(settings: Settings) -> None:
    roots = {cls(settings)._root for cls in (PhysicsIQDataset, PhysicsIQViewsDataset)}
    assert roots == {settings.paths.raw_dir / n for n in ("physics_iq", "physics_iq_views")}
