"""AC14: pinned weights, verified downloads through an injected downloader, and D9."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from fakes import FAKE, JOURNAL, FakeTrack
from support import CountingLock, plan_overrides, run_pipeline
from synthetic_clip import SyntheticFixtureDataset

from conftest import SettingsFactory
from physverify.config.models import PathsSettings
from physverify.core.errors import ConfigError, MissingInputError
from physverify.extraction._base import StageProvider, WeightKind, WeightSpec, provider_key
from physverify.extraction.weights import fetch, is_present, local_path, weight_file
from physverify.schema.stages import Stage

PAYLOAD = b"pinned checkpoint bytes"
SHA = hashlib.sha256(PAYLOAD).hexdigest()
REV = "a" * 40
HF = WeightSpec(WeightKind.HF, "org/model", REV, SHA)
HF_FILE = WeightSpec(WeightKind.HF, "org/model/sub/weights.pt", REV, SHA)
URL = WeightSpec(WeightKind.URL, "https://example.org/files/ckpt.pth?download=1", REV, SHA)
GIT = WeightSpec(WeightKind.GIT, "https://example.org/code/repo.git", REV)


class FakeDownloader:
    """Writes ``payload`` where the real downloader would, and a chosen git HEAD."""

    def __init__(self, payload: bytes = PAYLOAD, head: str = REV) -> None:
        self.payload = payload
        self.head = head
        self.calls: list[str] = []

    def hf_snapshot(
        self, repo_id: str, revision: str, cache_dir: Path, allow_patterns: Sequence[str]
    ) -> Path:
        self.calls.append(f"hf:{repo_id}@{revision}")
        org, name = repo_id.split("/")
        snapshot = cache_dir / f"models--{org}--{name}" / "snapshots" / revision
        for pattern in allow_patterns:
            if "*" not in pattern:
                (snapshot / pattern).parent.mkdir(parents=True, exist_ok=True)
                (snapshot / pattern).write_bytes(self.payload)
        return snapshot

    def http_file(self, url: str, dest: Path) -> None:
        self.calls.append(f"url:{url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.payload)

    def git_checkout(self, url: str, revision: str, dest: Path) -> None:
        self.calls.append(f"git:{url}@{revision}")
        (dest / ".git").mkdir(parents=True)
        (dest / ".git" / "HEAD").write_text(self.head + "\n")


@pytest.fixture
def paths(tmp_path: Path) -> PathsSettings:
    return PathsSettings.model_validate(
        {"data_dir": str(tmp_path / "data"), "hf_home": str(tmp_path / "hf")}
    )


def test_each_kind_has_its_own_location(paths: PathsSettings) -> None:
    snapshot = paths.hf_home / "hub" / "models--org--model" / "snapshots" / REV
    assert local_path(HF, paths) == local_path(HF_FILE, paths) == snapshot
    assert weight_file(HF, paths) == snapshot / "model.safetensors"
    assert weight_file(HF_FILE, paths) == snapshot / "sub" / "weights.pt"
    assert weight_file(URL, paths) == paths.weights_dir / "ckpt.pth"
    assert weight_file(GIT, paths) == paths.vendor_dir / "repo"


@pytest.mark.parametrize("spec", [HF, HF_FILE, URL])
def test_a_verified_file_is_fetched_once(spec: WeightSpec, paths: PathsSettings) -> None:
    downloader = FakeDownloader()
    assert not is_present(spec, paths)
    assert fetch(spec, paths, downloader) == weight_file(spec, paths)
    assert fetch(spec, paths, downloader).read_bytes() == PAYLOAD
    assert len(downloader.calls) == 1
    assert is_present(spec, paths)


def test_hf_fetch_pins_a_missing_main_ref(paths: PathsSettings) -> None:
    fetch(HF, paths, FakeDownloader())
    ref = local_path(HF, paths).parent.parent / "refs" / "main"
    assert ref.read_text() == REV


@pytest.mark.parametrize("spec", [HF, URL])
def test_a_wrong_sha256_is_an_error_and_the_file_is_removed(
    spec: WeightSpec, paths: PathsSettings
) -> None:
    with pytest.raises(ConfigError, match="sha256"):
        fetch(spec, paths, FakeDownloader(payload=b"tampered"))
    assert not weight_file(spec, paths).exists()


def test_git_checkouts_are_verified_by_head(paths: PathsSettings) -> None:
    with pytest.raises(ConfigError, match="HEAD"):
        fetch(GIT, paths, FakeDownloader(head="b" * 40))
    assert not local_path(GIT, paths).exists()
    downloader = FakeDownloader()
    assert fetch(GIT, paths, downloader) == paths.vendor_dir / "repo"
    assert is_present(GIT, paths)
    fetch(GIT, paths, downloader)
    assert len(downloader.calls) == 1


class WeightedTrack(FakeTrack):
    name: ClassVar[str] = "weighted"
    weights: ClassVar[tuple[WeightSpec, ...]] = (URL,)


def test_missing_weights_fail_before_the_lock(
    make_settings: SettingsFactory,
    providers: Mapping[str, type[StageProvider]],
    fixture_dataset: SyntheticFixtureDataset,
) -> None:
    registry = {**providers, provider_key(WeightedTrack): WeightedTrack}
    settings = make_settings(*plan_overrides(detect=FAKE, track="weighted", depth=FAKE))
    lock = CountingLock()
    with pytest.raises(MissingInputError, match="fetch-weights"):
        run_pipeline(settings, registry, fixture_dataset, gt_stages=(Stage.CAMERA,), lock=lock)
    assert lock.calls == 0
    assert not any(call.startswith(("init:", "lock:")) for call in JOURNAL.calls)


def test_every_real_weight_is_pinned(real_providers: Mapping[str, type[StageProvider]]) -> None:
    specs = {spec for provider in real_providers.values() for spec in provider.weights}
    assert {spec.kind for spec in specs} == set(WeightKind)
    for spec in specs:
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), spec.ref
        expected = r"" if spec.kind is WeightKind.GIT else r"[0-9a-f]{64}"
        assert re.fullmatch(expected, spec.sha256), spec.ref
