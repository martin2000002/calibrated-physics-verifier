"""The SAM video loader resolves every hub lookup in the configured cache, offline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from physverify.config.models import PathsSettings
from physverify.extraction._base import StageContext
from physverify.extraction.providers._sam_video import load_video_model
from physverify.extraction.providers.track_edgetam import MODEL
from physverify.extraction.weights import local_path

hub_constants = pytest.importorskip("huggingface_hub.constants")


class Recorder:
    """A transformers stand-in whose ``from_pretrained`` records the hub settings it sees."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def from_pretrained(self, snapshot: Path) -> Recorder:
        self.seen.append(
            {
                "snapshot": snapshot,
                "HF_HOME": os.environ.get("HF_HOME"),
                "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
                "constant": hub_constants.HF_HUB_CACHE,
                "offline": hub_constants.HF_HUB_OFFLINE,
            }
        )
        return self

    def to(self, device: str) -> Recorder:
        return self

    def eval(self) -> Recorder:
        return self


def test_the_model_and_processor_load_from_the_configured_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    fake = SimpleNamespace(**{MODEL.class_name: recorder, "Sam2VideoProcessor": recorder})
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "default"))
    paths = PathsSettings.model_validate(
        {"data_dir": str(tmp_path / "data"), "hf_home": str(tmp_path / "hf")}
    )
    ctx = cast(StageContext, SimpleNamespace(paths=paths, device="cpu"))
    before = hub_constants.HF_HUB_CACHE

    load_video_model(MODEL, ctx)

    expected = {
        "snapshot": local_path(MODEL.weights, paths),
        "HF_HOME": str(tmp_path / "hf"),
        "HF_HUB_CACHE": str(tmp_path / "hf" / "hub"),
        "constant": str(tmp_path / "hf" / "hub"),
        "offline": True,
    }
    assert recorder.seen == [expected, expected]
    assert os.environ["HF_HOME"] == str(tmp_path / "default")
    assert before == hub_constants.HF_HUB_CACHE
