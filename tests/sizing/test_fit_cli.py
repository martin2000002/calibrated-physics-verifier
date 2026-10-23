"""``size fit`` writes the fitted model into the verifier's config subdirectory."""

from __future__ import annotations

import math
import shutil
import tomllib
from pathlib import Path
from typing import Any

import pytest
from cli_runner import CliRunner

from physverify.app import sizing as app_sizing
from physverify.sizing.samples import SizeSample

MODEL = Path("verifier") / "size_model.toml"


def _samples(*_: Any) -> list[SizeSample]:
    return [
        SizeSample("cup", f"cup/batch-0/{video}", (math.exp(-2.4 + 0.01 * video),) * 3, "train")
        for video in range(20)
    ]


@pytest.fixture
def without_verifier_dir(config_dir: Path) -> Path:
    """The configs with ``verifier/`` gone: its files at the root, the fitted model removed."""
    verifier = config_dir / "verifier"
    (verifier / MODEL.name).unlink()
    for source in verifier.iterdir():
        shutil.move(source, config_dir / source.name)
    verifier.rmdir()
    return config_dir


def test_fit_creates_the_verifier_directory_and_writes_the_model(
    cli: CliRunner, without_verifier_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_sizing, "load_samples", _samples)
    result = cli("size", "fit")
    assert result.code == 0, result.err
    target = without_verifier_dir / MODEL
    assert f"model: {target} (1 classes" in result.out
    model = tomllib.loads(target.read_text(encoding="utf-8"))["size_model"]
    assert set(model["classes"]) == {"cup"}
