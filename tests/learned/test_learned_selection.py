import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from records import manifest

from physverify.app import learned
from physverify.calibration.sealed_ledger import SealedSetError, touched
from physverify.schema.clip import Split

SEALED_TOML = Path(__file__).resolve().parents[2] / "configs" / "verifier" / "sealed.toml"

CLIPS = ("clip-a", "clip-b")


class Dataset:
    def clips(self, subset: str | None) -> tuple[str, ...]:
        return CLIPS


def selected(monkeypatch: pytest.MonkeyPatch, looks: tuple[str, ...]) -> Any:
    monkeypatch.setattr(learned, "dataset_for", lambda app, name: Dataset())
    monkeypatch.setattr(learned, "sealed_looks", lambda app, selections, err: looks)
    return learned._selected(None, ("kubric_synthetic", "train_h"), "learned-images", print)  # type: ignore[arg-type]


def test_a_selection_off_every_seal_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _, clips = selected(monkeypatch, ())
    assert clips == CLIPS


def test_a_selection_that_looks_at_a_sealed_set_is_refused_without_an_unseal_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSVERIFY_UNSEAL", "kubric_synthetic:eval")
    with pytest.raises(SealedSetError, match="never comes from a sealed set"):
        selected(monkeypatch, ("kubric_synthetic:eval",))


def test_train_clips_touch_no_shipped_seal_and_eval_clips_do() -> None:
    """Under the shipped ``sealed.toml`` the learned commands read train_h once its clips are
    split=train; the same clips labelled eval are the sealed ``kubric_synthetic:eval``."""
    sets = tomllib.loads(SEALED_TOML.read_text(encoding="utf-8"))["sealed"]["sets"]
    clips = (manifest(),)
    as_train = tuple(replace(c, split=Split.TRAIN) for c in clips)
    as_eval = tuple(replace(c, split=Split.EVAL) for c in clips)
    assert touched("kubric_synthetic", as_train, sets) == ()
    assert touched("kubric_synthetic", as_eval, sets) == ("kubric_synthetic:eval",)
