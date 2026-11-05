from typing import Any

import pytest

from physverify.app import learned
from physverify.calibration.sealed_ledger import SealedSetError

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
