"""``calibration.sealed``: one-look specs, the ledger file and the repeat rule (WP-O AC11, AC14)."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibration_cases import clip

from physverify.calibration.sealed import (
    LedgerEntry,
    LedgerError,
    SealedSetError,
    SealedSettings,
    append_entry,
    check_repeat,
    freeze_of,
    ledger_path,
    read_entries,
    require_unsealed,
    sealed_settings,
    touched,
    unsealed,
)
from physverify.config.loader import load_settings
from physverify.core.errors import ConfigError
from physverify.schema.clip import Split

SPECS = ("kubric_synthetic:eval", "kubric_fresh")


def entry(freeze: str = "f1", sets: tuple[str, ...] = ("kubric_synthetic:eval",)) -> LedgerEntry:
    return LedgerEntry(
        utc="2026-10-20T00:00:00+00:00",
        commit="abc",
        command="verify",
        argv=("physverify", "verify"),
        sets=sets,
        sealed=SPECS,
        datasets=(),
        artifacts={},
        config_hash="cfg",
        freeze=freeze,
        repeat=False,
        run_dir=None,
        user="u",
        host="h",
    )


@pytest.mark.parametrize("spec", ["Kubric", "kubric:test", "kubric:", "1kubric", "a-b"])
def test_an_invalid_spec_is_a_config_error(
    config_dir: Path, environ: dict[str, str], spec: str
) -> None:
    """AC11."""
    settings = load_settings(config_dir, environ=environ, overrides=(f'sealed.sets=["{spec}"]',))
    with pytest.raises(ConfigError, match="sealed"):
        sealed_settings(settings)


def test_the_committed_sets_parse(config_dir: Path, environ: dict[str, str]) -> None:
    sealed = sealed_settings(load_settings(config_dir, environ=environ))
    assert "kubric_synthetic:eval" in sealed.sets
    assert ledger_path(sealed, Path("/h"), Path("/r")) == Path("/h/sealed-ledger.txt")


def test_a_split_spec_touches_only_its_split() -> None:
    """AC11: kubric_synthetic eval is touched, its calib is not, another dataset's eval is not."""
    calib, held = clip("c", "s0"), clip("e", "s1", Split.EVAL)
    assert touched("kubric_synthetic", [calib, held], SPECS) == ("kubric_synthetic:eval",)
    assert touched("kubric_synthetic", [calib], SPECS) == ()
    assert touched("physics_iq", [held], SPECS) == ()
    assert touched("kubric_fresh", [calib], SPECS) == ("kubric_fresh",)
    assert touched("kubric_fresh", [], SPECS) == ()


def test_every_touched_spec_must_be_unsealed() -> None:
    found = ("kubric_synthetic:eval", "kubric_fresh")
    require_unsealed(found, unsealed(" kubric_fresh, kubric_synthetic:eval "))
    with pytest.raises(SealedSetError, match="PHYSVERIFY_UNSEAL"):
        require_unsealed(found, unsealed("kubric_fresh"))
    with pytest.raises(SealedSetError):
        require_unsealed(found, unsealed(None))


def test_lines_append_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "ledger.txt"
    append_entry(path, entry("f1"))
    append_entry(path, entry("f2"))
    assert [e.freeze for e in read_entries(path)] == ["f1", "f2"]
    assert read_entries(tmp_path / "absent.txt") == ()


def test_an_unwritable_or_corrupt_ledger_fails_closed(tmp_path: Path) -> None:
    """AC14: the ledger's directory is a file."""
    (tmp_path / "file").write_text("x")
    with pytest.raises(LedgerError, match="cannot write"):
        append_entry(tmp_path / "file" / "ledger.txt", entry())
    corrupt = tmp_path / "corrupt.txt"
    corrupt.write_text("{not json\n")
    with pytest.raises(LedgerError, match="cannot read"):
        read_entries(corrupt)


def test_a_repeat_under_another_freeze_needs_the_repeat_variable() -> None:
    """Juicio answer 4: same freeze passes, another freeze needs PHYSVERIFY_UNSEAL_REPEAT."""
    earlier = [entry("f1"), entry("f9", sets=("kubric_fresh",))]
    found = ("kubric_synthetic:eval",)
    assert check_repeat(earlier, found, "f1", allow=False) is False
    with pytest.raises(SealedSetError, match="PHYSVERIFY_UNSEAL_REPEAT"):
        check_repeat(earlier, found, "f2", allow=False)
    assert check_repeat(earlier, found, "f2", allow=True) is True


def test_the_freeze_moves_with_each_part() -> None:
    base = freeze_of("c", "h", {"thresholds": "t", "error_profile": None})
    assert base == freeze_of("c", "h", {"thresholds": "t", "error_profile": None})
    assert base != freeze_of("d", "h", {"thresholds": "t", "error_profile": None})
    assert base != freeze_of("c", "h", {"thresholds": "u", "error_profile": None})
    assert SealedSettings().sets == ()
