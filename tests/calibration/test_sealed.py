"""``calibration.sealed_ledger``: one-look specs, the ledger file and the repeat rule."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest
from calibration_cases import clip

from physverify.calibration.sealed_ledger import (
    LedgerEntry,
    LedgerError,
    SealedSetError,
    append_entry,
    check_repeat,
    freeze_of,
    ledger_path,
    lock_path,
    locked,
    look_config_hash,
    read_entries,
    require_clean,
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
    settings = load_settings(config_dir, environ=environ, overrides=(f'sealed.sets=["{spec}"]',))
    with pytest.raises(ConfigError, match="sealed"):
        sealed_settings(settings)


def test_the_committed_sets_parse(config_dir: Path, environ: dict[str, str]) -> None:
    sealed = sealed_settings(load_settings(config_dir, environ=environ))
    assert "kubric_synthetic:eval" in sealed.sets
    assert ledger_path(sealed, Path("/h"), Path("/r")) == Path("/h/sealed-ledger.txt")


def test_a_split_spec_touches_only_its_split() -> None:
    """kubric_synthetic eval is touched, its calib is not, another dataset's eval is not."""
    calib, held = clip("c", "s0"), clip("e", "s1", Split.EVAL)
    assert touched("kubric_synthetic", [calib, held], SPECS) == ("kubric_synthetic:eval",)
    assert touched("kubric_synthetic", [calib], SPECS) == ()
    assert touched("physics_iq", [held], SPECS) == ()
    assert touched("kubric_fresh", [calib], SPECS) == ("kubric_fresh",)
    assert touched("kubric_fresh", [], SPECS) == ()


def test_the_freeze_sets_are_sealed_by_their_own_names(
    config_dir: Path, environ: dict[str, str]
) -> None:
    """Each fresh set seals only itself; the development real sets stay
    open and the new real controls are sealed."""
    specs = sealed_settings(load_settings(config_dir, environ=environ)).sets
    held, calib = clip("e", "s1", Split.EVAL), clip("c", "s0")
    for name in ("kubric_fresh", "kubric_fresh_ood", "kubric_fresh_blind"):
        assert touched(name, [held, calib], specs) == (name,)
    assert touched("kubric_synthetic", [held], specs) == ("kubric_synthetic:eval",)
    assert touched("kubric_synthetic", [calib], specs) == ()
    for name in ("physics_iq", "pisa_real"):
        assert touched(name, [held], specs) == ()
    for name in ("physics_iq_views", "objectron_static"):
        assert touched(name, [held], specs) == (name,)


def test_every_touched_spec_must_be_unsealed() -> None:
    found = ("kubric_synthetic:eval", "kubric_fresh")
    require_unsealed(found, unsealed(" kubric_fresh, kubric_synthetic:eval "))
    with pytest.raises(SealedSetError, match="PHYSVERIFY_UNSEAL"):
        require_unsealed(found, unsealed("kubric_fresh"))
    with pytest.raises(SealedSetError):
        require_unsealed(found, unsealed(None))


def test_lines_append_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "ledger.txt"
    path.touch()
    assert read_entries(path) == ()
    append_entry(path, entry("f1"))
    append_entry(path, entry("f2"))
    assert [e.freeze for e in read_entries(path)] == ["f1", "f2"]


def test_the_lock_excludes_other_looks_and_leaves_the_ledger_alone(tmp_path: Path) -> None:
    path = tmp_path / "ledger.txt"
    path.touch()
    with locked(path):
        fd = os.open(lock_path(path), os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        append_entry(path, entry("f1"))
    assert [e.freeze for e in read_entries(path)] == ["f1"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ledger.txt", "ledger.txt.lock"]


def test_a_missing_ledger_is_never_created(tmp_path: Path) -> None:
    """A wrong HOME or path fails closed instead of starting an empty ledger."""
    absent = tmp_path / "deep" / "ledger.txt"
    with pytest.raises(LedgerError, match="created once by hand"):
        read_entries(absent)
    with pytest.raises(LedgerError, match=f"touch {absent}"):
        append_entry(absent, entry())
    with pytest.raises(LedgerError, match="created once by hand"), locked(absent):
        pass
    assert not absent.parent.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a read-only file")
def test_an_unwritable_or_corrupt_ledger_fails_closed(tmp_path: Path) -> None:
    locked = tmp_path / "locked.txt"
    locked.touch()
    locked.chmod(0o444)
    with pytest.raises(LedgerError, match="cannot write"):
        append_entry(locked, entry())
    corrupt = tmp_path / "corrupt.txt"
    corrupt.write_text("{not json\n")
    with pytest.raises(LedgerError, match="cannot read"):
        read_entries(corrupt)


def test_a_repeat_under_another_freeze_needs_the_repeat_variable() -> None:
    """Same freeze passes, another freeze needs PHYSVERIFY_UNSEAL_REPEAT."""
    earlier = [entry("f1"), entry("f9", sets=("kubric_fresh",))]
    found = ("kubric_synthetic:eval",)
    assert check_repeat(earlier, found, "f1", allow=False) is False
    with pytest.raises(SealedSetError, match="PHYSVERIFY_UNSEAL_REPEAT"):
        check_repeat(earlier, found, "f2", allow=False)
    assert check_repeat(earlier, found, "f2", allow=True) is True


def test_an_allowed_look_at_a_covered_set_is_a_repeat_under_the_same_freeze() -> None:
    """run_final.sh's retry sets the repeat variable and is marked a repeat."""
    earlier = [entry("f1"), entry("f9", sets=("kubric_fresh",))]
    found = ("kubric_synthetic:eval",)
    assert check_repeat(earlier, found, "f1", allow=True) is True
    assert check_repeat(earlier, ("kubric_fresh_ood",), "f1", allow=True) is False
    assert check_repeat([], found, "f1", allow=True) is False


def test_the_freeze_moves_with_each_part() -> None:
    base = freeze_of("c", "h", {"thresholds": "t", "error_profile": None})
    assert base == freeze_of("c", "h", {"thresholds": "t", "error_profile": None})
    assert base != freeze_of("d", "h", {"thresholds": "t", "error_profile": None})
    assert base != freeze_of("c", "h", {"thresholds": "u", "error_profile": None})


def test_a_missing_sealed_section_fails_closed(config_dir: Path, environ: dict[str, str]) -> None:
    (config_dir / "verifier" / "sealed.toml").unlink()
    with pytest.raises(ConfigError, match=r"no \[sealed\] section"):
        sealed_settings(load_settings(config_dir, environ=environ))
    with pytest.raises(ConfigError, match=r"unknown section \[sealed\]"):
        load_settings(config_dir, environ=environ, overrides=("sealed.sets=[]",))
    (config_dir / "verifier" / "sealed.toml").write_text("[sealed]\nsets = []\n")
    with pytest.raises(ConfigError, match=r"\[sealed\] is invalid"):
        sealed_settings(load_settings(config_dir, environ=environ))


def test_the_look_hash_ignores_paths_and_threads_not_sections(
    config_dir: Path, environ: dict[str, str]
) -> None:
    base = look_config_hash(load_settings(config_dir, environ=environ))
    for same in ("runtime.threads=3", "paths.runs_dir=/elsewhere"):
        moved = load_settings(config_dir, environ=environ, overrides=(same,))
        assert look_config_hash(moved) == base
    for other in ("calibration.bootstrap=7", "runtime.device=cuda"):
        moved = load_settings(config_dir, environ=environ, overrides=(other,))
        assert look_config_hash(moved) != base


def test_a_look_needs_a_clean_checkout() -> None:
    require_clean("abc1234", "")
    with pytest.raises(SealedSetError, match="uncommitted"):
        require_clean("abc1234", " M configs/verifier/checks.toml\n")
    with pytest.raises(SealedSetError, match="git checkout"):
        require_clean("", None)
