"""``calibration.ledger_versions``: a look counts against one dataset version, and a legacy bare
line counts only against the version the committed mapping declares for it."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest

from physverify.calibration.ledger_versions import (
    LegacyLook,
    LegacySettings,
    legacy_settings,
    looks_of,
    resolved,
)
from physverify.calibration.sealed_ledger import (
    LedgerEntry,
    LedgerError,
    SealedSetError,
    check_repeat,
    earlier,
)
from physverify.config.loader import load_settings

LEGACY_COMMIT = "0" * 40
COMMIT = "c" * 40
FRESH = "kubric_fresh"
NEW = f"{FRESH}@v2-fresh2"
OLD = f"{FRESH}@v2-fresh"
LEGACY = LegacySettings(
    versions={FRESH: LegacyLook(version="v2-fresh", commit=LEGACY_COMMIT, reason="config")}
)


def line(sets: tuple[str, ...], commit: str = LEGACY_COMMIT, freeze: str = "old") -> LedgerEntry:
    return LedgerEntry(
        utc="2026-10-30T16:56:41+00:00",
        commit=commit,
        command="calibrate evaluate",
        argv=("physverify",),
        sets=sets,
        sealed=sets,
        datasets=(),
        artifacts={},
        config_hash="cfg",
        freeze=freeze,
        repeat=False,
        run_dir=None,
        user="u",
        host="h",
    )


def looked(entries: list[LedgerEntry], look: str) -> list[LedgerEntry]:
    return earlier(resolved(entries, {FRESH: look}, LEGACY), [look])


def test_a_legacy_line_blocks_only_its_mapped_version() -> None:
    legacy = [line((FRESH,))]
    assert [entry.sets for entry in looked(legacy, OLD)] == [(OLD,)]
    assert looked(legacy, NEW) == []


def test_a_new_version_stays_unseen_until_its_versioned_line_is_appended() -> None:
    entries = [line((FRESH,)), line(("kubric_fresh_blind@v2-fresh2",), COMMIT)]
    assert looked(entries, NEW) == []
    entries.append(line((NEW,), COMMIT, freeze="new"))
    assert [e.sets for e in looked(entries, NEW)] == [(NEW,)]


def test_a_second_look_at_a_versioned_line_is_refused() -> None:
    entries = resolved([line((FRESH,)), line((NEW,), COMMIT, "new")], {FRESH: NEW}, LEGACY)
    with pytest.raises(SealedSetError, match=rf"\['{NEW}'\] were already looked at"):
        check_repeat(entries, [NEW], "another", allow=False)
    assert check_repeat(entries, [NEW], "new", allow=False) is False
    with pytest.raises(SealedSetError, match=r"\['kubric_fresh@v2-fresh'\]"):
        check_repeat(resolved([line((FRESH,))], {FRESH: OLD}, LEGACY), [OLD], "new", allow=False)


@pytest.mark.parametrize(
    ("entry", "look"),
    [
        (line(("kubric_fresh_ood",)), "kubric_fresh_ood@v2-fresh-ood"),
        (line((FRESH,), commit=COMMIT), NEW),
    ],
    ids=["unmapped-dataset", "other-commit"],
)
def test_an_unmapped_legacy_line_fails_closed(entry: LedgerEntry, look: str) -> None:
    spec = look.partition("@")[0]
    with pytest.raises(LedgerError, match=f"names {spec} without a version.*no look at {look}"):
        resolved([entry], {spec: look}, LEGACY)


def test_lines_of_other_specs_and_of_plain_datasets_are_left_as_written() -> None:
    entries = (line(("kubric_fresh_ood",)), line(("physics_iq_views",)))
    plain = {"physics_iq_views": "physics_iq_views"}
    assert resolved(entries, plain, LEGACY) == entries
    assert resolved(entries, {}, LegacySettings()) == entries


class _Plain:
    name: ClassVar[str] = "plain"


class _Versioned(_Plain):
    def content_version(self) -> str:
        return "v-data"


class _Render(_Versioned):
    def sealed_version(self, sections: Mapping[str, Any]) -> str:
        return str(sections["render"])


def test_a_look_names_the_configured_render_then_the_data_version() -> None:
    sources: dict[str, Any] = {"render": _Render(), "versioned": _Versioned(), "plain": _Plain()}
    looks = looks_of(["render", "versioned:eval", "plain"], sources.__getitem__, {"render": "r1"})
    assert looks == {
        "render": "render@r1",
        "versioned:eval": "versioned:eval@v-data",
        "plain": "plain",
    }


def test_the_committed_mapping_names_the_versions_of_the_legacy_looks(config_dir: Path) -> None:
    versions = legacy_settings(load_settings(config_dir, environ={})).versions
    commit = "0641659e61d12b1a63d4c57cf5a391a2081d9757"
    assert {k: (v.version, v.commit) for k, v in versions.items()} == {
        "kubric_fresh": ("v2-fresh", commit),
        "kubric_synthetic": ("v1-precise", commit),
    }
