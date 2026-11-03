"""Looks counted per dataset version. A ledger line names ``spec@version`` for a dataset whose
content has versions, so a new render of a sealed set is a set never looked at, while the render a
line names stays looked at. Lines written before versions were recorded name the bare spec:
``[sealed_legacy] versions`` (configs/verifier/sealed_legacy.toml, committed files only) declares,
per dataset, the version such lines looked at and the commit they were written at. A bare line of
a versioned dataset the mapping does not cover fails closed: it is never taken as unseen."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

from pydantic import Field

from physverify.calibration.sealed_ledger import LedgerEntry, LedgerError, SealedSpec
from physverify.config.models import Settings, validated
from physverify.core.models import FrozenModel
from physverify.datasets._base import DatasetSource
from physverify.datasets._versions import VERSION_MARK, SealedRender, VersionedSource

LEGACY_SECTION = "sealed_legacy"
LEGACY_FILE = "configs/verifier/sealed_legacy.toml"


class LegacyLook(FrozenModel):
    version: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    """Only bare lines written at this commit are this version's looks."""
    reason: str = Field(min_length=1)


class LegacySettings(FrozenModel):
    versions: dict[str, LegacyLook] = {}


def legacy_settings(settings: Settings) -> LegacySettings:
    return validated(LegacySettings, settings.sections.get(LEGACY_SECTION, {}), LEGACY_SECTION)


def look_version(source: DatasetSource, sections: Mapping[str, Any]) -> str | None:
    """The version a look at ``source`` records: a sealed render's from the configuration (known
    before it is rendered), another versioned content's from its data, None for plain data."""
    if isinstance(source, SealedRender):
        return source.sealed_version(sections)
    if isinstance(source, VersionedSource):
        return source.content_version()
    return None


def look_name(spec: str, version: str | None) -> str:
    return spec if version is None else f"{spec}{VERSION_MARK}{version}"


def looks_of(
    specs: Collection[str],
    source_of: Callable[[str], DatasetSource],
    sections: Mapping[str, Any],
) -> dict[str, str]:
    """Each sealed spec -> the look a ledger line records for it now."""
    return {
        spec: look_name(spec, look_version(source_of(SealedSpec.parse(spec).dataset), sections))
        for spec in specs
    }


def resolved(
    entries: Sequence[LedgerEntry], looks: Mapping[str, str], legacy: LegacySettings
) -> tuple[LedgerEntry, ...]:
    """``entries`` with every bare set of a versioned spec in ``looks`` named by its legacy
    version, so they compare with the looks of ``looks`` as written."""
    return tuple(_resolved(entry, looks, legacy) for entry in entries)


def _resolved(entry: LedgerEntry, looks: Mapping[str, str], legacy: LegacySettings) -> LedgerEntry:
    sets = tuple(_legacy_look(entry, recorded, looks, legacy) for recorded in entry.sets)
    return entry if sets == entry.sets else entry.model_copy(update={"sets": sets})


def _legacy_look(
    entry: LedgerEntry, recorded: str, looks: Mapping[str, str], legacy: LegacySettings
) -> str:
    look = looks.get(recorded)
    if look is None or look == recorded:
        return recorded
    dataset = SealedSpec.parse(recorded).dataset
    mapped = legacy.versions.get(dataset)
    if mapped is None or mapped.commit != entry.commit:
        raise LedgerError(
            f"ledger line {entry.utc} (commit {entry.commit[:12]}) names {recorded} without a "
            f"version and [{LEGACY_SECTION}] versions ({LEGACY_FILE}) declares no version of "
            f"{dataset} at that commit: the version it looked at is unknown, so no look at "
            f"{look} is allowed until it is declared there"
        )
    return look_name(recorded, mapped.version)
