"""``name@version``: a dataset content named by its version label, for the datasets whose
content has versions (one label may be rendered more than once; ``dataset_identity`` tells
those apart)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Protocol, runtime_checkable

from physverify.datasets._base import DatasetSource

VERSION_MARK: Final = "@"


@runtime_checkable
class VersionedSource(Protocol):
    def content_version(self) -> str:
        """The version label of this content."""
        ...


@runtime_checkable
class SealedRender(Protocol):
    def sealed_version(self, sections: Mapping[str, Any]) -> str:
        """The version of the sealed render that the configuration ``sections`` declare."""
        ...


def versioned_name(source: DatasetSource) -> str | None:
    """``name@version`` of a dataset whose content has versions; None for plain datasets."""
    if not isinstance(source, VersionedSource):
        return None
    return f"{source.name}{VERSION_MARK}{source.content_version()}"


def versioned_parts(spec: str) -> tuple[str, str]:
    """``(name, version)`` of a ``name@version`` spec; an empty part when it has none."""
    name, _, version = spec.partition(VERSION_MARK)
    return name, version
