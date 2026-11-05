"""A table of named float columns keyed by string rows, saved as one ``.npz`` with string
metadata; ``sha256`` of the file identifies it."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from physverify.core.errors import PhysverifyError

FloatArray = NDArray[np.float64]


class FeatureTableError(PhysverifyError):
    """A feature table that does not match what its reader declares."""


@dataclass(frozen=True)
class FeatureTable:
    keys: tuple[str, ...]
    columns: tuple[str, ...]
    values: FloatArray
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.keys), len(self.columns)):
            raise FeatureTableError(
                f"values {self.values.shape} do not match {len(self.keys)} keys x "
                f"{len(self.columns)} columns"
            )
        if len(set(self.keys)) != len(self.keys):
            raise FeatureTableError("row keys must be unique")

    def row(self, key: str) -> FloatArray | None:
        index = self._index.get(key)
        return None if index is None else self.values[index]

    def require_columns(self, columns: Sequence[str]) -> None:
        if tuple(columns) != self.columns:
            raise FeatureTableError(f"columns {self.columns} are not the expected {tuple(columns)}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(
                handle,
                keys=np.asarray(self.keys, dtype=np.str_),
                columns=np.asarray(self.columns, dtype=np.str_),
                values=self.values.astype(np.float64),
                meta=np.asarray(json.dumps(dict(self.meta), sort_keys=True)),
            )

    @classmethod
    def load(cls, path: Path) -> FeatureTable:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                keys=tuple(str(k) for k in data["keys"]),
                columns=tuple(str(c) for c in data["columns"]),
                values=np.asarray(data["values"], dtype=np.float64),
                meta=dict(json.loads(str(data["meta"]))),
            )

    @classmethod
    def from_rows(
        cls, rows: Mapping[str, Sequence[float]], columns: Sequence[str], meta: Mapping[str, str]
    ) -> FeatureTable:
        keys = tuple(sorted(rows))
        values = np.asarray([list(rows[k]) for k in keys], dtype=np.float64).reshape(
            len(keys), len(columns)
        )
        return cls(keys, tuple(columns), values, dict(meta))

    @cached_property
    def _index(self) -> dict[str, int]:
        return {key: i for i, key in enumerate(self.keys)}
