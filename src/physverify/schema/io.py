"""Record base class and the only JSON / NPZ codecs of the project.

Files carry an envelope ``{"schema_version", "type", "data"}``. NPZ files keep array fields as
native arrays and the rest of the record as JSON in the ``__meta__`` entry; they are always read
with ``allow_pickle=False``. Everything here runs on numpy 1.21 (container) and 2.x (host).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, Union

import numpy as np

from physverify.schema._validate import ArraySpec, check_arrays, is_number
from physverify.schema.errors import SchemaError, SchemaVersionError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import Self

    from numpy.typing import NDArray

SCHEMA_VERSION = "1.0"

PathLike = Union[str, Path]
R = TypeVar("R", bound="Record")
T = TypeVar("T")

_META_KEY = "__meta__"
_ARRAY_TAG = "__ndarray__"


class Record:
    """Frozen schema record: validated on construction, NaN-aware equality, JSON/NPZ persistence.

    Subclasses are ``@dataclass(frozen=True, eq=False)``, declare their array fields in
    ``_arrays`` and implement ``from_dict``; ``to_dict`` and the file codecs are generic.
    """

    __dataclass_fields__: ClassVar[dict[str, Any]]
    _arrays: ClassVar[Mapping[str, ArraySpec]] = MappingProxyType({})

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        check_arrays(type(self).__name__, self.field_values(), self._arrays)

    def field_values(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, Record)
        return values_equal(self.field_values(), other.field_values())

    __hash__ = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return {name: encode(value) for name, value in self.field_values().items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        raise NotImplementedError

    @classmethod
    def decode_arrays(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        """Decode the declared array fields of ``data`` (encoded JSON or raw NPZ arrays)."""
        out: dict[str, Any] = {}
        for name, spec in cls._arrays.items():
            value = data.get(name)
            out[name] = decode_optional_array(value) if spec.optional else decode_array(value)
        return out

    def save_json(self, path: PathLike) -> None:
        Path(path).write_text(dumps(envelope(self, self.to_dict())), encoding="utf-8")

    @classmethod
    def load_json(cls, path: PathLike) -> Self:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return build(cls, open_envelope(raw, cls.__name__))

    def save_npz(self, path: PathLike) -> None:
        arrays: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        for name, value in self.field_values().items():
            if isinstance(value, np.ndarray):
                arrays[name] = _refuse_objects(name, value)
            else:
                meta[name] = encode(value)
        arrays[_META_KEY] = np.array(dumps(envelope(self, meta)))
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load_npz(cls, path: PathLike) -> Self:
        return build(cls, _read_npz(Path(path), cls.__name__))


def envelope(record: Record, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "type": type(record).__name__, "data": data}


def open_envelope(raw: object, expected_type: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "type", "data"}:
        raise SchemaError(f"not a physverify {expected_type} file")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{expected_type}: file schema_version {raw['schema_version']!r}, "
            f"code expects {SCHEMA_VERSION!r}"
        )
    if raw["type"] != expected_type:
        raise SchemaError(f"expected a {expected_type} file, got {raw['type']!r}")
    data = raw["data"]
    if not isinstance(data, dict):
        raise SchemaError(f"{expected_type}: data must be an object")
    return data


def build(cls: type[R], data: Mapping[str, Any]) -> R:
    """Run ``cls.from_dict`` and report malformed input as SchemaError."""
    try:
        return cls.from_dict(data)
    except SchemaError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise SchemaError(f"invalid {cls.__name__}: {exc!r}") from exc


def _read_npz(path: Path, expected_type: str) -> dict[str, Any]:
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SchemaError(f"{path}: not a readable npz file ({exc})") from exc
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise SchemaError(f"{path}: expected an .npz archive")
    with loaded:
        if _META_KEY not in loaded.files:
            raise SchemaError(f"{path}: missing {_META_KEY} entry")
        data = open_envelope(json.loads(str(loaded[_META_KEY].item())), expected_type)
        for key in loaded.files:
            if key != _META_KEY:
                data[key] = _load_entry(loaded, key, path)
    return data


def _load_entry(loaded: np.lib.npyio.NpzFile, key: str, path: Path) -> NDArray[Any]:
    try:
        value: NDArray[Any] = loaded[key]
    except ValueError as exc:
        raise SchemaError(f"{path}: entry {key!r} is a pickled object array") from exc
    return value


def _refuse_objects(name: str, value: NDArray[Any]) -> NDArray[Any]:
    if value.dtype.kind == "O":
        raise SchemaError(f"{name}: object arrays cannot be stored (no pickle)")
    return value


def encode(value: Any) -> Any:
    """Convert a record value to JSON-native types (arrays become tagged objects)."""
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return encode_array(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(encode(key)): encode(item) for key, item in value.items()}
    return value


def encode_array(value: NDArray[Any]) -> dict[str, Any]:
    _refuse_objects("array", value)
    return {_ARRAY_TAG: value.dtype.str, "shape": list(value.shape), "data": value.ravel().tolist()}


def decode_optional_array(value: Any) -> NDArray[Any] | None:
    return None if value is None else decode_array(value)


def optional(value: Any, convert: Callable[[Any], T]) -> T | None:
    """Apply ``convert`` unless the decoded value is None."""
    return None if value is None else convert(value)


def decode_array(value: Any) -> NDArray[Any]:
    """Accept a raw array (from NPZ) or a tagged JSON object (from ``encode_array``)."""
    if isinstance(value, np.ndarray):
        return value
    if not isinstance(value, Mapping) or _ARRAY_TAG not in value:
        raise SchemaError(f"expected an encoded array, got {type(value).__name__}")
    dtype = np.dtype(value[_ARRAY_TAG])
    if dtype.kind == "O":
        raise SchemaError("object arrays are not allowed")
    return np.array(value["data"], dtype=dtype).reshape(tuple(value["shape"]))


def dumps(payload: Any) -> str:
    return json.dumps(encode(payload), sort_keys=True, allow_nan=True, separators=(",", ":"))


def stable_hash(payload: Any) -> str:
    """sha256 of the canonical JSON of ``payload`` (sorted keys, arrays and enums encoded)."""
    return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()


def write_jsonl(path: PathLike, records: Iterable[Record]) -> None:
    lines = [dumps(envelope(record, record.to_dict())) for record in records]
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def read_jsonl(cls: type[R], path: PathLike) -> list[R]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [build(cls, open_envelope(json.loads(line), cls.__name__)) for line in lines if line]


def values_equal(a: Any, b: Any) -> bool:
    """Structural equality where NaN equals NaN and arrays compare by dtype, shape and value."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _arrays_equal(a, b)
    if is_number(a) and is_number(b):
        return bool(a == b) or (math.isnan(a) and math.isnan(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and _sequences_equal(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(values_equal(a[key], b[key]) for key in a)
    return type(a) is type(b) and bool(a == b)


def _sequences_equal(a: Sequence[Any], b: Sequence[Any]) -> bool:
    return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))


def _arrays_equal(a: Any, b: Any) -> bool:
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        return False
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))
