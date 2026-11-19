"""Minimal decoder of the protobuf wire format: enough to read a few Objectron fields.

Supported wire types: varint (0), 64-bit (1, skipped as raw bytes), length-delimited (2) and
32-bit (5). Groups (3, 4) are rejected. Floats are little-endian IEEE 754 binary32.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from physverify.core.errors import PhysverifyError

VARINT = 0
FIXED64 = 1
LENGTH = 2
FIXED32 = 5
FLOAT_BYTES = 4
_FIXED_BYTES = {FIXED64: 8, FIXED32: 4}
_MAX_VARINT_BYTES = 10


class WireError(PhysverifyError):
    """Malformed or truncated protobuf bytes."""


@dataclass(frozen=True)
class WireField:
    number: int
    wire_type: int
    value: int | bytes
    """The integer of a varint, else the raw payload bytes."""

    @property
    def payload(self) -> bytes:
        if not isinstance(self.value, bytes):
            raise WireError(f"field {self.number} is a varint, not bytes")
        return self.value

    @property
    def integer(self) -> int:
        if not isinstance(self.value, int):
            raise WireError(f"field {self.number} is not a varint")
        return self.value


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """``(value, next position)`` of the base-128 varint starting at ``pos``."""
    value = 0
    for shift_index in range(_MAX_VARINT_BYTES):
        if pos >= len(buf):
            raise WireError("truncated varint")
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << (7 * shift_index)
        if not byte & 0x80:
            return value, pos
    raise WireError("varint longer than 10 bytes")


def read_field(buf: bytes, pos: int) -> tuple[WireField, int]:
    """The field whose tag starts at ``pos`` and the position after it."""
    tag, pos = read_varint(buf, pos)
    number, wire_type = tag >> 3, tag & 0x7
    if wire_type == VARINT:
        value, pos = read_varint(buf, pos)
        return WireField(number, wire_type, value), pos
    if wire_type == LENGTH:
        size, pos = read_varint(buf, pos)
        return WireField(number, wire_type, _take(buf, pos, size)), pos + size
    if wire_type in _FIXED_BYTES:
        size = _FIXED_BYTES[wire_type]
        return WireField(number, wire_type, _take(buf, pos, size)), pos + size
    raise WireError(f"unsupported wire type {wire_type} (field {number})")


def _take(buf: bytes, pos: int, size: int) -> bytes:
    if pos + size > len(buf):
        raise WireError("truncated field")
    return buf[pos : pos + size]


def iter_fields(buf: bytes) -> Iterator[WireField]:
    pos = 0
    while pos < len(buf):
        field, pos = read_field(buf, pos)
        yield field


def floats(field: WireField) -> tuple[float, ...]:
    """A repeated float: packed (length-delimited) or one unpacked 32-bit value."""
    data = field.payload
    if field.wire_type not in (LENGTH, FIXED32) or len(data) % FLOAT_BYTES:
        raise WireError(f"field {field.number} is not a float or packed floats")
    return struct.unpack(f"<{len(data) // FLOAT_BYTES}f", data)


def repeated_floats(fields: Sequence[WireField], number: int) -> tuple[float, ...]:
    """Every value of the repeated float ``number``, whatever mix of packed and unpacked."""
    return tuple(v for f in fields if f.number == number for v in floats(f))


def first(fields: Sequence[WireField], number: int) -> WireField | None:
    return next((f for f in fields if f.number == number), None)


def leading_records(buf: bytes, number: int, *, complete: bool) -> tuple[list[bytes], int] | None:
    """Payloads of the consecutive ``number`` fields at the start of ``buf`` and where they end.

    ``buf`` may be a prefix of a message (``complete`` False): the answer is None until a field
    with another number starts, which proves no further ``number`` record follows. Encoders
    write fields in number order, so the leading records are all of them.
    """
    records: list[bytes] = []
    pos = 0
    while pos < len(buf):
        try:
            if read_varint(buf, pos)[0] >> 3 != number:
                return records, pos
            field, pos = read_field(buf, pos)
        except WireError:
            if complete:
                raise
            return None
        records.append(field.payload)
    return (records, pos) if complete else None
