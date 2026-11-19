"""Candidate pairs of contact records shared by the CONTACTS metrics."""

from __future__ import annotations

from physverify.schema.contacts import ContactRecord

NOT_A_CANDIDATE = -1.0


def same_pair(a: ContactRecord, b: ContactRecord) -> bool:
    return {a.object_a, a.object_b} == {b.object_a, b.object_b}
