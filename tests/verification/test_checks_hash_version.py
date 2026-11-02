"""L24 (ADR 0039): a check's version enters ``checks_hash``; with the same version, parameters and
variant the hash stays put."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel

from physverify.checks._base import Check, CheckInputs
from physverify.schema.checks import CheckResult
from physverify.verification.check_set import ChecksSettings, checks_hash

TOY = "T1"


class ToyParams(BaseModel):
    limit: float = 1.0


class ToyCheck:
    check_id: ClassVar[str] = TOY
    version: ClassVar[str] = "1"
    requires: ClassVar[frozenset[str]] = frozenset()
    time_dependent: ClassVar[bool] = False
    magnitude_unit: ClassVar[str] = "toy"
    Params: ClassVar[type[BaseModel]] = ToyParams

    def __init__(self, params: BaseModel) -> None:
        self.params = params

    def evaluate(self, inputs: CheckInputs) -> Sequence[CheckResult]:
        return ()


def toy(**attributes: Any) -> type[Check]:
    """A copy of ``ToyCheck`` with ``attributes`` declared on it."""
    return type("Toy", (ToyCheck,), {"version": ToyCheck.version, **attributes})


def hash_of(check: type[Check], **params: Any) -> str:
    return checks_hash({TOY: check}, ChecksSettings({TOY: params} if params else {}))


def test_the_same_version_and_params_hash_alike() -> None:
    assert hash_of(toy()) == hash_of(toy()) == hash_of(ToyCheck)
    assert hash_of(toy(), limit=1.0) == hash_of(ToyCheck)


def test_another_version_moves_the_hash() -> None:
    assert hash_of(toy(version="2")) != hash_of(ToyCheck)
    assert hash_of(toy(version="2"), limit=2.0) != hash_of(ToyCheck, limit=2.0)


def test_other_params_move_the_hash() -> None:
    assert hash_of(ToyCheck, limit=2.0) != hash_of(ToyCheck)


def test_a_variant_hashes_apart_from_its_default() -> None:
    assert hash_of(toy(variant="v2")) != hash_of(ToyCheck)
    assert hash_of(toy(variant="v2")) != hash_of(toy(variant="v3"))
