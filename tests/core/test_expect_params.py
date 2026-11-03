"""``expect_params``: the one narrowing of a plugin's params, shared by providers and families."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from physverify.core.models import expect_params
from physverify.extraction.providers import _common
from physverify.synth.families import f1_projectile


class Wanted(BaseModel):
    value: int = 1


class Other(BaseModel):
    value: int = 1


def test_the_declared_model_is_returned_as_is() -> None:
    params = Wanted()
    assert expect_params(params, Wanted) is params


def test_another_model_is_a_type_error_naming_both() -> None:
    with pytest.raises(TypeError, match="expected Wanted, got Other"):
        expect_params(Other(), Wanted)


def test_providers_and_families_share_the_core_helper() -> None:
    assert _common.expect_params is expect_params
    assert f1_projectile.expect_params is expect_params
