"""Base pydantic models shared by settings and persisted records."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from physverify.schema.io import json_ready

LENIENT = "lenient"
"""Validation-context flag: drop keys this version does not know (newer minor version)."""
SCHEMA_KEY = "schema"
SCHEMA_FIELD = "schema_"


def _is_none(value: object) -> bool:
    return value is None


def absent_when_none() -> Any:
    """A field default None whose key is left out of the record while it is None, so a record
    without it reads (and hashes) as before the field existed."""
    return Field(default=None, exclude_if=_is_none)


def expect_params[P: BaseModel](params: BaseModel, model: type[P]) -> P:
    """``params`` narrowed to the ``model`` a plugin declared; another model is a wiring bug."""
    if not isinstance(params, model):
        raise TypeError(f"expected {model.__name__}, got {type(params).__name__}")
    return params


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReportModel(FrozenModel):
    def to_json(self) -> str:
        payload = json_ready(self.model_dump(mode="json"))
        return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


class ArtifactModel(ReportModel):
    """A versioned artifact: a newer minor version may only add optional fields, which a lenient
    load drops; the ``schema_`` field is written as ``schema``."""

    @model_validator(mode="before")
    @classmethod
    def _drop_unknown(cls, data: Any, info: ValidationInfo) -> Any:
        if isinstance(data, dict) and info.context and info.context.get(LENIENT):
            known = {*cls.model_fields, *(f.alias for f in cls.model_fields.values() if f.alias)}
            return {key: value for key, value in data.items() if key in known}
        return data

    def to_json(self) -> str:
        payload = json.loads(super().to_json())
        if SCHEMA_FIELD in payload:
            payload[SCHEMA_KEY] = payload.pop(SCHEMA_FIELD)
        return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
