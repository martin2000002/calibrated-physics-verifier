"""Base pydantic models shared by settings and persisted records."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

from physverify.schema.io import json_ready

LENIENT = "lenient"
"""Validation-context flag: drop keys this version does not know (newer minor version)."""
SCHEMA_KEY = "schema"
SCHEMA_FIELD = "schema_"


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
