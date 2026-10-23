"""Base pydantic models shared by settings and persisted records."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

from physverify.schema.io import json_ready


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReportModel(FrozenModel):
    def to_json(self) -> str:
        payload = json_ready(self.model_dump(mode="json"))
        return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
