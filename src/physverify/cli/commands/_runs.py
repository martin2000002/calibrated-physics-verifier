"""JSON evidence written by the subcommands (not a plugin module)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from physverify.core.atomic import write_text_atomic
from physverify.schema.io import json_ready


def write_json(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
