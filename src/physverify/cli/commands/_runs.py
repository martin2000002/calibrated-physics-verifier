"""Run directories and JSON evidence shared by the A9 subcommands (not a plugin module)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from physverify.cli.context import AppContext
from physverify.core.atomic import write_text_atomic
from physverify.runtime.runs import RunContext, current_commit
from physverify.schema.io import json_ready


def new_run_dir(context: AppContext) -> Path:
    settings = context.settings
    run = RunContext.create(
        settings.paths.runs_dir,
        argv=context.invocation.argv,
        device=settings.runtime.device,
        profile=settings.profile,
        config_hash=settings.config_hash(),
        commit=current_commit(settings.repo_root),
        now=dt.datetime.now(dt.UTC),
    )
    return run.run_dir


def write_json(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
