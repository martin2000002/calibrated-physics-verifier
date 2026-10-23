"""Run directories: ``data/runs/<run_id>/run.json`` records how a result was produced."""

from __future__ import annotations

import datetime as dt
import json
import secrets
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from physverify.config.models import Settings
from physverify.core.atomic import write_text_atomic
from physverify.schema.io import json_ready

RUN_FILE: Final = "run.json"


def new_run_id(now: dt.datetime, token: str | None = None) -> str:
    suffix = token if token is not None else secrets.token_hex(3)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{suffix}"


def current_commit(repo_root: Path) -> str:
    """HEAD commit of the repository, or an empty string outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    argv: tuple[str, ...]
    device: str
    profile: str
    config_hash: str
    commit: str
    created_utc: str

    @classmethod
    def create(
        cls,
        runs_dir: Path,
        *,
        argv: Sequence[str],
        device: str,
        profile: str,
        config_hash: str,
        commit: str,
        now: dt.datetime,
    ) -> RunContext:
        run_id = new_run_id(now)
        context = cls(
            run_id=run_id,
            run_dir=runs_dir / run_id,
            argv=tuple(argv),
            device=device,
            profile=profile,
            config_hash=config_hash,
            commit=commit,
            created_utc=now.astimezone(dt.UTC).isoformat(timespec="seconds"),
        )
        context.run_dir.mkdir(parents=True, exist_ok=False)
        context.write()
        return context

    def write(self) -> None:
        record = asdict(self)
        record["run_dir"] = str(self.run_dir)
        record["argv"] = list(self.argv)
        write_text_atomic(self.run_dir / RUN_FILE, json.dumps(record, indent=2))


def new_run(settings: Settings, argv: Sequence[str], now: dt.datetime | None = None) -> RunContext:
    """A fresh run directory under ``paths.runs_dir`` with its ``run.json`` written."""
    return RunContext.create(
        settings.paths.runs_dir,
        argv=argv,
        device=settings.runtime.device,
        profile=settings.profile,
        config_hash=settings.config_hash(),
        commit=current_commit(settings.repo_root),
        now=now if now is not None else dt.datetime.now(dt.UTC),
    )


def write_json(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
