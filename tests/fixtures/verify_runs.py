"""Run directories as ``verify --policy raw`` leaves them, for the readers of ``--from-run``."""

from __future__ import annotations

import json
from pathlib import Path

from wiring import run_checks_hash

from physverify.config.models import Settings
from physverify.runtime.runs import RUN_FILE
from physverify.verification.runner import CHECKS_HASH_FILE, VerificationReport, write_report


def write_verify_run(
    settings: Settings,
    run_id: str,
    report: VerificationReport,
    *,
    checks_hash: str | None = None,
    device: str | None = None,
) -> Path:
    """``report`` with the ``run.json`` and ``checks_hash.txt`` of a run under ``settings``; a
    keyword names what the run differs in."""
    folder = settings.paths.runs_dir / run_id
    folder.mkdir(parents=True)
    write_report(folder, report)
    record = {
        "profile": settings.profile,
        "device": settings.runtime.device if device is None else device,
        "commit": "c0",
    }
    (folder / RUN_FILE).write_text(json.dumps(record), encoding="utf-8")
    stamp = run_checks_hash(settings) if checks_hash is None else checks_hash
    (folder / CHECKS_HASH_FILE).write_text(stamp + "\n", encoding="utf-8")
    return folder
