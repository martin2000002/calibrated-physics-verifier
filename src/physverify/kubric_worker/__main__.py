"""Container entry point: ``python -m physverify.kubric_worker <task> --jobs <jsonl>``.

Runs inside the Kubric image (Python 3.9, numpy 1.21): stdlib + numpy + physverify.schema only.
One process handles every job of the file, so bpy and kubric are imported once per run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from physverify.kubric_worker import TASKS_PACKAGE

if TYPE_CHECKING:
    from collections.abc import Sequence

EXIT_USAGE = 2


def available_tasks() -> list[str]:
    package = importlib.import_module(TASKS_PACKAGE)
    return sorted(
        info.name
        for info in pkgutil.iter_modules(package.__path__)
        if not info.name.startswith("_") and not info.ispkg
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="physverify.kubric_worker")
    parser.add_argument("task", help="module name under physverify.kubric_worker.tasks")
    parser.add_argument("--jobs", required=True, type=Path, help="JSON lines, one job each")
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args(argv)


def read_jobs(path: Path) -> list[dict[str, Any]]:
    jobs = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        job = json.loads(line)
        if not isinstance(job, dict):
            raise ValueError(f"{path}:{number}: a job must be a JSON object")
        jobs.append(job)
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = available_tasks()
    if args.task not in tasks:
        print(f"unknown task {args.task!r}; available: {tasks}", file=sys.stderr)
        return EXIT_USAGE
    module = importlib.import_module(f"{TASKS_PACKAGE}.{args.task}")
    for job in read_jobs(args.jobs):
        job.setdefault("threads", args.threads)
        module.run(job)
    return 0


if __name__ == "__main__":
    sys.exit(main())
