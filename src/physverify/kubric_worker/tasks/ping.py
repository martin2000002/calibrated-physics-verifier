"""Boundary check: compile and import the shared schema inside the container and round-trip
records written by the host.

Job: ``{"inputs": [{"type": "<Record class>", "path": "/data/..."}], "output_dir": "/data/...",
"report": "/data/.../report.json"}``. Each input is loaded and written back unchanged (same
file name and format) into ``output_dir``.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import physverify.kubric_worker
import physverify.schema
from physverify.schema.io import Record

if TYPE_CHECKING:
    from types import ModuleType


def schema_modules() -> list[ModuleType]:
    names = sorted(info.name for info in pkgutil.iter_modules(physverify.schema.__path__))
    return [importlib.import_module(f"physverify.schema.{name}") for name in names]


def record_types(modules: list[ModuleType]) -> dict[str, type[Record]]:
    found: dict[str, type[Record]] = {}
    for module in modules:
        for name, value in vars(module).items():
            if isinstance(value, type) and issubclass(value, Record) and value is not Record:
                found[name] = value
    return found


def compile_sources() -> list[str]:
    """Compile every boundary module with this interpreter (catches newer syntax)."""
    roots = [
        Path(physverify.schema.__file__).parent,
        Path(physverify.kubric_worker.__file__).parent,
    ]
    compiled = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            compiled.append(path.name)
    return compiled


def round_trip(item: dict[str, Any], types: dict[str, type[Record]], output_dir: Path) -> str:
    cls = types[item["type"]]
    source = Path(item["path"])
    target = output_dir / source.name
    if source.suffix == ".npz":
        cls.load_npz(source).save_npz(target)
    else:
        cls.load_json(source).save_json(target)
    return str(target)


def run(job: dict[str, Any]) -> None:
    types = record_types(schema_modules())
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "compiled": compile_sources(),
        "record_types": sorted(types),
        "written": [round_trip(item, types, output_dir) for item in job.get("inputs", [])],
    }
    Path(job["report"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
