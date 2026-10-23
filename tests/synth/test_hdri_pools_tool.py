"""`tools/hdri_pools.py` prints the pools table from a manifest and a zip listing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

TOOL = Path(__file__).resolve().parents[2] / "tools" / "hdri_pools.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hdri_pools_tool", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_table_lists_the_ids_outside_the_training_panoramas(tmp_path: Path) -> None:
    manifest = tmp_path / "HDRI_haven.json"
    manifest.write_text(json.dumps({"assets": {"lake": {}, "forest": {}, "seen": {}}}))
    listing = tmp_path / "openpano-listing.txt"
    listing.write_text("openpano/panoramas/train/seen_16k.jpg 1024\nopenpano/README.md 10\n")

    lines = _load_tool().pools_table(manifest, listing)

    assert lines[0] == "[dataset.hdri_pools]"
    dev = json.loads(lines[1].removeprefix("openpano_ood_dev = "))
    reserved = json.loads(lines[2].removeprefix("openpano_ood_reserved = "))
    assert len(dev) == 1 and len(reserved) == 1
    assert sorted(dev + reserved) == ["forest", "lake"]
