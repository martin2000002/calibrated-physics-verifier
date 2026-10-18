from __future__ import annotations

import json
import subprocess
import sys

from physverify.cli.context import PLUGIN_KINDS

EXPECTED = {
    "wan21_generate": True,
    "dpo_step": True,
    "extract_pipeline": False,
    "verify_checks": False,
    "kubric_render": False,
}
HEAVY_MODULES = ("torch", "transformers", "diffusers", "kubric", "bpy")


def test_the_registry_holds_the_four_targets() -> None:
    registry = PLUGIN_KINDS["timing_targets"]()
    assert {name: cls.requires_gpu for name, cls in registry.items()} == EXPECTED


def test_building_the_timing_registry_imports_no_heavy_module() -> None:
    script = (
        "import json, sys\n"
        "from physverify.cli.context import PLUGIN_KINDS\n"
        "names = sorted(PLUGIN_KINDS['timing_targets']())\n"
        f"heavy = [m for m in {HEAVY_MODULES!r} if m in sys.modules]\n"
        "print(json.dumps({'names': names, 'heavy': heavy}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    report = json.loads(done.stdout)
    assert report["heavy"] == []
    assert report["names"] == sorted(EXPECTED)
