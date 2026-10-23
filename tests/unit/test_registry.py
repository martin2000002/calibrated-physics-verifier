from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from cli_tree import command_modules
from fx_protocol import Widget

from physverify.app.plugins import PLUGIN_KINDS
from physverify.core.errors import DuplicatePluginError, InvalidPluginError, MissingPluginsError
from physverify.core.registry import discover

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "plugins"
HEAVY_MODULES = ("torch", "transformers", "kubric", "bpy")


def test_discovers_listed_classes_and_skips_private_modules() -> None:
    registry = discover("fx_good", (Widget,), "name")
    assert sorted(registry) == ["alpha", "beta", "beta_two"]
    assert registry["alpha"]().run() == "alpha"


def test_a_new_file_is_found_without_editing_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = f"fx_grow_{uuid.uuid4().hex[:8]}"
    shutil.copytree(FIXTURES / "fx_good", tmp_path / package)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert "gamma" not in discover(package, (Widget,), "name")
    (tmp_path / package / "gamma.py").write_text(
        "class Gamma:\n    name = 'gamma'\n\n    def run(self):\n        return 'gamma'\n\n\n"
        "PLUGINS = (Gamma,)\n"
    )
    importlib.invalidate_caches()
    assert sorted(discover(package, (Widget,), "name")) == ["alpha", "beta", "beta_two", "gamma"]


def test_duplicate_keys_raise() -> None:
    with pytest.raises(DuplicatePluginError, match="same"):
        discover("fx_duplicate", (Widget,), "name")


def test_class_not_satisfying_the_protocol_raises() -> None:
    with pytest.raises(InvalidPluginError, match="Widget"):
        discover("fx_invalid", (Widget,), "name")


def test_module_without_plugins_raises() -> None:
    with pytest.raises(MissingPluginsError, match="PLUGINS"):
        discover("fx_missing", (Widget,), "name")


def test_validate_hook_can_reject() -> None:
    def reject(plugin: type) -> None:
        raise InvalidPluginError(plugin.__name__)

    with pytest.raises(InvalidPluginError, match="Alpha"):
        discover("fx_good", (Widget,), "name", reject)


def test_callable_key() -> None:
    registry = discover("fx_good", (Widget,), lambda plugin: f"widget/{plugin.name}")
    assert "widget/alpha" in registry


def test_every_real_plugin_kind_builds() -> None:
    commands = command_modules()
    assert set(commands) >= {"doctor", "dataset", "extract", "verify", "diagnose", "data", "timing"}
    assert "ping" in PLUGIN_KINDS["container_tasks"]()
    for kind, build in PLUGIN_KINDS.items():
        assert build() is not None, kind


def test_registries_do_not_import_heavy_modules() -> None:
    script = (
        "import json, sys\n"
        "from physverify.app.plugins import PLUGIN_KINDS\n"
        "from physverify.cli.context import COMMANDS_PACKAGE, check_command, command_name\n"
        "from physverify.core.registry import discover_modules\n"
        "sizes = {kind: len(build()) for kind, build in PLUGIN_KINDS.items()}\n"
        "discover_modules(COMMANDS_PACKAGE, check_command, command_name)\n"
        f"heavy = [m for m in {HEAVY_MODULES!r} if m in sys.modules]\n"
        "print(json.dumps({'sizes': sizes, 'heavy': heavy}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    report = json.loads(done.stdout)
    assert report["heavy"] == []
    assert len(report["sizes"]) == 11
