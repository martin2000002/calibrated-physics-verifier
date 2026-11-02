"""Source pins (ADR 0039): no provider, check or check variant changes its code, or the code of a
physverify module it imports, without a new ``version`` or a re-pin justified by
``tools/equivalence/run_equivalence.sh``; every one of them has a pin."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "pin_sources.py"
PROVIDER = "providers:depth/vda_small"
PHYSICS_MODULE = "physverify.physics.stencil_tail"
PLUMBING_MODULE = "physverify.core.registry"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pin_sources_tool", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load_tool()


@pytest.fixture(scope="module")
def plugins(tool: ModuleType) -> dict[str, type]:
    found: dict[str, type] = tool.pinned_plugins()
    return found


@pytest.fixture(scope="module")
def current(tool: ModuleType, plugins: Mapping[str, type]) -> dict[str, Any]:
    pins: dict[str, Any] = tool.current_pins(tool.SourceGraph(), plugins)
    return pins


def edited(tool: ModuleType, module: str) -> Callable[[str], str]:
    """The source reader with one line appended to ``module``."""

    def read(name: str) -> str:
        source: str = tool.module_source(name)
        return f"{source}\nEDITED = True\n" if name == module else source

    return read


def pins_after_editing(tool: ModuleType, plugins: Mapping[str, type], module: str) -> Any:
    return tool.current_pins(tool.SourceGraph(edited(tool, module)), plugins)


def importers(tool: ModuleType, plugins: Mapping[str, type], module: str) -> set[str]:
    graph = tool.SourceGraph()
    return {name for name, cls in plugins.items() if module in graph.closure(cls.__module__)}


def test_every_registered_plugin_has_a_pin(tool: ModuleType, current: Mapping[str, Any]) -> None:
    diff = tool.compare(current, tool.load_pins())
    assert not (diff.missing or diff.unknown), "\n".join(diff.messages())


def test_no_source_changed_under_its_version(tool: ModuleType, current: Mapping[str, Any]) -> None:
    diff = tool.compare(current, tool.load_pins())
    assert not diff.unbumped, "\n".join(diff.messages())


def test_pins_record_the_current_versions(tool: ModuleType, current: Mapping[str, Any]) -> None:
    diff = tool.compare(current, tool.load_pins())
    assert not diff.rerecord, "\n".join(diff.messages())


def test_an_edited_provider_fails_by_name(
    tool: ModuleType, plugins: Mapping[str, type], current: Mapping[str, Any]
) -> None:
    after = pins_after_editing(tool, plugins, plugins[PROVIDER].__module__)
    diff = tool.compare(after, current)
    assert PROVIDER in diff.unbumped
    assert any(m.startswith(f"{PROVIDER}: ") and "bump" in m for m in diff.messages())


def test_a_bumped_or_re_pinned_provider_passes(
    tool: ModuleType, plugins: Mapping[str, type], current: Mapping[str, Any]
) -> None:
    after = pins_after_editing(tool, plugins, plugins[PROVIDER].__module__)
    stale = tool.compare(after, current).unbumped
    bumped = {
        name: dataclasses.replace(pin, version=f"{pin.version}+1") if name in stale else pin
        for name, pin in after.items()
    }
    assert tool.compare(bumped, current).unbumped == ()
    assert tool.compare(after, tool.accepted(current, after, stale)).messages() == []


def test_a_physics_edit_fails_exactly_the_checks_that_import_it(
    tool: ModuleType, plugins: Mapping[str, type], current: Mapping[str, Any]
) -> None:
    stale = set(tool.compare(pins_after_editing(tool, plugins, PHYSICS_MODULE), current).unbumped)
    assert stale == importers(tool, plugins, PHYSICS_MODULE)
    assert "checks:R4_jump" in stale
    assert "checks:R1" not in stale


def test_the_plumbing_packages_are_not_pinned(
    tool: ModuleType, plugins: Mapping[str, type], current: Mapping[str, Any]
) -> None:
    after = pins_after_editing(tool, plugins, PLUMBING_MODULE)
    assert tool.compare(after, current).unbumped == ()


def test_imports_for_type_checking_only_are_not_followed(tool: ModuleType) -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "import physverify.physics.ties\n"
        "if TYPE_CHECKING:\n"
        "    from physverify.schema import states\n"
        "def lazy():\n"
        "    from physverify.physics import tracks\n"
    )
    found = set(tool.runtime_imports(source, "physverify.checks"))
    assert {"physverify.physics.ties", "physverify.physics.tracks"} <= found
    assert not any("states" in name for name in found)


def test_pins_round_trip(tool: ModuleType, current: Mapping[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "pins.json"
    tool.save_pins(current, path)
    assert tool.load_pins(path) == current
