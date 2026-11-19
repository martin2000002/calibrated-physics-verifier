"""Source pins of the registered providers, checks and check variants.

    PYTHONPATH=src python tools/pin_sources.py                       # list every stale pin
    PYTHONPATH=src python tools/pin_sources.py --accept checks:R1    # record the current pin

A pin is ``{version, digest}``: the plugin's declared ``version`` and the sha256 of its module's
source plus every ``physverify`` module that module imports at run time, transitively (outside
``core``, ``config``, ``app`` and ``cli``). ``tests/contracts/test_source_pins.py`` fails when a
digest moves under an unchanged version, so no output change ships without a version bump.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from physverify.app.plugins import PLUGIN_KINDS

PINS_PATH: Final = Path(__file__).resolve().parents[1] / "tests/contracts/data/source_pins.json"
PINNED_KINDS: Final = ("providers", "checks", "check_variants")
ROOT_PACKAGE: Final = "physverify"
UNPINNED_PACKAGES: Final = frozenset({"core", "config", "app", "cli"})
KIND_SEPARATOR: Final = ":"
TOOL: Final = "tools/pin_sources.py"

SourceReader = Callable[[str], str]


@dataclass(frozen=True)
class Pin:
    version: str
    digest: str


@dataclass(frozen=True)
class PinDiff:
    """``unbumped``: the digest moved, the version did not; ``rerecord``: the version moved and
    the pin still holds the former one."""

    missing: tuple[str, ...]
    unknown: tuple[str, ...]
    unbumped: tuple[str, ...]
    rerecord: tuple[str, ...]

    def messages(self) -> list[str]:
        return [
            *(f"{n}: no source pin; record it with {accept(n)}" for n in self.missing),
            *(f"{n}: pinned but not registered; drop it with {accept(n)}" for n in self.unknown),
            *(unbumped_message(n) for n in self.unbumped),
            *(f"{n}: version bumped; record the new pin with {accept(n)}" for n in self.rerecord),
        ]


def accept(name: str) -> str:
    return f"`PYTHONPATH=src python {TOOL} --accept {name}`"


def unbumped_message(name: str) -> str:
    return (
        f"{name}: its source or a physverify module it imports changed, its version did not. "
        "If its output can change, bump its `version`; if not (a refactor), show it with "
        f"tools/equivalence/run_equivalence.sh, cite that in the commit and re-pin with "
        f"{accept(name)}"
    )


def module_source(name: str) -> str:
    """The source file of the module ``name``."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(name)
    return Path(spec.origin).read_text(encoding="utf-8")


def is_pinned_module(name: str) -> bool:
    """A ``physverify`` module whose code can change a measurement (not the plumbing packages)."""
    parts = name.split(".")
    if parts[0] != ROOT_PACKAGE or (len(parts) > 1 and parts[1] in UNPINNED_PACKAGES):
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def is_package(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    return spec is not None and spec.submodule_search_locations is not None


def runtime_imports(source: str, package: str) -> Iterator[str]:
    """Every name an import of ``source`` may bind as a module, lazy imports included and
    ``if TYPE_CHECKING:`` blocks left out; ``package`` resolves relative imports."""
    for node in _runtime_nodes(ast.parse(source)):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = importlib.util.resolve_name("." * node.level + base, package)
            yield base
            yield from (f"{base}.{alias.name}" for alias in node.names)


def _runtime_nodes(node: ast.AST) -> Iterator[ast.AST]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If) and _is_type_checking(child.test):
            for other in child.orelse:
                yield other
                yield from _runtime_nodes(other)
            continue
        yield child
        yield from _runtime_nodes(child)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


@dataclass
class SourceGraph:
    """The pinned modules each module reaches, read through ``read`` (memoized per instance)."""

    read: SourceReader = module_source
    _imports: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def closure(self, root: str) -> list[str]:
        """``root`` and every pinned module it imports at run time, with their packages."""
        seen: set[str] = set()
        pending = [root]
        while pending:
            name = pending.pop()
            if name not in seen:
                seen.add(name)
                pending.extend(n for n in self.imports(name) if n not in seen)
        return sorted(seen)

    def imports(self, name: str) -> tuple[str, ...]:
        if name not in self._imports:
            package = name if is_package(name) else name.rpartition(".")[0]
            found = {*_parents(name), *runtime_imports(self.read(name), package)}
            self._imports[name] = tuple(sorted(n for n in found if is_pinned_module(n)))
        return self._imports[name]

    def digest(self, root: str) -> str:
        hasher = hashlib.sha256()
        for name in self.closure(root):
            hasher.update(f"{name}\0{self.read(name)}\0".encode())
        return hasher.hexdigest()


def _parents(name: str) -> Iterator[str]:
    parts = name.split(".")
    yield from (".".join(parts[:end]) for end in range(1, len(parts)))


def pinned_plugins() -> dict[str, Any]:
    """Every pinned plugin class by ``<kind>:<registry key>``."""
    return {
        f"{kind}{KIND_SEPARATOR}{key}": plugin
        for kind in PINNED_KINDS
        for key, plugin in PLUGIN_KINDS[kind]().items()
    }


def current_pins(graph: SourceGraph, plugins: Mapping[str, Any]) -> dict[str, Pin]:
    return {
        name: Pin(str(plugin.version), graph.digest(plugin.__module__))
        for name, plugin in plugins.items()
    }


def compare(current: Mapping[str, Pin], stored: Mapping[str, Pin]) -> PinDiff:
    shared = sorted(set(current) & set(stored))
    return PinDiff(
        missing=tuple(sorted(set(current) - set(stored))),
        unknown=tuple(sorted(set(stored) - set(current))),
        unbumped=tuple(
            n
            for n in shared
            if current[n].version == stored[n].version and current[n].digest != stored[n].digest
        ),
        rerecord=tuple(n for n in shared if current[n].version != stored[n].version),
    )


def load_pins(path: Path = PINS_PATH) -> dict[str, Pin]:
    if not path.is_file():
        return {}
    data: dict[str, dict[str, dict[str, str]]] = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"{kind}{KIND_SEPARATOR}{key}": Pin(**pin)
        for kind, pins in data.items()
        for key, pin in pins.items()
    }


def save_pins(pins: Mapping[str, Pin], path: Path = PINS_PATH) -> None:
    data: dict[str, dict[str, dict[str, str]]] = {}
    for name in sorted(pins):
        kind, _, key = name.partition(KIND_SEPARATOR)
        data.setdefault(kind, {})[key] = {
            "version": pins[name].version,
            "digest": pins[name].digest,
        }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def accepted(
    stored: Mapping[str, Pin], current: Mapping[str, Pin], names: Iterable[str]
) -> dict[str, Pin]:
    """``stored`` with each of ``names`` set to its current pin, or dropped if not registered."""
    pins = dict(stored)
    for name in names:
        if name in current:
            pins[name] = current[name]
        elif pins.pop(name, None) is None:
            raise SystemExit(f"{name}: neither registered nor pinned")
    return pins


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accept", nargs="+", default=[], metavar="KIND:KEY")
    args = parser.parse_args(argv)
    current = current_pins(SourceGraph(), pinned_plugins())
    stored = load_pins()
    if args.accept:
        save_pins(accepted(stored, current, args.accept))
        return 0
    messages = compare(current, stored).messages()
    for message in messages:
        print(message)
    return 1 if messages else 0


if __name__ == "__main__":
    raise SystemExit(main())
