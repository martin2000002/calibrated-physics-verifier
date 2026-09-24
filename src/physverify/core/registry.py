"""Plugin discovery: one file per plugin, a ``PLUGINS`` tuple, immutable registries.

``discover`` imports every public module of a package (names starting with ``_`` are
skipped), validates each class listed in ``PLUGINS`` against the protocols and returns a
read-only mapping. Heavy libraries must be imported inside ``run()``, never at module level.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType, ModuleType
from typing import Any, TypeVar

from physverify.core.errors import DuplicatePluginError, InvalidPluginError, MissingPluginsError

T = TypeVar("T")
Key = str | Callable[[Any], str]


class Registry(Mapping[str, T]):
    """Read-only name -> plugin mapping."""

    def __init__(self, items: Mapping[str, T]) -> None:
        self._items: Mapping[str, T] = MappingProxyType(dict(items))

    def __getitem__(self, key: str) -> T:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({sorted(self._items)})"


def plugin_modules(package: str) -> list[ModuleType]:
    """Import the public, non-package modules of ``package`` in name order."""
    root = importlib.import_module(package)
    names = sorted(
        info.name
        for info in pkgutil.iter_modules(root.__path__)
        if not info.name.startswith("_") and not info.ispkg
    )
    return [importlib.import_module(f"{package}.{name}") for name in names]


def protocol_members(protocol: type) -> frozenset[str]:
    """Attribute and method names a class must provide to satisfy ``protocol``."""
    members: set[str] = set()
    for base in protocol.__mro__:
        if base is object or not getattr(base, "_is_protocol", False):
            continue
        members.update(name for name in getattr(base, "__annotations__", {}))
        members.update(
            name
            for name, value in vars(base).items()
            if not name.startswith("_") and (callable(value) or isinstance(value, property))
        )
    return frozenset(members)


def satisfies(cls: object, protocol: type) -> bool:
    return inspect.isclass(cls) and all(hasattr(cls, name) for name in protocol_members(protocol))


def _key_of(plugin: Any, key: Key) -> str:
    value = key(plugin) if callable(key) else getattr(plugin, key)
    if not isinstance(value, str) or not value:
        raise InvalidPluginError(f"{plugin!r}: registry key must be a non-empty str")
    return value


def discover(
    package: str,
    protocols: tuple[type, ...],
    key: Key,
    validate: Callable[[Any], None] | None = None,
) -> Registry[Any]:
    """Registry of every class listed in the ``PLUGINS`` tuples of ``package``.

    A class must satisfy at least one of ``protocols``; ``validate`` may add checks and
    raise InvalidPluginError.
    """
    found: dict[str, Any] = {}
    for module in plugin_modules(package):
        for plugin in _plugins_of(module):
            if not any(satisfies(plugin, protocol) for protocol in protocols):
                names = ", ".join(p.__name__ for p in protocols)
                raise InvalidPluginError(f"{module.__name__}: {plugin!r} does not satisfy {names}")
            if validate is not None:
                validate(plugin)
            _add(found, _key_of(plugin, key), plugin, module.__name__)
    return Registry(found)


def discover_modules(
    package: str, check: Callable[[ModuleType], None], key: Callable[[ModuleType], str]
) -> Registry[ModuleType]:
    """Registry of modules that are plugins themselves (CLI commands, container tasks)."""
    found: dict[str, ModuleType] = {}
    for module in plugin_modules(package):
        check(module)
        _add(found, key(module), module, module.__name__)
    return Registry(found)


def _plugins_of(module: ModuleType) -> tuple[Any, ...]:
    if not hasattr(module, "PLUGINS"):
        raise MissingPluginsError(f"{module.__name__} does not define PLUGINS")
    plugins = module.PLUGINS
    if not isinstance(plugins, tuple):
        raise InvalidPluginError(f"{module.__name__}.PLUGINS must be a tuple of classes")
    return plugins


def _add[V](found: dict[str, V], name: str, plugin: V, origin: str) -> None:
    if name in found:
        raise DuplicatePluginError(f"duplicate plugin key {name!r} (again in {origin})")
    found[name] = plugin
