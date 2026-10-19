"""Layered configuration (lowest to highest precedence):

model defaults < configs/*.toml < active profile ``[<section>.profiles.<name>]`` <
configs/local.toml < environment ``PHYSVERIFY__<SECTION>__<KEY>`` < --device / --profile <
--set section.key=value.

Every top-level table lives in exactly one ``configs/*.toml`` file (``local*.toml`` excluded).
"""

from __future__ import annotations

import copy
import string
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from physverify.config.models import PROFILE_CONTEXT, PathsSettings, RuntimeSettings, Settings
from physverify.core.errors import ConfigError

ENV_PREFIX = "PHYSVERIFY__"
DEFAULT_PROFILE = "fast"
PROFILES_KEY = "profiles"
LOCAL_FILE = "local.toml"

Table = dict[str, Any]


def load_settings(
    config_dir: Path,
    *,
    environ: Mapping[str, str],
    profile: str | None = None,
    device: str | None = None,
    overrides: Sequence[str] = (),
) -> Settings:
    config_dir = config_dir.resolve()
    if not config_dir.is_dir():
        raise ConfigError(f"config directory not found: {config_dir}")
    base, sources = read_config_files(config_dir)
    local_path = config_dir / LOCAL_FILE
    local = _read_toml(local_path) if local_path.is_file() else {}
    if local:
        sources.append(str(local_path))
    upper = [local, env_layer(environ), device_layer(device), set_layer(overrides)]
    name = resolve_profile(profile, base, upper)
    merged = merge(apply_profile(base, name), *(strip_profiles(layer) for layer in upper))
    sources.extend(_describe_overrides(environ, device, overrides))
    return _build(merged, name, config_dir, environ, tuple(sources))


def read_config_files(config_dir: Path) -> tuple[Table, list[str]]:
    """Merge ``*.toml`` (not ``local*.toml``); a table repeated across files is an error."""
    merged: Table = {}
    owner: dict[str, str] = {}
    files = sorted(p for p in config_dir.glob("*.toml") if not p.name.startswith("local"))
    for path in files:
        for key, value in _read_toml(path).items():
            if key in owner:
                raise ConfigError(f"table [{key}] is defined in both {owner[key]} and {path.name}")
            owner[key] = path.name
            merged[key] = value
    return merged, [str(p) for p in files]


def _read_toml(path: Path) -> Table:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def parse_value(text: str) -> Any:
    """A TOML value (number, bool, string, array, inline table), else the raw string."""
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def _nested(keys: Sequence[str], value: Any) -> Table:
    table: Table = {keys[-1]: value}
    for key in reversed(keys[:-1]):
        table = {key: table}
    return table


def env_layer(environ: Mapping[str, str]) -> Table:
    layers: list[Table] = []
    for name in sorted(environ):
        if not name.startswith(ENV_PREFIX):
            continue
        keys = [part.lower() for part in name[len(ENV_PREFIX) :].split("__")]
        if len(keys) < 2 or not all(keys):
            raise ConfigError(f"{name}: expected {ENV_PREFIX}<SECTION>__<KEY>")
        layers.append(_nested(keys, parse_value(environ[name])))
    return merge(*layers)


def device_layer(device: str | None) -> Table:
    return {} if device is None else {"runtime": {"device": device}}


def set_layer(overrides: Sequence[str]) -> Table:
    layers: list[Table] = []
    for item in overrides:
        path, sep, text = item.partition("=")
        keys = path.strip().split(".")
        if not sep or len(keys) < 2 or not all(keys):
            raise ConfigError(f"--set expects section.key=value, got {item!r}")
        layers.append(_nested(keys, parse_value(text.strip())))
    return merge(*layers)


def merge(*layers: Mapping[str, Any]) -> Table:
    """Deep merge; later layers win, tables merge key by key."""
    result: Table = {}
    for layer in layers:
        for key, value in layer.items():
            current = result.get(key)
            if isinstance(current, dict) and isinstance(value, Mapping):
                result[key] = merge(current, value)
            else:
                result[key] = copy.deepcopy(value)
    return result


def known_profiles(base: Mapping[str, Any]) -> frozenset[str]:
    names = {DEFAULT_PROFILE}
    for section in base.values():
        if isinstance(section, Mapping) and isinstance(section.get(PROFILES_KEY), Mapping):
            names.update(section[PROFILES_KEY])
    return frozenset(names)


def resolve_profile(
    requested: str | None, base: Mapping[str, Any], upper: Sequence[Mapping[str, Any]]
) -> str:
    layered = merge(base, *upper).get("runtime", {})
    name = requested or layered.get("default_profile") or DEFAULT_PROFILE
    if name not in known_profiles(base):
        raise ConfigError(f"unknown profile {name!r}; known: {sorted(known_profiles(base))}")
    return str(name)


def apply_profile(base: Mapping[str, Any], name: str) -> Table:
    result: Table = {}
    for key, section in base.items():
        if isinstance(section, Mapping):
            profiles = section.get(PROFILES_KEY, {})
            body = {k: v for k, v in section.items() if k != PROFILES_KEY}
            result[key] = merge(body, profiles.get(name, {}))
        else:
            result[key] = copy.deepcopy(section)
    return result


def strip_profiles(layer: Mapping[str, Any]) -> Table:
    return {
        key: {k: v for k, v in value.items() if k != PROFILES_KEY}
        if isinstance(value, Mapping)
        else value
        for key, value in layer.items()
    }


def _describe_overrides(
    environ: Mapping[str, str], device: str | None, overrides: Sequence[str]
) -> list[str]:
    described = [f"env:{name}" for name in sorted(environ) if name.startswith(ENV_PREFIX)]
    if device is not None:
        described.append(f"--device {device}")
    described.extend(f"--set {item}" for item in overrides)
    return described


def expand(value: str, environ: Mapping[str, str]) -> str:
    try:
        return string.Template(value).substitute(environ)
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"cannot expand {value!r}: missing variable {exc}") from exc


def _expanded(table: Mapping[str, Any], environ: Mapping[str, str]) -> Table:
    return {k: expand(v, environ) if isinstance(v, str) else v for k, v in table.items()}


def _build(
    merged: Table,
    profile: str,
    config_dir: Path,
    environ: Mapping[str, str],
    sources: tuple[str, ...],
) -> Settings:
    repo_root = config_dir.parent
    home = Path(environ.get("HOME") or Path.home())
    try:
        paths = PathsSettings.model_validate(
            _expanded(merged.get("paths", {}), environ), context={PROFILE_CONTEXT: profile}
        )
        runtime = RuntimeSettings.model_validate(_expanded(merged.get("runtime", {}), environ))
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc
    lock = runtime.heavy_lock
    if lock and not Path(lock).is_absolute():
        runtime = runtime.model_copy(update={"heavy_lock": str(repo_root / lock)})
    sections = {k: v for k, v in merged.items() if k not in ("paths", "runtime")}
    return Settings(
        paths=paths.anchored(repo_root, home),
        runtime=runtime,
        profile=profile,
        config_dir=config_dir,
        repo_root=repo_root,
        sections=MappingProxyType(sections),
        sources=sources,
    )
