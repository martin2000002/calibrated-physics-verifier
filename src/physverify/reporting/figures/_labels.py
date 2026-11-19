"""The visible strings of the figures: English defaults that a TOML file may replace.

Sections: ``[format]`` (``decimal``), ``[common]`` and one per figure name, whose keys must be
known, and the open naming sections (``[checks]``, ``[datasets]``, ...), where a missing entry
shows the identifier itself.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from physverify.core.errors import ConfigError
from physverify.schema.checks import InputLevel

COMMON = "common"
FORMAT = "format"
DECIMAL = "decimal"
DECIMALS = frozenset({".", ","})
NAMES = ("checks", "datasets", "stages", "units", "judges")
MINUS = "\N{MINUS SIGN}"

COMMON_TEXTS: Mapping[str, str] = MappingProxyType(
    {level.value: f"{level.value.capitalize()} inputs" for level in InputLevel}
)


@dataclass(frozen=True)
class Text:
    """The strings of one figure (its own, then the common ones) and its number format."""

    own: Mapping[str, str]
    common: Mapping[str, str]
    names: Mapping[str, Mapping[str, str]]
    decimal: str = "."

    def __call__(self, key: str, **values: object) -> str:
        template = self.own[key] if key in self.own else self.common[key]
        return template.format(**values) if values else template

    def name(self, section: str, identifier: str) -> str:
        return self.names.get(section, {}).get(identifier, identifier)

    def number(self, value: float, digits: int = 2) -> str:
        return self._signed(f"{value:.{digits}f}")

    def tick(self, value: float, significant: int = 6) -> str:
        """A tick label: the shortest form of ``value`` to ``significant`` digits."""
        return self._signed(f"{value:.{significant}g}")

    def _signed(self, text: str) -> str:
        return text.replace(".", self.decimal).replace("-", MINUS)


@dataclass(frozen=True)
class Labels:
    texts: Mapping[str, Mapping[str, str]]
    """Figure name (or ``common``) -> key -> string."""
    names: Mapping[str, Mapping[str, str]]
    decimal: str = "."

    def scope(self, figure: str) -> Text:
        return Text(self.texts.get(figure, {}), self.texts[COMMON], self.names, self.decimal)


def load_labels(path: Path | None, defaults: Mapping[str, Mapping[str, str]]) -> Labels:
    """``defaults``: the English strings per figure name and ``common``; the TOML at ``path``
    replaces any of them. An unknown section or key is a ConfigError."""
    found = {} if path is None else _read(path)
    where = str(path)
    unknown = sorted(set(found) - {FORMAT, *NAMES, *defaults})
    if unknown:
        raise ConfigError(f"{where}: unknown label sections {unknown}")
    texts = {
        section: {**strings, **_known(found.get(section, {}), strings, f"{where} [{section}]")}
        for section, strings in defaults.items()
    }
    names = {section: _strings(found.get(section, {}), f"{where} [{section}]") for section in NAMES}
    return Labels(texts, names, _decimal(found.get(FORMAT, {}), f"{where} [{FORMAT}]"))


def _read(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"unreadable labels file {path}: {error}") from error


def _strings(table: object, where: str) -> dict[str, str]:
    if not isinstance(table, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in table.items()
    ):
        raise ConfigError(f"{where}: every entry must be a string")
    return dict(table)


def _known(table: object, defaults: Mapping[str, str], where: str) -> dict[str, str]:
    strings = _strings(table, where)
    unknown = sorted(set(strings) - set(defaults))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}; known: {sorted(defaults)}")
    return strings


def _decimal(table: object, where: str) -> str:
    strings = _known(table, {DECIMAL: "."}, where)
    decimal = strings.get(DECIMAL, ".")
    if decimal not in DECIMALS:
        raise ConfigError(f"{where}: decimal must be one of {sorted(DECIMALS)}")
    return decimal
