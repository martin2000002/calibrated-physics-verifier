"""VideoPhy label tables: settings, CSV rows, the lexical object-object rule and
the mapping from a video URL to its member in the published zip.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import urllib.parse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from physverify.core.errors import ConfigError, UsageError
from physverify.core.models import FrozenModel
from physverify.datasets._base import LabelledVideo
from physverify.schema.clip import Split

SECTION = "videophy"
SMOKE = "smoke"
LABELS_DIR = "labels"
VIDEOS_DIR = "videos"
CLIP_ID_DIGITS = 12
TRUNCATION_MARK = ";"
TOKEN = re.compile(r"[a-z0-9]+")
LEXICAL_EXCLUSIONS = ("human_agent", "nonrigid", "fracture", "fluid")
NO_RIGID_EVENT = "no_interaction"
SPLITS = {"test": Split.EVAL, "dev": Split.CALIB}


class HubFile(FrozenModel):
    repo: str
    revision: str
    file: str
    sha256: str | None = None
    prefix: str = ""

    def url(self, hub_url: str) -> str:
        return f"{hub_url}/{self.repo}/resolve/{self.revision}/{urllib.parse.quote(self.file)}"


class PartSettings(FrozenModel):
    """One published table: (version, split) with its label, reference and video files."""

    version: Literal["v1", "v2"]
    split: Literal["test", "dev"]
    labels: HubFile
    videos: HubFile
    reference: HubFile | None = None
    human_column: str
    human_cut: float
    reference_column: str = ""
    reference_cut: float = 0.0
    reference_name: str = ""
    generator_column: str = ""
    scope_column: str
    scope_value: str
    rules_column: str = ""


class FilterSettings(FrozenModel):
    rigid_event: tuple[str, ...]
    human_agent: tuple[str, ...]
    nonrigid: tuple[str, ...]
    fracture: tuple[str, ...]
    fluid: tuple[str, ...]

    def terms(self, name: str) -> tuple[str, ...]:
        terms: tuple[str, ...] = getattr(self, name)
        return terms


class VideophySettings(FrozenModel):
    hub_url: str
    license: str
    retries: int = Field(ge=0)
    max_frames: int = Field(ge=0)
    filter_version: int = Field(ge=1)
    subset_file: str
    queries_file: str
    reviewed_subsets: tuple[str, ...]
    smoke_ids: tuple[str, ...]
    subsets: dict[str, tuple[str, ...]]
    filter: FilterSettings
    parts: dict[str, PartSettings]

    def parts_of(self, subset: str) -> tuple[str, ...]:
        """Part names of a subset; ``smoke`` is drawn from the development parts."""
        name = "h2_dev" if subset == SMOKE else subset
        if name not in self.subsets:
            known = ", ".join([*sorted(self.subsets), SMOKE])
            raise UsageError(f"unknown videophy subset {subset!r}; known: {known}")
        missing = [p for p in self.subsets[name] if p not in self.parts]
        if missing:
            raise ConfigError(f"[videophy.subsets] {name} names unknown parts: {missing}")
        return self.subsets[name]


VideophyRow = LabelledVideo


def clip_id_of(version: str, video_url: str) -> str:
    digest = hashlib.sha1(video_url.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{version}-{digest[:CLIP_ID_DIGITS]}"


def to_float(text: str | None) -> float:
    try:
        return float(text) if text not in (None, "") else math.nan
    except ValueError:
        return math.nan


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generator_of(record: Mapping[str, str], column: str) -> str:
    """The table's generator column, else the first directory of the URL path."""
    if column:
        return record.get(column, "") or ""
    return urllib.parse.urlsplit(record["video_url"]).path.strip("/").split("/")[0]


def tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN.findall(text.lower()))


def has_term(words: Sequence[str], term: str) -> bool:
    """A whole token, or a contiguous token sequence for a multi-word term."""
    wanted = tokens(term)
    width = len(wanted)
    return width > 0 and any(
        tuple(words[i : i + width]) == wanted for i in range(len(words) - width + 1)
    )


def lexical_reason(caption: str, rules: FilterSettings) -> str | None:
    """None when the caption passes the object-object rule, else why it is out."""
    words = tokens(caption)
    if not any(has_term(words, t) for t in rules.rigid_event):
        return NO_RIGID_EVENT
    for name in LEXICAL_EXCLUSIONS:
        if any(has_term(words, t) for t in rules.terms(name)):
            return name
    return None


def in_scope(record: Mapping[str, str], part: PartSettings) -> bool:
    return record.get(part.scope_column, "") == part.scope_value


def build_rows(
    name: str,
    part: PartSettings,
    records: Iterable[Mapping[str, str]],
    reference: Mapping[str, float],
) -> list[VideophyRow]:
    return [
        VideophyRow(
            part=name,
            version=part.version,
            split=SPLITS[part.split],
            clip_id=clip_id_of(part.version, r["video_url"]),
            video_url=r["video_url"],
            caption=r["caption"],
            generator=generator_of(r, part.generator_column),
            human_raw=to_float(r.get(part.human_column)),
            human_cut=part.human_cut,
            reference_raw=reference.get(r["video_url"], math.nan),
            reference_cut=part.reference_cut,
            reference_name=part.reference_name,
            violated_rules=r.get(part.rules_column, "") if part.rules_column else "",
        )
        for r in records
    ]


def reference_scores(records: Iterable[Mapping[str, str]], column: str) -> dict[str, float]:
    return {r["video_url"]: to_float(r.get(column)) for r in records}


def url_member(video_url: str, prefix: str) -> str:
    return prefix + urllib.parse.urlsplit(video_url).path.lstrip("/")


def member_map(urls: Iterable[str], names: Iterable[str], prefix: str) -> dict[str, str]:
    """``{url: member}``: the exact path, else the path cut at its first ``;`` when the zip
    holds that cut name and no other URL cuts to it. URLs absent here are fetched directly."""
    present = set(names)
    wanted = {url: url_member(url, prefix) for url in urls}
    cut = {url: path.split(TRUNCATION_MARK, 1)[0] for url, path in wanted.items()}
    shared = Counter(cut.values())
    found: dict[str, str] = {}
    for url, path in wanted.items():
        if path in present:
            found[url] = path
        elif TRUNCATION_MARK in path and cut[url] in present and shared[cut[url]] == 1:
            found[url] = cut[url]
    return found
