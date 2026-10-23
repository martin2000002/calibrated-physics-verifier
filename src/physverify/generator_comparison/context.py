"""The settings and stores of a generator comparison campaign, and what they say about its
prompts, arms and gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from physverify.config.models import Settings
from physverify.core.choices import choice
from physverify.core.errors import ConfigError, UsageError
from physverify.datasets._videophy_tables import LABELS_DIR, read_table
from physverify.datasets.videophy import FETCH_HINT, VideophyDataset
from physverify.finetune.layout import SELECTED_FILE, CampaignStore
from physverify.finetune.records import BASE_TAG, read_record
from physverify.finetune.selection import Selection
from physverify.finetune.settings import FinetuneSettings, load_finetune
from physverify.generator_comparison.layout import ComparisonLayout
from physverify.generator_comparison.prereg import ComparisonPreregistration
from physverify.generator_comparison.records import GateReport
from physverify.generator_comparison.settings import ComparisonSettings, load_comparison

ARMS: Final = ("base", "ft")


@dataclass(frozen=True)
class ComparisonContext:
    settings: Settings
    cfg: ComparisonSettings
    finetune: FinetuneSettings
    store: CampaignStore
    layout: ComparisonLayout

    @classmethod
    def of(cls, settings: Settings) -> ComparisonContext:
        store = CampaignStore.from_settings(settings)
        return cls(
            settings,
            load_comparison(settings),
            load_finetune(settings),
            store,
            ComparisonLayout.of(store.root),
        )


def selection(h: ComparisonContext) -> Selection:
    path = h.store.select_dir(h.cfg.selection_run) / SELECTED_FILE
    if not path.is_file():
        raise UsageError(f"{path} does not exist: run 'finetune select'")
    return read_record(path, Selection)


def arm_of(h: ComparisonContext, arm: str) -> tuple[str, Path | None]:
    """(model tag, adapter) of an arm; ft is the checkpoint chosen by ``finetune select``."""
    if choice(arm, ARMS, "--arm") == "base":
        return BASE_TAG, None
    chosen = selection(h).selected
    if chosen is None:
        raise UsageError("no checkpoint passed selection, so there is no ft arm")
    return chosen.model_tag, Path(chosen.adapter)


def reference_path(h: ComparisonContext) -> Path:
    dataset = VideophyDataset(h.settings)
    part = dataset.settings.parts.get(h.cfg.gate.part)
    if part is None or part.reference is None:
        raise ConfigError(f"[videophy.parts.{h.cfg.gate.part}] has no reference table")
    return h.settings.paths.raw_dir / VideophyDataset.name / LABELS_DIR / part.reference.file


def reference_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise UsageError(f"{path} is missing; run `{FETCH_HINT}`")
    return read_table(path)


def arm_tags(prereg: ComparisonPreregistration) -> list[str]:
    return [BASE_TAG] if prereg.adapter_tag is None else [BASE_TAG, prereg.adapter_tag]


def stored_gate(h: ComparisonContext) -> GateReport | None:
    return read_record(h.layout.gate, GateReport) if h.layout.gate.is_file() else None
