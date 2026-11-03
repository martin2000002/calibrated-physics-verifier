"""The selective-risk study reads the calibration chain's population and k, declared once."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from records import manifest
from settings_factory import SettingsFactory

from physverify.calibration.settings import calibration_settings
from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.schema.clip import ClipManifest, Split
from physverify.selective_risk.settings import (
    SECTION,
    AbstentionSettings,
    Population,
    population,
)
from physverify.selective_risk.tables import (
    Session,
    collect_real,
    collect_synthetic,
    named_ks,
    split_clips,
)


def test_the_population_is_the_calibration_section(make_settings: SettingsFactory) -> None:
    settings = make_settings(
        "calibration.dataset=other", "calibration.real_datasets=['physics_iq']", profile="fast"
    )
    expected = Population("other", calibration_settings(settings).subset, ("physics_iq",))
    assert population(settings) == expected
    assert expected.subset == "local"


@pytest.mark.parametrize("key", ["dataset", "calib_split", "eval_split", "real_datasets"])
def test_the_abstention_section_declares_no_population(
    make_settings: SettingsFactory, key: str
) -> None:
    settings = make_settings(f"abstention.{key}='x'")
    with pytest.raises(ConfigError):
        settings.section(SECTION, AbstentionSettings)


class _Planned:
    """A subset whose clips sit in both splits, as a planned Kubric subset does."""

    def __init__(self, clips: Sequence[ClipManifest]) -> None:
        self._clips = clips
        self.asked: list[str | None] = []

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        self.asked.append(subset)
        return self._clips


def test_one_split_of_the_subset_is_read() -> None:
    calib = manifest()
    held = replace(calib, clip_id="e0", split=Split.EVAL)
    source = _Planned((calib, held))
    assert split_clips(source, "full", Split.EVAL) == [held]  # type: ignore[arg-type]
    assert split_clips(source, "full", Split.CALIB) == [calib]  # type: ignore[arg-type]
    assert source.asked == ["full", "full"]


class _NoOutputs:
    """A chain whose every clip has a failed stage, so the clips read are the failed ones; with
    no check its routing has nothing to refuse."""

    checks: dict[str, Any] = {}

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def settings_of(self, variant: object) -> Settings:
        return self._settings

    def resolver(self, variant: object) -> Callable[[Settings], None]:
        return lambda settings: None

    def outputs(self, clip: ClipManifest, variant: object) -> None:
        return None


def test_the_study_reads_one_split_of_the_calibration_population(
    make_settings: SettingsFactory,
) -> None:
    calib = manifest()
    held = replace(calib, clip_id="e0", split=Split.EVAL)
    source, asked = _Planned((calib, held)), []

    def chain(dataset: str) -> tuple[Any, Any]:
        asked.append(dataset)
        return _NoOutputs(make_settings()), source

    session = Session(
        chain=chain,
        calibration_dir=Path(),
        settings=AbstentionSettings(),
        population=Population("synthetic", "full", ()),
        suite=None,  # type: ignore[arg-type]
        reference_table=None,  # type: ignore[arg-type]
        thresholds=None,  # type: ignore[arg-type]
        min_span_iou=0.5,
        checks=(),
        couplings=frozenset(),
        configured_k=2.0,
    )
    collected = collect_synthetic(session, Split.EVAL)
    assert (asked, source.asked) == (["synthetic"], ["full"])
    assert (list(collected.clips), collected.failed) == (["e0"], ["e0"])


class _Unroutable(_NoOutputs):
    def resolver(self, variant: object) -> Callable[[Settings], None]:
        def refuse(settings: Settings) -> None:
            raise ConfigError("unroutable")

        return refuse


@pytest.mark.parametrize("real", [False, True])
def test_a_refused_routing_stops_the_study_instead_of_failing_each_clip(
    make_settings: SettingsFactory, real: bool
) -> None:
    source = _Planned((manifest(),))
    session = Session(
        chain=lambda dataset: (_Unroutable(make_settings()), source),
        calibration_dir=Path(),
        settings=AbstentionSettings(),
        population=Population("synthetic", "full", ()),
        suite=None,  # type: ignore[arg-type]
        reference_table=None,  # type: ignore[arg-type]
        thresholds=None,  # type: ignore[arg-type]
        min_span_iou=0.5,
        checks=(),
        couplings=frozenset(),
        configured_k=2.0,
    )
    with pytest.raises(ConfigError, match="unroutable"):
        collect_real(session, "real") if real else collect_synthetic(session, Split.CALIB)
    assert source.asked == []


def test_the_configured_k_is_reported_next_to_k_star() -> None:
    assert named_ks(None, 2.5) == {"k_2": 2.5}
    assert list(named_ks(1.7, 2.5).items()) == [("k_star", 1.7), ("k_2", 2.5)]
