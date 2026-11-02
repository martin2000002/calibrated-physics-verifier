"""Every reader of the provisional threshold table applies the unit and checks_hash guard."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.app.context import AppContext, GlobalOptions, Invocation
from physverify.app.plugins import PLUGIN_KINDS
from physverify.app.selective_risk import _session
from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.human_agreement.rule_setup import build_rule
from physverify.human_agreement.settings import agreement_settings
from physverify.verification.runner import configured_suite

MISMATCH = r"R5a \(tau in accel_wald_chi2_3, check reports accel_rel"
V1 = "check_variants.R5a=default"


def suite(settings: Settings, _: Path) -> object:
    return configured_suite(
        settings, plugins.configured("checks", settings), PLUGIN_KINDS["policies"]()
    )


def agreement_rule(settings: Settings, _: Path) -> object:
    agreement = agreement_settings(settings).model_copy(update={"rule": "provisional_any_fail"})
    registry = plugins.configured("checks", settings)
    return build_rule(settings, registry, agreement, None, None)  # type: ignore[arg-type]


def selective_risk(_: Settings, config_dir: Path) -> object:
    options = GlobalOptions(config_dir=config_dir, profile="precise", overrides=(V1,))
    context = AppContext(Invocation(argv=("physverify",), environ={}), options)
    return _session(context, None, None, provisional=True)


def scorer(settings: Settings, _: Path) -> object:
    return ProvisionalScorer(settings, plugins.configured("checks", settings)).thresholds()


@pytest.mark.parametrize("reader", [suite, agreement_rule, selective_risk, scorer])
def test_every_provisional_reader_refuses_a_tau_in_another_unit(
    make_settings: SettingsFactory, config_dir: Path, reader: Callable[[Settings, Path], object]
) -> None:
    with pytest.raises(ConfigError, match=MISMATCH):
        reader(make_settings(V1, profile="precise"), config_dir)


@pytest.mark.parametrize("flag", ["oracle", "perceived"])
def test_an_explicit_selective_risk_table_of_other_checks_is_refused(
    config_dir: Path, tmp_path: Path, flag: str
) -> None:
    table = tmp_path / "foreign.json"
    table.write_text(
        '{"status": "calibrated", "source": "elsewhere", "checks_hash": "' + "0" * 64 + '",'
        ' "checks": {"R1": {"tau": 0.7}}}',
        encoding="utf-8",
    )
    options = GlobalOptions(config_dir=config_dir, profile="precise")
    context = AppContext(Invocation(argv=("physverify",), environ={}), options)
    tables = {"oracle": None, "perceived": None} | {flag: table}
    with pytest.raises(ConfigError, match="calibrated with other check versions"):
        _session(context, tables["oracle"], tables["perceived"], provisional=True)
