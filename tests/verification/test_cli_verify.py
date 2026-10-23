"""AC14: ``physverify verify`` over the fixture clip, without the heavy lock."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from cli_runner import CliResult, CliRunner
from synthetic_clip import SyntheticFixtureDataset

from physverify.cli import context
from physverify.runtime.heavy import holder_path
from physverify.verification.runner import CHECKS_FILE, TABLE_FILE, RowStatus

DATASET = ("--dataset", SyntheticFixtureDataset.name)
EXIT_USAGE = 2


@pytest.fixture(autouse=True)
def fixture_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    kinds: Mapping[str, Any] = context.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {
        **kinds,
        "datasets": lambda: {**datasets(), SyntheticFixtureDataset.name: SyntheticFixtureDataset},
    }
    monkeypatch.setattr(context, "PLUGIN_KINDS", MappingProxyType(patched))


def run_dir(result: CliResult) -> Path:
    (line,) = [line for line in result.out.splitlines() if line.startswith("run: ")]
    return Path(line.removeprefix("run: "))


def table(result: CliResult) -> list[dict[str, str]]:
    with (run_dir(result) / TABLE_FILE).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_the_oracle_run_exits_zero_and_writes_the_run(cli: CliRunner, lock_path: Path) -> None:
    result = cli("verify", *DATASET, "--level", "oracle")
    assert result.code == 0, result.err
    folder = run_dir(result)
    assert {"run.json", CHECKS_FILE, TABLE_FILE} <= {p.name for p in folder.iterdir()}
    rows = table(result)
    assert len(rows) == 9
    assert {row["status"] for row in rows} == {RowStatus.SCORED.value, RowStatus.NO_EVENT.value}
    assert "thresholds: provisional" in result.out
    assert not holder_path(lock_path).exists()


def test_a_perceived_run_never_takes_the_lock(cli: CliRunner, lock_path: Path) -> None:
    result = cli("verify", *DATASET, "--level", "perceived", "--policy", "raw")
    assert result.code == 0, result.err
    assert {row["status"] for row in table(result)} == {RowStatus.ABSTAINED.value}
    assert not holder_path(lock_path).exists()


@pytest.mark.parametrize(
    "argv",
    [
        ("--level", "perceived", "--sweep"),
        ("--level", "oracle", "--gt", "depth"),
        (),
    ],
    ids=["sweep_perceived", "gt_on_oracle", "no_level"],
)
def test_usage_errors(cli: CliRunner, argv: tuple[str, ...]) -> None:
    assert cli("verify", *DATASET, *argv).code == EXIT_USAGE


def test_an_unknown_dataset_is_a_usage_error(cli: CliRunner) -> None:
    assert cli("verify", "--dataset", "nope", "--level", "oracle").code == EXIT_USAGE
