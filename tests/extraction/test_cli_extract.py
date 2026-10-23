"""AC16: ``physverify extract run`` over the fixture with heavy fakes (``--help`` is in AC19)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from cli_runner import CliRunner
from fakes import FAKE, FAKES, FakeDepth
from support import plan_overrides
from synthetic_clip import CLIP_ID, SyntheticFixtureDataset

from physverify.app import plugins
from physverify.cli.commands import extract
from physverify.core.errors import PluginError
from physverify.extraction._base import StageContext, provider_key
from physverify.runtime.heavy import holder_path
from physverify.schema.stages import STAGE_ORDER

FAKE_PLAN = [
    arg for item in plan_overrides(detect=FAKE, track=FAKE, depth=FAKE) for arg in ("--set", item)
]
RUN = ("extract", "run", "--dataset", "synthetic_fixture", "--gt", "camera")


@pytest.fixture(autouse=True)
def fixture_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real registries plus the heavy fakes and the fixture dataset."""
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    providers, datasets = kinds["providers"], kinds["datasets"]
    patched = {
        **kinds,
        "providers": lambda: {**providers(), **{provider_key(f): f for f in FAKES}},
        "datasets": lambda: {**datasets(), SyntheticFixtureDataset.name: SyntheticFixtureDataset},
    }
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))


def events_of(output: str) -> list[dict[str, Any]]:
    (line,) = [line for line in output.splitlines() if line.startswith("run: ")]
    run_dir = Path(line.removeprefix("run: "))
    lines = (run_dir / extract.EVENTS_FILE).read_text().splitlines()
    return [json.loads(line) for line in lines]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("extract", "run", "--dataset", "nope"), "unknown dataset"),
        ((*RUN, "--clips", "nope"), "unknown clip ids"),
    ],
)
def test_unknown_dataset_or_clip_is_a_usage_error(
    cli: CliRunner, argv: tuple[str, ...], message: str
) -> None:
    result = cli(*argv)
    assert result.code == 2
    assert message in result.err


def test_a_fixture_run_succeeds_and_its_rerun_never_takes_the_lock(
    cli: CliRunner, lock_path: Path
) -> None:
    first = cli(*FAKE_PLAN, *RUN)
    assert first.code == 0, first.err
    events = events_of(first.out)
    assert [e["stage"] for e in events] == [s.value for s in STAGE_ORDER]
    assert not any(e["cached"] for e in events)
    holder_path(lock_path).unlink(missing_ok=True)
    again = cli(*FAKE_PLAN, *RUN, "--clips", CLIP_ID)
    assert again.code == 0, again.err
    assert all(e["cached"] for e in events_of(again.out))
    assert not holder_path(lock_path).exists()


def test_a_failed_clip_exits_with_one(cli: CliRunner) -> None:
    fail = f"extraction.providers.depth.fake.fail_clip={CLIP_ID}"
    result = cli(*FAKE_PLAN, "--set", fail, *RUN)
    assert result.code == 1
    assert f"FAILED {CLIP_ID} at depth" in result.out
    failure = events_of(result.out)[-1]
    assert (failure["clip_id"], failure["stage"]) == (CLIP_ID, "depth")
    assert "fails on" in failure["error"]


def test_an_aborted_run_still_writes_the_finished_events(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(self: FakeDepth, ctx: StageContext) -> None:
        raise PluginError("boom")

    monkeypatch.setattr(FakeDepth, "run", explode)
    result = cli(*FAKE_PLAN, *RUN)
    assert result.code == 1
    assert "boom" in result.err
    events = events_of(result.out)
    assert [e["stage"] for e in events] == ["detect", "track", "depth"]
    assert (events[-1]["clip_id"], events[-1]["error"]) == (CLIP_ID, "aborted: PluginError: boom")
