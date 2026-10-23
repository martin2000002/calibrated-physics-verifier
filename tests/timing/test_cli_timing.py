"""``physverify timing`` end to end over fake targets, a fake dataset and a fake CUDA probe.

The numbers the fakes return are scripted; nothing here times real work.
"""

from __future__ import annotations

import json
import signal
from collections.abc import Callable, Mapping
from pathlib import Path
from types import FrameType, MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from cli_runner import CliResult, CliRunner

from physverify.app import plugins, timing
from physverify.extraction.weights import ALLOW_PATTERNS, hub_cache
from physverify.generation import wan21
from physverify.schema.clip import ClipManifest
from physverify.timing.records import (
    EXTRACT_SPAN_S,
    GENERATE_LOAD_S,
    VERIFY_WALL_S,
    TargetStatus,
    generate_measure,
)
from physverify.timing.report import SUMMARY, TOTAL_LABEL
from physverify.timing.results_io import RESULTS, SAMPLES, STATS, load_results
from physverify.timing.settings import load_timing
from physverify.timing.target_services import TimingServices

FAKE_DATASET = "fake_ds"
SCRIPTED_COST = {generate_measure(1, "per_video_s"): 10.0, EXTRACT_SPAN_S: 4.0, VERIFY_WALL_S: 1.0}
QUICK = ("--warmup", "0", "--repetitions", "2")
FILES = (RESULTS, SAMPLES, STATS, SUMMARY)

Install = Callable[..., None]
TimingCli = Callable[..., CliResult]


@pytest.fixture
def install(
    monkeypatch: pytest.MonkeyPatch, make_services: Callable[..., TimingServices]
) -> Install:
    """Registries with only the given timing targets and the fake dataset."""
    dataset = make_services().datasets[FAKE_DATASET]

    def register(*targets: type[Any]) -> None:
        registry = {cls.name: cls for cls in targets}
        patched: Mapping[str, Callable[[], Mapping[str, Any]]] = {
            **plugins.PLUGIN_KINDS,
            "timing_targets": lambda: registry,
            "datasets": lambda: {FAKE_DATASET: dataset},
            "providers": dict,
            "checks": dict,
            "policies": dict,
        }
        monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))

    return register


@pytest.fixture(autouse=True)
def default_targets(install: Install, make_fake_target: Callable[..., type[Any]]) -> None:
    install(
        make_fake_target("cpu_a"),
        make_fake_target(
            "gpu_a", requires_gpu=True, script=(SCRIPTED_COST,), setup={GENERATE_LOAD_S: 30.0}
        ),
        make_fake_target("broken", script=(RuntimeError("boom"),)),
    )


@pytest.fixture(autouse=True)
def probe(monkeypatch: pytest.MonkeyPatch, fake_probe: Callable[..., Any]) -> Any:
    instance = fake_probe()
    monkeypatch.setattr(timing, "TorchCudaProbe", lambda: instance)
    return instance


@pytest.fixture
def timing_cli(cli: CliRunner, data_dir: Path) -> TimingCli:
    def invoke(*argv: str) -> CliResult:
        return cli(*argv, env={"PHYSVERIFY__PATHS__DATA_DIR": str(data_dir)})

    return invoke


def run_dir_of(output: str) -> Path:
    (line,) = [line for line in output.splitlines() if line.startswith("run: ")]
    return Path(line.removeprefix("run: "))


def test_the_command_has_help(timing_cli: TimingCli) -> None:
    result = timing_cli("timing", "--help")
    assert result.code == 0
    for command in ("list", "env", "fetch", "run", "summarize", "budget"):
        assert command in result.out


def test_list_prints_every_registered_target(timing_cli: TimingCli) -> None:
    result = timing_cli("timing", "list")
    assert result.code == 0
    rows = [line.split("\t") for line in result.out.splitlines()]
    assert rows[0][:2] == ["name", "requires_gpu"]
    assert [(row[0], row[1]) for row in rows[1:]] == [
        ("broken", "False"),
        ("cpu_a", "False"),
        ("gpu_a", "True"),
    ]


def test_env_reports_why_a_cpu_run_is_not_a_measurement(timing_cli: TimingCli) -> None:
    result = timing_cli("timing", "env")
    assert result.code == 0
    report = json.loads(result.out)
    assert report["valid_measurement"] is False
    assert "device is 'cpu', not cuda" in report["reasons"]


def test_a_harness_check_writes_the_files_and_prints_no_numbers(timing_cli: TimingCli) -> None:
    result = timing_cli("timing", "run", "--target", "cpu_a", "--dataset", FAKE_DATASET, *QUICK)
    assert result.code == 0, result.err
    run_dir = run_dir_of(result.out)
    directory = run_dir / "timing"
    assert {path.name for path in directory.iterdir()} >= set(FILES)
    lines = result.out.splitlines()
    assert lines[1].startswith("> HARNESS CHECK ONLY")
    assert lines[2] == f"cpu_a: {TargetStatus.MEASURED.value}"
    assert lines[3:] == [f"{name}: {directory / name}" for name in FILES]
    assert TOTAL_LABEL not in result.out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ("timing", "run", "--target", "gpu_a", "--dataset", FAKE_DATASET),
            "requires --device cuda",
        ),
        (
            ("--device", "cuda", "timing", "run", "--target", "cpu_a", "--dataset", FAKE_DATASET),
            "CUDA is not available",
        ),
        (("timing", "run", "--target", "nope", "--dataset", FAKE_DATASET), "unknown timing target"),
        (("timing", "run", "--target", "cpu_a"), "--dataset is required"),
        (("timing", "run", "--target", "cpu_a", "--dataset", "nope"), "nope"),
    ],
)
def test_usage_errors_exit_2_before_any_run_exists(
    timing_cli: TimingCli, data_dir: Path, argv: tuple[str, ...], message: str
) -> None:
    result = timing_cli(*argv)
    assert result.code == 2
    assert message in result.out + result.err
    assert not list(data_dir.rglob("run.json"))


def test_the_selection_is_read_only_when_a_target_needs_it(
    timing_cli: TimingCli,
    install: Install,
    make_fake_target: Callable[..., type[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Like wan21_generate, an earlier target can create the clips that the next one selects."""
    produced: list[str] = []
    fake = plugins.PLUGIN_KINDS["datasets"]()[FAKE_DATASET]

    def late_clips(self: Any, subset: str | None = None) -> list[ClipManifest]:
        return list(fake.clips(self, subset)) if produced else []

    def produce(n: int, clip: ClipManifest) -> dict[str, float]:
        produced.append(clip.clip_id)
        return {"fake.wall_s": 1.0}

    own = fake.clips(None)[:1]
    install(make_fake_target("producer", script=produce, own_clips=own), make_fake_target("cpu_a"))
    late = type("LateDataset", (fake,), {"clips": late_clips})
    kinds = {**plugins.PLUGIN_KINDS, "datasets": lambda: {FAKE_DATASET: late}}
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(kinds))
    argv = ("--target", "producer", "--target", "cpu_a", "--dataset", FAKE_DATASET, *QUICK)
    result = timing_cli("timing", "run", *argv)
    assert result.code == 0, result.out + result.err
    targets = load_results(run_dir_of(result.out) / "timing").targets
    assert [(t.name, t.status) for t in targets] == [
        ("producer", TargetStatus.MEASURED),
        ("cpu_a", TargetStatus.MEASURED),
    ]
    assert len(targets[1].clip_ids) == len(fake.clips(None))


def test_a_failing_target_exits_1_and_keeps_its_record(timing_cli: TimingCli) -> None:
    result = timing_cli("timing", "run", "--target", "broken", "--dataset", FAKE_DATASET, *QUICK)
    assert result.code == 1
    (target,) = load_results(run_dir_of(result.out) / "timing").targets
    assert target.status is TargetStatus.FAILED
    assert target.failures and "RuntimeError: boom" in target.failures[0].error


def test_summarize_rebuilds_from_the_run_or_its_timing_folder(timing_cli: TimingCli) -> None:
    ran = timing_cli("timing", "run", "--target", "cpu_a", "--dataset", FAKE_DATASET, *QUICK)
    run_dir = run_dir_of(ran.out)
    stats = (run_dir / "timing" / STATS).read_bytes()
    for folder in (run_dir, run_dir / "timing"):
        (run_dir / "timing" / STATS).unlink()
        (run_dir / "timing" / SUMMARY).unlink()
        result = timing_cli("timing", "summarize", "--run", str(folder))
        assert result.code == 0, result.err
        assert result.out.startswith("> HARNESS CHECK ONLY")
        assert (run_dir / "timing" / STATS).read_bytes() == stats
        assert (run_dir / "timing" / SUMMARY).is_file()
    missing = timing_cli("timing", "summarize", "--run", str(run_dir.parent))
    assert missing.code == 2


def test_budget_refuses_a_harness_check(timing_cli: TimingCli) -> None:
    ran = timing_cli("timing", "run", "--target", "cpu_a", "--dataset", FAKE_DATASET, *QUICK)
    budget = ("timing", "budget", "--run", str(run_dir_of(ran.out)))
    result = timing_cli(*budget, "--gpu-hours", "10", "--candidates", "8")
    assert result.code == 2
    assert "valid H200 measurement" in result.out + result.err


def test_a_valid_run_prints_the_total_and_feeds_the_budget(
    timing_cli: TimingCli, probe: Any
) -> None:
    probe.is_available, probe.gpus = True, [{"name": "NVIDIA H200"}]
    cuda = ("--device", "cuda", "--profile", "precise")
    ran = timing_cli(*cuda, "timing", "run", "--target", "gpu_a", "--dataset", FAKE_DATASET, *QUICK)
    assert ran.code == 0, ran.err
    assert "HARNESS CHECK" not in ran.out
    assert f"{TOTAL_LABEL}: 15 s" in ran.out
    budget = ("timing", "budget", "--run", str(run_dir_of(ran.out)))
    result = timing_cli(*budget, "--gpu-hours", "10", "--candidates", "8")
    assert result.code == 0, result.err
    assert "training prompts 300; preference pairs 300-8400" in result.out
    assert "generator load L_g: 30 s once per job" in result.out
    assert "omitted (pass --extract-batch" in result.out
    bad = timing_cli(*budget, "--gpu-hours", "0", "--candidates", "8")
    assert bad.code == 2


def test_sigterm_closes_the_run_and_restores_the_handler(
    timing_cli: TimingCli, install: Install, make_fake_target: Callable[..., type[Any]]
) -> None:
    def script(n: int, clip: ClipManifest) -> dict[str, float]:
        if n == 1:
            signal.raise_signal(signal.SIGTERM)
        return {"fake.wall_s": 1.0}

    target = make_fake_target("term", script=script)
    install(target)
    caught: list[int] = []

    def guard(signum: int, frame: FrameType | None) -> None:
        caught.append(signum)

    previous = signal.signal(signal.SIGTERM, guard)
    try:
        result = timing_cli("timing", "run", "--target", "term", "--dataset", FAKE_DATASET, *QUICK)
        assert signal.getsignal(signal.SIGTERM) is guard
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert not caught
    assert result.code == 1
    assert "interrupted" in result.err
    (record,) = load_results(run_dir_of(result.out) / "timing").targets
    assert record.status is TargetStatus.PARTIAL
    assert record.reason is not None and "interrupted" in record.reason
    assert target.journal.closed == 1


def test_fetch_snapshots_before_verifying_so_a_cut_download_resumes(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Any]
) -> None:
    spec = SimpleNamespace(ref="org/wan/model_index.json", revision="rev1")
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(wan21, "wan_weight_spec", lambda gen: spec)

    def verify(weight: Any, paths: Any, downloader: Any) -> Path:
        calls.append(("fetch", weight))
        return Path("w")

    monkeypatch.setattr(wan21, "fetch", verify)

    class Downloader:
        def hf_snapshot(self, repo: str, revision: str, cache: Path, allow: Any) -> Path:
            calls.append(("snapshot", repo, revision, cache, tuple(allow)))
            return cache

        def http_file(self, url: str, dest: Path) -> None:
            raise AssertionError("not used by the generator")

        def git_checkout(self, url: str, revision: str, dest: Path) -> None:
            raise AssertionError("not used by the generator")

    settings = make_settings()
    paths, gen = settings.paths, load_timing(settings).generation
    assert wan21.fetch_generator(gen, paths, Downloader()) == Path("w")
    assert calls == [
        ("snapshot", "org/wan", "rev1", hub_cache(paths), (*ALLOW_PATTERNS, "model_index.json")),
        ("fetch", spec),
    ]
