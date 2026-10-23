"""Stage spans from A5's observer hook, CUDA memory without importing torch, measure names."""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from physverify.core.errors import TimingError
from physverify.extraction._base import StageEvent
from physverify.schema.stages import Stage
from physverify.timing.recorder import (
    NoCudaMemory,
    StageRecorder,
    TeeObserver,
    TorchCudaMemory,
    memory_for,
)
from physverify.timing.records import MIB, check_measures, generate_measure, measure_name, unit_of
from physverify.timing.target_services import cuda_sync

HEAVY_MODULES = ("torch", "transformers", "diffusers", "kubric", "bpy")
TIMING_MODULES = ("settings", "records", "hooks", "target_services", "envinfo", "recorder")


def event(
    clip_id: str = "c0",
    stage: Stage = Stage.DETECT,
    *,
    cached: bool = False,
    wall_s: float = math.nan,
    peak_rss_mb: float = math.nan,
) -> StageEvent:
    return StageEvent(clip_id, stage, "fake", cached, "cpu", wall_s, peak_rss_mb)


class Log:
    def __init__(self, name: str, entries: list[tuple[str, str, str]]) -> None:
        self.name = name
        self.entries = entries

    def on_stage_start(self, stage_event: StageEvent) -> None:
        self.entries.append((self.name, "start", stage_event.clip_id))

    def on_stage_end(self, stage_event: StageEvent) -> None:
        self.entries.append((self.name, "end", stage_event.clip_id))


@pytest.fixture
def clocks(fake_clock: Callable[..., Any]) -> tuple[Any, Any]:
    return fake_clock(100.0), fake_clock(7.0)


def test_span_adds_wall_cpu_and_cuda_to_the_event_values(
    clocks: tuple[Any, Any], fake_memory: Callable[..., Any]
) -> None:
    clock, cpu = clocks
    memory = fake_memory(2048.0)
    recorder = StageRecorder(memory, clock, cpu)
    wall_s, peak_rss_mb = 0.1 + 0.2, 1234.5678

    recorder.on_stage_start(event())
    clock.advance(2.5)
    cpu.advance(9.0)
    recorder.on_stage_end(event(wall_s=wall_s, peak_rss_mb=peak_rss_mb))

    (span,) = recorder.spans
    assert span.wall_s == wall_s
    assert span.peak_rss_mb == peak_rss_mb
    assert (span.span_s, span.cpu_s, span.cuda_peak_mb) == (2.5, 9.0, 2048.0)
    assert (span.clip_id, span.stage, span.provider, span.device) == ("c0", "detect", "fake", "cpu")
    assert memory.resets == 1


def test_cached_cells_are_ignored(clocks: tuple[Any, Any], fake_memory: Callable[..., Any]) -> None:
    memory = fake_memory(1.0)
    recorder = StageRecorder(memory, *clocks)

    recorder.on_stage_start(event(cached=True))
    recorder.on_stage_end(event(cached=True))

    assert recorder.spans == ()
    assert memory.resets == 0


def test_a_start_without_end_is_dropped(
    clocks: tuple[Any, Any], fake_memory: Callable[..., Any]
) -> None:
    clock, cpu = clocks
    recorder = StageRecorder(fake_memory(), clock, cpu)

    recorder.on_stage_start(event("c0"))
    clock.advance(5.0)
    recorder.on_stage_start(event("c0"))
    clock.advance(1.0)
    recorder.on_stage_end(event("c0"))
    recorder.on_stage_start(event("c1"))

    assert [(span.clip_id, span.span_s) for span in recorder.spans] == [("c0", 1.0)]


def test_an_end_without_start_is_ignored(
    clocks: tuple[Any, Any], fake_memory: Callable[..., Any]
) -> None:
    recorder = StageRecorder(fake_memory(), *clocks)

    recorder.on_stage_end(event(wall_s=1.0))
    recorder.on_stage_start(event("c0", Stage.DETECT))
    recorder.on_stage_end(event("c0", Stage.TRACK))

    assert recorder.spans == ()


def test_spans_keep_arrival_order(clocks: tuple[Any, Any], fake_memory: Callable[..., Any]) -> None:
    recorder = StageRecorder(fake_memory(), *clocks)

    for stage in (Stage.DETECT, Stage.TRACK, Stage.DEPTH):
        recorder.on_stage_start(event(stage=stage))
        recorder.on_stage_end(event(stage=stage))

    assert [span.stage for span in recorder.spans] == ["detect", "track", "depth"]


def test_tee_forwards_in_order_and_skips_none() -> None:
    entries: list[tuple[str, str, str]] = []
    tee = TeeObserver(Log("a", entries), None, Log("b", entries))

    tee.on_stage_start(event("c0"))
    tee.on_stage_end(event("c0"))

    assert entries == [
        ("a", "start", "c0"),
        ("b", "start", "c0"),
        ("a", "end", "c0"),
        ("b", "end", "c0"),
    ]


def test_memory_for_picks_by_device() -> None:
    assert isinstance(memory_for("cpu"), NoCudaMemory)
    assert isinstance(memory_for("cuda"), TorchCudaMemory)
    assert isinstance(memory_for("cuda:1"), TorchCudaMemory)
    assert NoCudaMemory().peak_mb() is None


def _fake_torch(available: bool) -> Any:
    calls: list[tuple[str, str]] = []
    cuda = SimpleNamespace(
        is_available=lambda: available,
        reset_peak_memory_stats=lambda device: calls.append(("reset", device)),
        max_memory_reserved=lambda device: 3 * MIB,
        synchronize=lambda device: calls.append(("sync", device)),
    )
    return SimpleNamespace(cuda=cuda, calls=calls)


def test_torch_memory_reads_an_already_loaded_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _fake_torch(available=True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    memory = TorchCudaMemory("cuda:1")

    memory.reset()

    assert memory.peak_mb() == 3.0
    assert torch.calls == [("reset", "cuda:1")]


def test_torch_memory_without_cuda_reports_none(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _fake_torch(available=False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    TorchCudaMemory().reset()

    assert TorchCudaMemory().peak_mb() is None
    assert torch.calls == []


def test_cuda_sync_only_on_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _fake_torch(available=True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    cuda_sync("cpu")
    cuda_sync("cuda:0")

    assert torch.calls == [("sync", "cuda:0")]


def test_timing_modules_never_import_heavy_libraries() -> None:
    imports = "; ".join(f"import physverify.timing.{module}" for module in TIMING_MODULES)
    code = (
        f"import sys; {imports}; "
        "from physverify.timing.recorder import TorchCudaMemory; "
        "m = TorchCudaMemory(); m.reset(); assert m.peak_mb() is None; "
        "from physverify.timing.target_services import cuda_sync, cuda_empty_cache; "
        "cuda_sync('cuda'); cuda_empty_cache(); "
        f"print(sorted(set({HEAVY_MODULES!r}) & set(sys.modules)))"
    )

    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


@pytest.mark.parametrize(
    ("measure", "unit"),
    [("extract.wall_s", "s"), ("extract.peak_rss_mb", "MB"), ("verify.r1_penetration.wall_s", "s")],
)
def test_unit_follows_the_suffix(measure: str, unit: str) -> None:
    assert unit_of(measure) == unit


@pytest.mark.parametrize(
    "measure", ["extract.wall", "Extract.wall_s", "extract..wall_s", "wall_ms"]
)
def test_invalid_measure_names_are_rejected(measure: str) -> None:
    with pytest.raises(TimingError, match="invalid measure name"):
        unit_of(measure)


def test_measure_builders() -> None:
    assert measure_name("extract", "depth", "wall_s") == "extract.depth.wall_s"
    assert generate_measure(4, "per_video_s") == "generate.b4.per_video_s"
    with pytest.raises(TimingError):
        measure_name("extract", "Depth", "wall_s")


def test_check_measures_accepts_non_negative_reals() -> None:
    check_measures({"a_s": 0.0, "b_s": 2, "c_mb": np.float32(1.5)})


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0, True, "1.0", None], ids=repr)
def test_check_measures_rejects_bad_values(value: Any) -> None:
    with pytest.raises(TimingError, match="a_s"):
        check_measures({"a_s": value})
