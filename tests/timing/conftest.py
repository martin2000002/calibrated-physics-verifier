"""Timing test fixtures: no network, settings over ``configs/`` with ``tmp_path`` as data, fake
clocks, CUDA probes and memory, a fake dataset, a programmable fake target and fake services.

Fakes are exposed through fixtures (factories) rather than an importable module, so no helper
module name can collide with the other test packages under ``--import-mode=importlib``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import pytest
from network import no_network as no_network
from timing_settings import FAKE_DATASET, make_clip
from timing_settings import clip_factory as clip_factory
from timing_settings import data_dir as data_dir
from timing_settings import make_settings as make_settings
from timing_settings import timing_environ as timing_environ

from physverify.config.models import Settings
from physverify.core.errors import TimingError
from physverify.runtime.heavy import NullLock
from physverify.schema.clip import ClipManifest
from physverify.synth.store import ContainerRunner
from physverify.timing._base import TimingContext
from physverify.timing.hooks import TargetHooks
from physverify.timing.target_services import ClipSelection, TimingServices

FAKE_CLIP_IDS = ("c0", "c1", "c2")

ScriptItem = Mapping[str, float] | BaseException
Script = Sequence[ScriptItem] | Callable[[int, ClipManifest], ScriptItem]


class FakeClock:
    """A clock that only moves when told to."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeCudaProbe:
    is_available: bool = False
    gpus: list[dict[str, Any]] = field(default_factory=list)
    versions_by_name: dict[str, str | None] = field(default_factory=dict)

    def available(self) -> bool:
        return self.is_available

    def devices(self) -> list[dict[str, Any]]:
        return list(self.gpus)

    def versions(self) -> dict[str, str | None]:
        return dict(self.versions_by_name)


class FakeCudaMemory:
    """Returns the scripted peaks in order (the last one repeats); counts resets."""

    def __init__(self, *peaks: float | None) -> None:
        self._peaks = peaks or (None,)
        self._reads = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def peak_mb(self) -> float | None:
        peak = self._peaks[min(self._reads, len(self._peaks) - 1)]
        self._reads += 1
        return peak


class FakeDataset:
    """Three frameless clips; enough for selection, never for decoding."""

    name: ClassVar[str] = FAKE_DATASET

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return [make_clip(clip_id) for clip_id in FAKE_CLIP_IDS]

    def video(self, clip: ClipManifest) -> Any:
        raise NotImplementedError("the fake timing dataset has no frames")

    def ground_truth(self, clip: ClipManifest) -> None:
        return None

    def fetch(self, subset: str | None, dest: Path) -> None:
        return None


@dataclass
class FakeJournal:
    instances: int = 0
    calls: list[str] = field(default_factory=list)
    prepared: list[tuple[str, ...]] = field(default_factory=list)
    closed: int = 0


class FakeTarget(TargetHooks):
    """Call ``n`` answers ``script[n]`` (the last entry repeats) or ``script(n, clip)``; an
    exception entry is raised. Every class made by ``make_fake_target`` has its own journal."""

    name: ClassVar[str] = "fake"
    requires_gpu: ClassVar[bool] = False
    script: ClassVar[Script] = ({"fake.wall_s": 1.0},)
    reason: ClassVar[str | None] = None
    setup: ClassVar[Mapping[str, float]] = {}
    info: ClassVar[Mapping[str, str]] = {}
    journal: ClassVar[FakeJournal] = FakeJournal()

    def __init__(self, services: TimingServices) -> None:
        self.services = services
        self.journal.instances += 1

    def unavailable(self, ctx: TimingContext) -> str | None:
        return self.reason

    def prepare(self, clips: Sequence[ClipManifest], ctx: TimingContext) -> Mapping[str, float]:
        self.journal.prepared.append(tuple(clip.clip_id for clip in clips))
        return dict(self.setup)

    def describe(self, ctx: TimingContext) -> Mapping[str, str]:
        return dict(self.info)

    def close(self) -> None:
        self.journal.closed += 1

    def run(self, clip: ClipManifest, ctx: TimingContext) -> Mapping[str, float]:
        index = len(self.journal.calls)
        self.journal.calls.append(clip.clip_id)
        item = self._answer(index, clip)
        if isinstance(item, BaseException):
            raise item
        return dict(item)

    def _answer(self, index: int, clip: ClipManifest) -> ScriptItem:
        script = type(self).script
        if callable(script):
            return script(index, clip)
        return script[min(index, len(script) - 1)]


@pytest.fixture
def make_fake_target() -> Callable[..., type[FakeTarget]]:
    """``make_fake_target(name, requires_gpu=, script=, unavailable=, own_clips=, setup=,
    info=)``; ``own_clips`` (a clip sequence) makes the class override ``TargetHooks.own_clips``.
    """

    def build(
        name: str = "fake",
        *,
        requires_gpu: bool = False,
        script: Script = ({"fake.wall_s": 1.0},),
        unavailable: str | None = None,
        own_clips: Sequence[ClipManifest] | None = None,
        setup: Mapping[str, float] | None = None,
        info: Mapping[str, str] | None = None,
    ) -> type[FakeTarget]:
        namespace: dict[str, Any] = {
            "name": name,
            "requires_gpu": requires_gpu,
            "script": script,
            "reason": unavailable,
            "setup": dict(setup or {}),
            "info": dict(info or {}),
            "journal": FakeJournal(),
        }
        if own_clips is not None:
            clips = tuple(own_clips)
            namespace["own_clips"] = lambda self, ctx: clips
        return type(f"Fake_{name}", (FakeTarget,), namespace)

    return build


def _no_docker() -> ContainerRunner:
    raise TimingError("timing tests never run docker")


def _no_sync(device: str) -> None:
    return None


def _no_empty_cache() -> None:
    return None


@pytest.fixture
def make_services() -> Callable[..., TimingServices]:
    """Fake services (fake dataset, NullLock, no docker, fake clocks and no CUDA); keyword
    arguments replace fields, e.g. ``make_services(selection=ClipSelection("fake_ds", limit=1))``.
    """

    def build(**changes: Any) -> TimingServices:
        services = TimingServices(
            datasets={FAKE_DATASET: FakeDataset},
            providers={},
            checks={},
            policies={},
            lock_factory=lambda: NullLock("timing test"),
            docker=_no_docker,
            cuda=FakeCudaProbe(),
            selection=ClipSelection(FAKE_DATASET),
            clock=FakeClock(),
            cpu_clock=FakeClock(),
            memory=lambda device: FakeCudaMemory(),
            sync=_no_sync,
            empty_cache=_no_empty_cache,
        )
        return replace(services, **changes)

    return build


@pytest.fixture
def fake_clock() -> Callable[..., FakeClock]:
    return FakeClock


@pytest.fixture
def fake_probe() -> Callable[..., FakeCudaProbe]:
    return FakeCudaProbe


@pytest.fixture
def fake_memory() -> Callable[..., FakeCudaMemory]:
    return FakeCudaMemory
