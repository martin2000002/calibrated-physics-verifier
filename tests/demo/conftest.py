"""Fixtures of the demo: settings over ``configs/``, tiny mp4 clips, a fake verifier stack.

Fakes are fixtures (factories), not an importable module, so no helper name collides with other
test packages under ``--import-mode=importlib``. Nothing here needs a GPU or perception weights.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.core.video import write_mp4
from physverify.datasets.local_video import LocalVideoDataset
from physverify.finetune.candidate_scoring import ScoringDeps
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.verification.inputs import ClipInputs
from physverify.verification.runner import RowStatus, TableRow, VerificationReport

FAIL_SPAN = (2, 4)
N_FRAMES = 8


@pytest.fixture
def demo_environ(environ: dict[str, str], tmp_path: Path) -> dict[str, str]:
    return {**environ, "PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")}


@pytest.fixture
def make_settings(config_dir: Path, demo_environ: dict[str, str]) -> Callable[..., Settings]:
    def build(*overrides: str, profile: str | None = None) -> Settings:
        return load_settings(config_dir, environ=demo_environ, profile=profile, overrides=overrides)

    return build


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(profile="fast")


@pytest.fixture
def make_result() -> Callable[..., CheckResult]:
    def build(
        check_id: str,
        verdict: Verdict,
        magnitude: float | None = 0.0,
        threshold: float | None = 1.0,
        clip_id: str = "p__k00",
        frame_span: tuple[int, int] | None = None,
    ) -> CheckResult:
        abstained = verdict is Verdict.ABSTAIN
        return CheckResult(
            check_id=check_id,
            clip_id=clip_id,
            subject=("o0",),
            frame_span=frame_span,
            magnitude=None if abstained else magnitude,
            sigma=math.nan,
            unit="",
            verdict=verdict,
            input_level=InputLevel.PERCEIVED,
            threshold=None if abstained else threshold,
            abstain_reason=AbstainReason.MISSING_INPUT if abstained else None,
        )

    return build


def _row(clip_id: str, check_id: str, status: RowStatus, verdict: Verdict | None) -> TableRow:
    return TableRow(
        clip_id=clip_id,
        check_id=check_id,
        status=status,
        magnitude=None if verdict in (None, Verdict.ABSTAIN) else 0.5,
        sigma=math.nan,
        verdict=verdict,
        abstain_reason=AbstainReason.MISSING_INPUT if verdict is Verdict.ABSTAIN else None,
        n_subjects=1,
        n_abstained=int(verdict is Verdict.ABSTAIN),
        unit="",
        row_step=0,
    )


@pytest.fixture
def clip_results(make_result: Callable[..., CheckResult]) -> Callable[[str], list[CheckResult]]:
    """R1 PASS, R2 FAIL over FAIL_SPAN, R3a ABSTAIN; R4 had no event (no result)."""

    def build(clip_id: str) -> list[CheckResult]:
        return [
            make_result("R1", Verdict.PASS, 0.2, 1.0, clip_id),
            make_result("R2", Verdict.FAIL, 3.0, 2.0, clip_id, FAIL_SPAN),
            make_result("R3a", Verdict.ABSTAIN, clip_id=clip_id),
        ]

    return build


def clip_rows(clip_id: str) -> list[TableRow]:
    return [
        _row(clip_id, "R1", RowStatus.SCORED, Verdict.PASS),
        _row(clip_id, "R2", RowStatus.SCORED, Verdict.FAIL),
        _row(clip_id, "R3a", RowStatus.ABSTAINED, Verdict.ABSTAIN),
        _row(clip_id, "R4", RowStatus.NO_EVENT, None),
    ]


class FakeSource:
    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for clip in clips:
            yield ClipInputs(clip, None, "no perception in tests", InputLevel.PERCEIVED)


class FakeRunner:
    def __init__(self, results: Callable[[str], list[CheckResult]]) -> None:
        self.results = results
        self.clips: list[str] = []

    def run(self, source: Any, clips: Sequence[ClipManifest]) -> VerificationReport:
        items = list(source.load(clips))
        ids = [item.clip.clip_id for item in items]
        self.clips += ids
        results = tuple(r for cid in ids for r in self.results(cid))
        rows = tuple(row for cid in ids for row in clip_rows(cid))
        return VerificationReport(results, rows, tuple(clips))


@pytest.fixture
def fake_scoring(
    clip_results: Callable[[str], list[CheckResult]],
) -> Callable[[Settings], ScoringDeps]:
    """The real ``local_video`` manifests, a source without perception, fixed verdicts."""

    def build(settings: Settings) -> ScoringDeps:
        runner = FakeRunner(clip_results)
        return ScoringDeps(
            clips=LocalVideoDataset(settings).clips, source=FakeSource(), runner=runner
        )

    return build


@pytest.fixture
def write_video(tmp_path: Path) -> Callable[..., Path]:
    def build(name: str = "clip.mp4", height: int = 96, width: int = 128, shade: int = 90) -> Path:
        frames = np.full((N_FRAMES, height, width, 3), shade, dtype=np.uint8)
        for index, frame in enumerate(frames):
            frame[:, : 8 * (index + 1)] = 200
        path = tmp_path / "videos" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_mp4(frames, path, 8.0)
        return path

    return build
