"""Human-agreement fixtures: miniature VideoPhy tables under the real file names, a recorded
review and fake perceived verify runs. No network, no videos (``compare`` only reads tables and
runs).

Shared by ``tests/human_agreement`` and ``tests/scripts/human_agreement``; a conftest activates
the fixtures for its directory by importing them."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.datasets._videophy_review import write_review
from physverify.datasets.videophy import VideophyDataset
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.verification.runner import (
    THRESHOLDS_KEY,
    RowStatus,
    TableRow,
    VerificationReport,
    write_report,
)

BALL = "A ball rolls across the table."
BOOK = "A book falls onto the floor."
BOX = "A box slides down a ramp."
CUBE = "A cube falls onto a table."
CAN = "A can rolls off a desk."
V1_TEST = {
    "videophy_test_public.csv": [
        ("v1t/a1", BALL, "gen1", "1", "solid_solid"),
        ("v1t/a2", BALL, "gen1", "0", "solid_solid"),
        ("v1t/b1", BOOK, "gen2", "0", "solid_solid"),
        ("v1t/b2", BOOK, "gen2", "1", "solid_solid"),
        ("v1t/b3", BOOK, "gen2", "", "solid_solid"),
        ("v1t/c1", "A man drops a cup.", "gen1", "0", "solid_solid"),
        ("v1t/d1", "A cup sits on a shelf.", "gen1", "1", "solid_solid"),
        ("v1t/e1", BALL, "gen1", "1", "solid_fluid"),
    ],
}
V1_TEST_HEADER = ("video_url", "caption", "source", "majority_pc", "states_of_matter")
V1_REFERENCE = [("v1t/a1", "0.8"), ("v1t/a2", "0.3"), ("v1t/b1", "0.2"), ("v1t/b2", "0.6")]
V2_TEST = [
    ("v2t/f1", BOX, "gen3", "5", "Object Interactions", ""),
    ("v2t/f2", BOX, "gen3", "2", "Object Interactions", "the box passes through the ramp"),
    ("v2t/f3", BOX, "gen3", "4", "Object Interactions", ""),
]
V2_REFERENCE = [("v2t/f1", "4.5"), ("v2t/f2", "3.0")]
V1_DEV = [
    ("v1d/g1", CUBE, "gen1", "1", "solid_solid"),
    ("v1d/g2", CUBE, "gen1", "0", "solid_solid"),
    ("v1d/g3", CUBE, "gen2", "0", "solid_solid"),
]
V2_DEV = [("v2d/gen4/h1", CAN, "5", "Object Interactions", "")]
URL = "https://videos.example.org/"


def url(path: str) -> str:
    return URL + path + ".mp4"


def _write(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows([(url(r[0]), *r[1:]) for r in rows])


def write_tables(labels: Path) -> None:
    _write(labels / "videophy_test_public.csv", V1_TEST_HEADER, V1_TEST["videophy_test_public.csv"])
    _write(labels / "videophy_autoeval_scores.csv", ("video_url", "pc"), V1_REFERENCE)
    v2 = ("video_url", "caption", "model_name", "pc", "category", "human_violated_rules")
    _write(labels / "videophy2_test.csv", v2, V2_TEST)
    _write(labels / "videophy2_autoeval_scores.csv", ("video_url", "pc_pred"), V2_REFERENCE)
    _write(
        labels / "train.csv",
        ("video_url", "caption", "source", "physics", "states_of_matter"),
        V1_DEV,
    )
    v2d = ("video_url", "caption", "pc", "category", "human_violated_rules")
    _write(labels / "videophy2_training.csv", v2d, V2_DEV)


PROVISIONAL_ENV = {"PHYSVERIFY__AGREEMENT__RULE": "provisional_any_fail"}


@pytest.fixture(autouse=True)
def provisional_by_default(environ: dict[str, str]) -> None:
    """The committed rule is ``calibrated``; these tests start from the provisional one, and a
    test that wants the calibrated rule passes ``--set agreement.rule=...`` (higher precedence)."""
    environ.update(PROVISIONAL_ENV)


SettingsFactory = Callable[..., Settings]


@pytest.fixture
def make_settings(config_dir: Path, environ: dict[str, str]) -> SettingsFactory:
    def build(*overrides: str) -> Settings:
        return load_settings(
            config_dir, environ=environ, overrides=("agreement.bootstrap_n=50", *overrides)
        )

    return build


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    built = make_settings()
    write_tables(built.paths.raw_dir / VideophyDataset.name / "labels")
    return built


@pytest.fixture
def dataset(settings: Settings) -> VideophyDataset:
    return VideophyDataset(settings)


def keep_all(dataset: VideophyDataset) -> None:
    """Record a review that keeps every candidate caption of the reviewed test parts."""
    from physverify.datasets._videophy_review import record
    from physverify.datasets._videophy_subset import audit, reviewed_candidates

    pending = audit(dataset, "h2_test")
    decisions = {p: dict.fromkeys(t, "keep") for p, t in pending.decisions.items()}
    filled = pending.model_copy(update={"decisions": decisions, "date": "2026-10-17"})
    write_review(dataset.subset_path, record(filled, reviewed_candidates(dataset, "h2_test")))


@dataclass(frozen=True)
class Row:
    """One check row of a fake run: FAIL/PASS rows are SCORED, the rest abstain or have no event."""

    check_id: str
    verdict: Verdict | None
    magnitude: float | None = None
    status: RowStatus = RowStatus.SCORED
    reason: AbstainReason | None = None
    sigma: float = 0.01


def fake_results(clip_id: str, rows: Sequence[Row], label: str) -> list[CheckResult]:
    return [
        CheckResult(
            check_id=r.check_id,
            clip_id=clip_id,
            subject=(),
            frame_span=None,
            magnitude=r.magnitude,
            sigma=r.sigma,
            unit="m",
            verdict=r.verdict or Verdict.ABSTAIN,
            input_level=InputLevel.PERCEIVED,
            abstain_reason=r.reason,
            provenance={THRESHOLDS_KEY: label, "detect": "fake"},
        )
        for r in rows
        if r.status not in (RowStatus.NO_EVENT, RowStatus.ERROR)
    ]


def fake_table(clip_id: str, rows: Sequence[Row]) -> list[TableRow]:
    return [
        TableRow(
            clip_id=clip_id,
            check_id=r.check_id,
            status=r.status,
            magnitude=r.magnitude,
            sigma=r.sigma if r.magnitude is not None else math.nan,
            verdict=r.verdict,
            abstain_reason=r.reason,
            n_subjects=1,
            n_abstained=int(r.status is RowStatus.ABSTAINED),
            unit="m",
            row_step=1,
        )
        for r in rows
    ]


def write_run(
    run_dir: Path,
    clips: Mapping[str, Sequence[Row]],
    profile: str = "precise",
    label: str = "provisional",
) -> Path:
    run_dir.mkdir(parents=True)
    results = [r for c, rows in clips.items() for r in fake_results(c, rows, label)]
    table = [r for c, rows in clips.items() for r in fake_table(c, rows)]
    write_report(run_dir, VerificationReport(tuple(results), tuple(table), ()))
    run = {"profile": profile, "device": "cuda", "commit": "abc"}
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return run_dir


FAIL = Row("R1", Verdict.FAIL, 0.10)
PASS = Row("R1", Verdict.PASS, 0.01)
NO_EVENT = Row("R3a", None, None, RowStatus.NO_EVENT)
CROSSING = Row("R1", None, 0.04, RowStatus.ABSTAINED, AbstainReason.INTERVAL_CROSSES_THRESHOLD)


class World:
    """What the tests need from this module (``--import-mode=importlib`` cannot import it)."""

    Row = Row
    FAIL, PASS, NO_EVENT, CROSSING = FAIL, PASS, NO_EVENT, CROSSING
    BALL, BOOK, BOX, CUBE, CAN = BALL, BOOK, BOX, CUBE, CAN

    def __init__(self, settings: Settings, dataset: VideophyDataset, root: Path) -> None:
        self.settings, self.dataset, self.root = settings, dataset, root

    @staticmethod
    def url(path: str) -> str:
        return url(path)

    def clip(self, path: str) -> str:
        """The clip id of a fixture video path such as ``v1t/a1``."""
        rows = [
            r for s in ("h2_test", "h2_dev") for p in self.dataset.candidates(s).values() for r in p
        ]
        return {r.video_url: r.clip_id for r in rows}[url(path)]

    def keep_all(self) -> None:
        keep_all(self.dataset)

    def run(self, name: str, clips: Mapping[str, Sequence[Row]], **kwargs: Any) -> Path:
        """A fake verify run; ``clips`` keys are fixture video paths."""
        by_id = {self.clip(path): rows for path, rows in clips.items()}
        return write_run(self.root / "runs" / name, by_id, **kwargs)


@pytest.fixture
def world(settings: Settings, dataset: VideophyDataset, tmp_path: Path) -> World:
    return World(settings, dataset, tmp_path)
