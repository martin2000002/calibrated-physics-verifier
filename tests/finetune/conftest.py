"""Fixtures of the preference cycle: settings over ``configs/``, tiny fakes, record factories.

Fakes are fixtures (factories), not an importable module, so no helper name collides with other
test packages under ``--import-mode=importlib``. Nothing here needs a GPU, diffusers or peft.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.finetune.records import CandidateScore
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable, PrimitiveKind

FAKE_LABELS = frozenset(
    {
        "cube",
        "ball",
        "cylinder",
        "cup",
        "bottle",
        "book",
        "shoe",
        "cereal_box",
        "tennis_ball",
        "basketball",
        "soccer_ball",
    }
)


@pytest.fixture
def fake_labels() -> frozenset[str]:
    return FAKE_LABELS


@pytest.fixture
def ft_environ(environ: dict[str, str], tmp_path: Path) -> dict[str, str]:
    return {**environ, "PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")}


@pytest.fixture
def make_settings(config_dir: Path, ft_environ: dict[str, str]) -> Callable[..., Settings]:
    """``make_settings("section.key=value", ..., profile=None)`` over the committed configs."""

    def build(*overrides: str, profile: str | None = None) -> Settings:
        return load_settings(config_dir, environ=ft_environ, profile=profile, overrides=overrides)

    return build


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(profile="precise")


@pytest.fixture
def make_states() -> Callable[..., ObjectStateTable]:
    """States of T frames x N objects from ``position (T, N, 3)`` and ``size (T, N, 3)``."""

    def build(
        position: np.ndarray,
        size: np.ndarray,
        labels: Sequence[str],
        visible: np.ndarray | None = None,
    ) -> ObjectStateTable:
        n_t, n_obj = position.shape[:2]
        table = ObjectStateTable.blank(
            clip_id="clip",
            frame=CoordFrame.CAMERA_OPENCV,
            unit=LengthUnit.RELATIVE,
            source_key="lift-key",
            t=np.arange(n_t, dtype=float),
            frame_index=np.arange(n_t),
            object_ids=[f"o{i}" for i in range(n_obj)],
            labels=list(labels),
            primitive=[PrimitiveKind.UNKNOWN.value] * n_obj,
        )
        shown = np.ones((n_t, n_obj), dtype=bool) if visible is None else visible
        return dataclasses.replace(
            table,
            position=np.asarray(position, dtype=float),
            size=np.asarray(size, dtype=float),
            visible=shown,
        )

    return build


@pytest.fixture
def make_result() -> Callable[..., CheckResult]:
    def build(
        check_id: str,
        verdict: Verdict,
        magnitude: float | None = 0.0,
        threshold: float | None = 1.0,
        sigma: float = math.nan,
        clip_id: str = "p__k00",
    ) -> CheckResult:
        return CheckResult(
            check_id=check_id,
            clip_id=clip_id,
            subject=("o0",),
            frame_span=None,
            magnitude=magnitude,
            sigma=sigma,
            unit="",
            verdict=verdict,
            input_level=InputLevel.PERCEIVED,
            threshold=threshold,
            abstain_reason=AbstainReason.MISSING_INPUT if verdict is Verdict.ABSTAIN else None,
        )

    return build


@pytest.fixture
def make_score() -> Callable[..., CandidateScore]:
    def build(candidate_id: str, score: float | None, **update: Any) -> CandidateScore:
        fields: dict[str, Any] = {
            "scorer": "provisional",
            "scorer_version": "1",
            "calibrated": False,
            "thresholds": "t",
            "candidate_id": candidate_id,
            "prompt_id": candidate_id.split("__")[0],
            "model_tag": "base",
            "score": score,
            "sigma": None,
            "abstained": score is None,
            "reason": None,
            "n_decided": 2,
            "n_failed": 0,
            "coverage": 1.0,
            "decided": ["R1", "R2"],
            "per_check": {},
            "motion": 2.0,
            "objects_ok": True,
            "results_path": "",
        }
        return CandidateScore(**{**fields, **update})

    return build
