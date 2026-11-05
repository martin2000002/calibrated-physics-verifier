"""The dry run of a learned variant's samples and training: counts only, the train split only,
the leakage guard before anything is counted."""

from dataclasses import replace
from typing import Any, ClassVar

import pytest
from records import manifest, perturbation

from physverify.app import learned
from physverify.checks.variants._learned import LeakageError, dense_layout
from physverify.core.errors import UsageError
from physverify.extraction.pair_features import dense_columns
from physverify.learned.arms import ARM_STAGES
from physverify.schema.clip import ClipManifest, Split

SEED = int(perturbation().seed)


def clips() -> tuple[ClipManifest, ...]:
    base = replace(manifest(), split=Split.TRAIN)
    clean = replace(base, clip_id="clean", scene_id="s1", perturbation=None)
    return (replace(base, clip_id="positive"), clean)


class _Check:
    TRACK_COLUMNS: ClassVar[tuple[str, ...]] = ("a", "b")


class _Runner:
    def outputs(self, clip: ClipManifest, variant: object) -> None:
        return None


def plan(monkeypatch: pytest.MonkeyPatch, chosen: tuple[ClipManifest, ...], **kw: Any) -> Any:
    monkeypatch.setattr(learned, "_selected", lambda app, sel, command, err: (None, chosen))
    monkeypatch.setattr(learned, "learned_check", lambda app, check_id: _Check())
    monkeypatch.setattr(learned, "cached_runner", lambda app, dataset: _Runner())
    evaluated = kw.get("evaluated", ())
    return learned.plan_training(None, ("d", "s"), "R1", (0, 0.5), evaluated, err=print)  # type: ignore[arg-type]


def test_the_plan_counts_clips_by_arm_and_role_without_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = plan(monkeypatch, clips(), evaluated=[frozenset({SEED + 1})])
    assert found.clips == 2 and sum(found.arms.values()) == 2
    assert all(len(name) == 2 + len(ARM_STAGES) for name in found.arms)
    assert found.roles == {"clean": 1, "other": 1}  # the positive targets R5, not R1
    assert sum(found.validation_roles.values()) <= 2
    assert found.without_inputs == ("positive", "clean") and found.leakage_checked == 1
    assert found.dense_columns == len(dense_layout(_Check.TRACK_COLUMNS, dense_columns()))
    assert set(found.controls) == {"G-L0", "G-L1", "C-dec", "C-inv", "C-scr", "C-hdri"}


def test_the_plan_runs_the_leakage_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LeakageError):
        plan(monkeypatch, clips(), evaluated=[frozenset({SEED})])


@pytest.mark.parametrize("split", [Split.CALIB, Split.EVAL])
def test_training_reads_the_train_split_only(monkeypatch: pytest.MonkeyPatch, split: Split) -> None:
    mixed = (*clips(), replace(manifest(), clip_id="other", split=split))
    with pytest.raises(UsageError, match="not split=train"):
        learned.training_clips(mixed)
    with pytest.raises(UsageError, match="not split=train"):
        plan(monkeypatch, mixed)
