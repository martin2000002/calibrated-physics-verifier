from __future__ import annotations

from pathlib import Path

from physverify.finetune.records import BASE_TAG, write_record
from physverify.generator_comparison.layout import ComparisonLayout
from physverify.generator_comparison.prereg_state import (
    DEV_SPLIT,
    contaminated,
    dev_pc,
    dev_usability,
)
from physverify.generator_comparison.records import TASKS, JudgeOutput, JudgeRow
from physverify.generator_comparison.settings import ComparisonSettings

STAMPS = {task: f"stamp-{task}" for task in TASKS}


def _write_dev(layout: ComparisonLayout, stamps: dict[str, str]) -> None:
    for task in TASKS:
        rows = {f"d{i}__k00": JudgeRow(value=float(1 + i % 4), reason=None) for i in range(8)}
        output = JudgeOutput(
            split=DEV_SPLIT,
            model_tag=BASE_TAG,
            task=task,
            stamp=stamps[task],
            scale=(1.0, 5.0),
            prereg_sha256=None,
            rows=rows,
        )
        path = layout.judge_output(DEV_SPLIT, BASE_TAG, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_record(path, output)


def test_dev_usability_needs_outputs_under_the_current_stamps(
    tmp_path: Path, comparison_cfg: ComparisonSettings
) -> None:
    layout = ComparisonLayout.of(tmp_path)
    assert dev_usability(layout, [BASE_TAG], STAMPS, comparison_cfg) is None
    _write_dev(layout, STAMPS)
    found = dev_usability(layout, [BASE_TAG], STAMPS, comparison_cfg)
    assert found is not None and found.usable
    _write_dev(layout, {**STAMPS, "sa": "older-judge"})
    assert dev_usability(layout, [BASE_TAG], STAMPS, comparison_cfg) is None


def test_dev_pc_skips_outputs_of_another_pc_stamp(tmp_path: Path) -> None:
    layout = ComparisonLayout.of(tmp_path)
    assert dev_pc(layout, [BASE_TAG], STAMPS["pc"]) == {}
    _write_dev(layout, STAMPS)
    found = dev_pc(layout, [BASE_TAG], STAMPS["pc"])
    assert len(found) == 8 and found[(BASE_TAG, "d0__k00")] == 1.0
    assert dev_pc(layout, [BASE_TAG], "older-judge") == {}


def test_contamination_is_unknown_without_a_dataset_record() -> None:
    assert contaminated(None) is None
    assert contaminated({"name": "kubric", "split": "calib"}) is False
    assert contaminated({"name": "local_video", "split": "calib"}) is True
    assert contaminated({"name": "kubric", "split": "eval"}) is True
