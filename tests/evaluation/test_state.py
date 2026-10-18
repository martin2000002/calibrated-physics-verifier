from __future__ import annotations

from pathlib import Path

from physverify.evaluation.layout import H3Layout
from physverify.evaluation.records import TASKS, JudgeOutput, JudgeRow
from physverify.evaluation.settings import H3Settings
from physverify.evaluation.state import DEV_SPLIT, dev_pc, dev_usability
from physverify.finetune.records import BASE_TAG, write_record

STAMPS = {task: f"stamp-{task}" for task in TASKS}


def _write_dev(layout: H3Layout, stamps: dict[str, str]) -> None:
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
    tmp_path: Path, h3_cfg: H3Settings
) -> None:
    layout = H3Layout.of(tmp_path)
    assert dev_usability(layout, [BASE_TAG], STAMPS, h3_cfg) is None
    _write_dev(layout, STAMPS)
    found = dev_usability(layout, [BASE_TAG], STAMPS, h3_cfg)
    assert found is not None and found.usable
    _write_dev(layout, {**STAMPS, "sa": "older-judge"})
    assert dev_usability(layout, [BASE_TAG], STAMPS, h3_cfg) is None


def test_dev_pc_skips_outputs_of_another_pc_stamp(tmp_path: Path) -> None:
    layout = H3Layout.of(tmp_path)
    assert dev_pc(layout, [BASE_TAG], STAMPS["pc"]) == {}
    _write_dev(layout, STAMPS)
    found = dev_pc(layout, [BASE_TAG], STAMPS["pc"])
    assert len(found) == 8 and found[(BASE_TAG, "d0__k00")] == 1.0
    assert dev_pc(layout, [BASE_TAG], "older-judge") == {}
