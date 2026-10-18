"""Whether the external judge may be used (ADR 0013 D4): the regression gate on published
VideoPhy-2 test videos and the usability test on our dev videos. Pure: values in, verdicts out."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from physverify.datasets._videophy_tables import VideophyRow
from physverify.evaluation.judges._base import JudgeScore
from physverify.evaluation.records import GateTask, TaskUsability, Usability
from physverify.evaluation.settings import GateSettings, UsabilitySettings


def gate_rows(
    rows: Sequence[VideophyRow], present: Callable[[VideophyRow], bool], n: int
) -> list[VideophyRow]:
    """The first ``n`` rows by clip id with a published reference and a video on disk."""
    kept = [r for r in rows if math.isfinite(r.reference_raw) and present(r)]
    return sorted(kept, key=lambda r: r.clip_id)[:n]


def gate_task(
    task: str,
    stamp: str,
    judged: Mapping[str, JudgeScore],
    reference: Mapping[str, float],
    cfg: GateSettings,
) -> GateTask:
    """Exact matches count a missing value as a mismatch; the MAE is over the parsed rows."""
    ids = sorted(reference)
    values = {i: judged[i].value if i in judged else None for i in ids}
    parsed = {i: v for i, v in values.items() if v is not None}
    exact = sum(v == reference[i] for i, v in parsed.items())
    mae = float(np.mean([abs(v - reference[i]) for i, v in parsed.items()])) if parsed else None
    failures = len(ids) - len(parsed)
    passed = (
        bool(ids)
        and exact >= cfg.min_exact * len(ids)
        and mae is not None
        and mae <= cfg.max_mae
        and failures <= cfg.max_parse_fail
    )
    return GateTask(
        task=task,
        stamp=stamp,
        n=len(ids),
        exact=exact,
        exact_share=exact / len(ids) if ids else 0.0,
        mae=mae,
        parse_failures=failures,
        passed=passed,
    )


def task_usability(values: Sequence[float | None], cfg: UsabilitySettings) -> TaskUsability:
    """Unusable when it fails to read too many videos or cannot tell them apart."""
    parsed = [v for v in values if v is not None]
    fail_share = 1 - len(parsed) / len(values) if values else 1.0
    modal = Counter(parsed).most_common(1)[0][1] / len(parsed) if parsed else None
    sd = float(np.std(parsed)) if parsed else None
    usable = (
        fail_share <= cfg.max_parse_fail_share
        and modal is not None
        and modal < cfg.max_modal_share
        and sd is not None
        and sd > 0
    )
    return TaskUsability(
        n=len(values), parse_fail_share=fail_share, modal_share=modal, sd=sd, usable=usable
    )


def usability(values: Mapping[str, Sequence[float | None]], cfg: UsabilitySettings) -> Usability:
    tasks = {task: task_usability(v, cfg) for task, v in sorted(values.items())}
    return Usability(tasks=tasks, usable=bool(tasks) and all(t.usable for t in tasks.values()))
