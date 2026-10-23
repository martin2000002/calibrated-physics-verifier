"""The ``compare-generators status`` and ``compare-generators freeze`` lines of a campaign."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Final

from physverify.finetune.stop_rule import fired_reason
from physverify.generator_comparison.candidate_queries import eval_records, eval_scores
from physverify.generator_comparison.context import ComparisonContext, stored_gate
from physverify.generator_comparison.prereg import ComparisonPreregistration, optional_prereg
from physverify.generator_comparison.records import JUDGE_SPLITS, load_judge_output

GATE_LINE: Final = "  gate: "
GATE_PASSED: Final = "passed"
"""``scripts/h200/run_generator_comparison.sh eval`` reads this status line to skip the judge
without a passed gate."""


def freeze_lines(record: ComparisonPreregistration) -> list[str]:
    usability = "not judged yet" if record.usability is None else str(record.usability.usable)
    fired = fired_reason(record.stop_rule, record.stop_rule_deviation)
    return [
        f"rule {record.rule} ({record.rule_status}, label {record.rule_label}); "
        f"contaminated {record.contaminated}",
        f"ft model {record.adapter_tag or 'none'}; prompts {json.dumps(record.eval_counts)}",
        f"margins {json.dumps(record.margins)}; external judge usable on dev: {usability}",
        f"MDE (estimated on dev): verifier {record.mde.verifier_mde}, PC {record.mde.pc_mde}",
        *[f"deviation: {d}" for d in record.deviations],
        *([] if fired is None else [f"NO VERDICT: {fired}"]),
        *(["POST HOC: refrozen after eval data existed"] if record.post_hoc else []),
    ]


def _tag_lines(h: ComparisonContext, tag: str, scorers: Callable[[], Iterable[str]]) -> list[str]:
    records = eval_records(h.store, tag)
    lines = [f"  {tag}: {len(records)} eval candidates"]
    for name in scorers():
        scores = eval_scores(h.store, tag, name)
        if scores:
            abstained = sum(s.score is None for s in scores)
            lines.append(f"    {name}: {len(scores)} scored, {abstained} abstained")
    latents = {r.latents_path is not None for r in records}
    if latents - {h.cfg.eval_store_latents}:
        lines.append(
            "    WARNING: store_latents differs from "
            f"h3.eval_store_latents={h.cfg.eval_store_latents}"
        )
    return lines


def _judge_lines(h: ComparisonContext) -> list[str]:
    lines = []
    for split in JUDGE_SPLITS:
        for path in h.layout.judge_outputs(split):
            output = load_judge_output(path)
            if output is not None:
                unread = sum(r.value is None for r in output.rows.values())
                lines.append(
                    f"  judge {split}/{path.stem}: {len(output.rows)} rows, {unread} unread"
                )
    return lines


def status_lines(h: ComparisonContext, scorers: Callable[[], Iterable[str]]) -> list[str]:
    """``scorers`` gives the registered scorer names, asked only when a model tag exists."""
    prereg = optional_prereg(h.layout.prereg)
    gate = stored_gate(h)
    lines = [
        f"campaign {h.store.campaign}: {h.layout.root}",
        "  pre-registration: "
        + (
            "none" if prereg is None else f"{prereg.rule_status}, ft {prereg.adapter_tag or 'none'}"
        ),
        GATE_LINE + ("not run" if gate is None else (GATE_PASSED if gate.passed else "failed")),
    ]
    for tag in h.store.model_tags():
        lines.extend(_tag_lines(h, tag, scorers))
    return [*lines, *_judge_lines(h)]
