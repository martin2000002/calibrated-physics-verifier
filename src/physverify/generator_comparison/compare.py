"""``compare-generators compare``: load the frozen eval data, bind it to the pre-registration and
build the report.

Every score manifest and eval judge output must name a pre-registration of the chain (by content
hash), and every eval score must carry the frozen scorer version and rule label; file times are
only reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from physverify.core.errors import UsageError
from physverify.finetune.layout import CampaignStore
from physverify.finetune.pairs import Rescore, stamp_of
from physverify.finetune.records import BASE_TAG, CandidateScore, read_record
from physverify.finetune.settings import FinetuneSettings
from physverify.finetune.stop_rule import fired_reason
from physverify.generator_comparison.analysis_inputs import AnalysisInputs
from physverify.generator_comparison.arms import EVAL_SPLIT, analysis_prompts, build_arms
from physverify.generator_comparison.candidate_queries import eval_ids, eval_scores
from physverify.generator_comparison.judging import gate_problem
from physverify.generator_comparison.layout import H3Layout
from physverify.generator_comparison.prereg import (
    H3Preregistration,
    chain,
    chain_shas,
    first_created,
    stamp_problem,
)
from physverify.generator_comparison.preregistered_numbers import analyse
from physverify.generator_comparison.records import (
    TASKS,
    GateReport,
    JudgeKey,
    JudgeOutput,
    ScoringManifest,
    keyed_values,
    load_judge_output,
)
from physverify.generator_comparison.report import H3Report
from physverify.generator_comparison.settings import H3Settings
from physverify.generator_comparison.stats import method_label
from physverify.generator_comparison.verdict import ExternalStatus, VerdictInputs, h3_verdict
from physverify.human_agreement.prereg import (
    FreezeAudit,
    earliest_mtime,
    frozen_after,
    frozen_before,
)

RESCORE_HINT: Final = "run `physverify compare-generators rescore`"


@dataclass(frozen=True)
class CompareInputs:
    store: CampaignStore
    layout: H3Layout
    cfg: H3Settings
    finetune: FinetuneSettings
    prereg: H3Preregistration
    prereg_sha256: str
    scorer: str
    rescore: Rescore
    commit: str


def require_stamp(scores: Sequence[CandidateScore], prereg: H3Preregistration, tag: str) -> None:
    if not scores:
        raise UsageError(
            f"no eval scores of {tag}: run `physverify compare-generators score --arm ...`"
        )
    try:
        stamp = stamp_of(scores)
    except ValueError as exc:
        raise UsageError(f"{tag}: {exc}; {RESCORE_HINT}") from exc
    problem = stamp_problem(stamp.scorer_version, stamp.thresholds, prereg)
    if problem is not None:
        raise UsageError(f"{tag} scores: {problem}; {RESCORE_HINT}")


def require_bound(sha256: str | None, known: set[str], what: str) -> None:
    if sha256 not in known:
        raise UsageError(f"{what} is not bound to this pre-registration; produce it again")


def require_manifest(layout: H3Layout, tag: str, scorer: str, known: set[str]) -> ScoringManifest:
    path = layout.scoring(tag, scorer)
    if not path.is_file():
        raise UsageError(f"{path} is missing: run `physverify compare-generators score`")
    manifest = read_record(path, ScoringManifest)
    require_bound(manifest.prereg_sha256, known, str(path))
    return manifest


def eval_judges(
    layout: H3Layout, tags: Sequence[str], prereg: H3Preregistration, known: set[str]
) -> dict[str, JudgeOutput | None]:
    """``{"<tag>__<task>": output}``; a present output must be bound and carry the frozen stamp."""
    found: dict[str, JudgeOutput | None] = {}
    for tag in tags:
        for task in TASKS:
            path = layout.judge_output(EVAL_SPLIT, tag, task)
            output = load_judge_output(path)
            if output is not None:
                require_bound(output.prereg_sha256, known, str(path))
                if output.stamp != prereg.judge_stamps.get(task):
                    raise UsageError(f"{path} has stamp {output.stamp!r}, not the frozen one")
            found[f"{tag}__{task}"] = output
    return found


def external_status(
    layout: H3Layout, prereg: H3Preregistration, outputs: Mapping[str, JudgeOutput | None]
) -> ExternalStatus:
    """A gate that did not run the frozen judge stamps counts as not run."""
    gate = read_record(layout.gate, GateReport) if layout.gate.is_file() else None
    if gate_problem(gate, prereg.judge_stamps) is not None:
        return "failed_gate" if gate is not None and not gate.passed else "not_run"
    if prereg.usability is None:
        return "not_run"
    if not prereg.usability.usable:
        return "unusable"
    return "ok" if outputs and all(o is not None for o in outputs.values()) else "not_run"


def task_values(
    outputs: Mapping[str, JudgeOutput | None], task: str
) -> dict[JudgeKey, float | None]:
    merged: dict[JudgeKey, float | None] = {}
    for key, output in outputs.items():
        if key.endswith(f"__{task}"):
            merged.update(keyed_values(output))
    return merged


def eval_files(inputs: CompareInputs, tags: Sequence[str]) -> list[Path]:
    store, layout = inputs.store, inputs.layout
    scores = [
        store.score_path(tag, inputs.scorer, cid) for tag in tags for cid in eval_ids(store, tag)
    ]
    return [*scores, *layout.judge_outputs(EVAL_SPLIT)]


def audit(inputs: CompareInputs, tags: Sequence[str]) -> FreezeAudit:
    first = first_created(inputs.layout, inputs.prereg)
    earliest = earliest_mtime(eval_files(inputs, tags))
    return FreezeAudit(
        prereg_sha256=inputs.prereg_sha256,
        versions=len(chain(inputs.layout, inputs.prereg)),
        created_utc=inputs.prereg.created_utc,
        replaces_sha256=inputs.prereg.replaces_sha256,
        first_created_utc=first,
        earliest_eval_utc=earliest,
        blind=frozen_before(first, earliest),
        frozen_after_eval=frozen_after(inputs.prereg.created_utc, earliest),
        post_hoc=inputs.prereg.post_hoc,
        frozen_commit=inputs.prereg.commit,
        report_commit=inputs.commit,
        deviations=list(inputs.prereg.deviations),
    )


def build_compare(inputs: CompareInputs) -> H3Report:
    prereg, cfg, cand = inputs.prereg, inputs.cfg, inputs.finetune.candidates
    tag = prereg.adapter_tag
    tags = [BASE_TAG] if tag is None else [BASE_TAG, tag]
    known = chain_shas(inputs.layout, prereg, inputs.prereg_sha256)
    scores = {t: eval_scores(inputs.store, t, inputs.scorer) for t in tags}
    for t in tags:
        require_manifest(inputs.layout, t, inputs.scorer, known)
        require_stamp(scores[t], prereg, t)
    outputs = eval_judges(inputs.layout, tags, prereg, known)
    status = external_status(inputs.layout, prereg, outputs)
    prompts = analysis_prompts(inputs.store.prompts(), cfg.exclude_tags)
    arms = build_arms(
        prompts, scores[BASE_TAG], scores.get(tag or "", []), cand.k_eval, cand.k_eval_finetuned
    )
    result = analyse(
        AnalysisInputs(
            arms=arms,
            has_ft=tag is not None,
            k_ft=cand.k_eval_finetuned,
            rescore=inputs.rescore,
            rules=inputs.finetune.pairs,
            select=inputs.finetune.select,
            cfg=cfg,
            pc=task_values(outputs, "pc"),
            sa=task_values(outputs, "sa"),
            external_status=status,
        )
    )
    verdict = h3_verdict(
        VerdictInputs(
            rule_status=prereg.rule_status,
            contaminated=prereg.contaminated,
            label_matches=True,
            post_hoc=prereg.post_hoc,
            stop_rule_fired=fired_reason(prereg.stop_rule, prereg.stop_rule_deviation),
            has_ft_model=tag is not None,
            primary=result.primary,
            external_status=status,
            external=result.external,
            margin_pc=prereg.margins["pc"],
            alpha=cfg.alpha,
            guards={name: g.state for name, g in result.guards.items()},
        )
    )
    return H3Report(
        campaign=inputs.store.campaign,
        rule_label=prereg.rule_label,
        rule_status=prereg.rule_status,
        ft_model_tag=tag,
        scipy_version=prereg.scipy_version,
        method=method_label(cfg),
        analysis=result,
        verdict=verdict,
        margins=prereg.margin_derivations,
        mde=prereg.mde,
        audit=audit(inputs, tags),
    )
