"""The current values of every pre-registered generator-comparison field (``prereg.PREREGISTERED_FIELDS``),
from the configuration, the campaign store and the files they name. Nothing here decides; ``prereg``
does."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import scipy

from physverify.calibration.artifact_io import CalibrationError, load_calibration
from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.core.digests import sha256_if_file
from physverify.core.errors import MissingInputError, PhysverifyError
from physverify.datasets.local_video import NAME as LOCAL_VIDEO
from physverify.finetune.candidates import adapter_tag
from physverify.finetune.layout import CampaignStore
from physverify.finetune.pairs import Rescore
from physverify.finetune.records import BASE_TAG, CandidateScore, PromptRecord
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.selection import Selection
from physverify.finetune.settings import SECTION as FINETUNE_SECTION
from physverify.finetune.settings import FinetuneSettings
from physverify.finetune.stop_rule import StopRuleState
from physverify.generator_comparison.analysis_inputs import JudgeValues
from physverify.generator_comparison.arms import (
    EVAL_SPLIT,
    PromptArms,
    analysis_prompts,
    build_arms,
)
from physverify.generator_comparison.candidate_queries import all_eval_records, candidate_snapshot
from physverify.generator_comparison.judge_gate import usability
from physverify.generator_comparison.judges._base import Judge
from physverify.generator_comparison.layout import ComparisonLayout
from physverify.generator_comparison.margins import MarginDerivation, margin
from physverify.generator_comparison.prereg import ComparisonPreregistration, Mde
from physverify.generator_comparison.preregistered_numbers import (
    SECONDARY_TESTS,
    crn_ft_vs_base,
    judged,
    judged_against_base,
)
from physverify.generator_comparison.records import (
    TASKS,
    JudgeKey,
    Usability,
    judge_values,
    keyed_values,
    load_judge_output,
)
from physverify.generator_comparison.settings import AUTOEVAL_KEY, SECTION, ComparisonSettings
from physverify.generator_comparison.stats import mde_paired, sample_sd
from physverify.generator_comparison.verdict import RuleStatus
from physverify.schema.io import stable_hash
from physverify.verification.scope import check_scope, require_record_for

PLAIN_SECTIONS: Final = (FINETUNE_SECTION, "checks", "verification", "extraction")
TIMING: Final = ("timing", "generation")
CONTAMINATED_DATASET: Final = LOCAL_VIDEO
CALIB_SPLIT: Final = "calib"
DEV_SPLIT: Final = "dev"
MARGIN_KEYS: Final = {"pc": "noninferiority_margin_pc", "sa": "sa_guard_margin"}


def section_hashes(sections: Mapping[str, Any]) -> dict[str, str]:
    """``[h3]`` without its machine-local ``autoeval`` subtree (the judge is in its stamps)."""
    comparison = {k: v for k, v in dict(sections.get(SECTION) or {}).items() if k != AUTOEVAL_KEY}
    timing = dict(sections.get(TIMING[0]) or {}).get(TIMING[1])
    hashes = {name: stable_hash({name: sections.get(name)}) for name in PLAIN_SECTIONS}
    return {
        **hashes,
        SECTION: stable_hash({SECTION: comparison}),
        "timing_generation": stable_hash({"timing.generation": timing}),
    }


def eval_counts(
    prompts: Sequence[PromptRecord], analysed: Sequence[PromptRecord]
) -> dict[str, int]:
    unseen = sum(PromptArms(p, {}, {}).unseen for p in analysed)
    return {
        "eval": sum(p.split == EVAL_SPLIT for p in prompts),
        "n": len(analysed),
        "unseen": unseen,
        "seen": len(analysed) - unseen,
        "clusters": len({p.template_id for p in analysed}),
    }


def rule_state(scorer: AggregateScorer) -> tuple[RuleStatus, str | None]:
    if not scorer.calibrated:
        return "provisional", scorer.thresholds()
    if scorer.unavailable() is not None:
        return "unavailable", None
    return "calibrated", scorer.thresholds()


def calibration_dataset(path: Path) -> dict[str, Any] | None:
    try:
        return load_calibration(path).artifact.dataset.model_dump(mode="json")
    except (CalibrationError, OSError):
        return None


def contaminated(dataset: Mapping[str, Any] | None) -> bool | None:
    """Calibrated on our own videos or outside the calibration split; None when
    the calibration dataset is unknown (missing or unreadable artefact), never assumed clean."""
    if dataset is None:
        return None
    return dataset.get("name") == CONTAMINATED_DATASET or dataset.get("split") != CALIB_SPLIT


def derive_margins(
    rows: Sequence[Mapping[str, str]], columns: Mapping[str, str], cfg: ComparisonSettings
) -> dict[str, MarginDerivation]:
    """Recomputed from the published table; a value other than ``[h3]``'s stops the freeze."""
    derived = {
        task: margin(rows, columns[task], cfg.generator_column, cfg.margin_min_generator_n)
        for task in TASKS
    }
    for task, derivation in derived.items():
        configured = float(getattr(cfg, MARGIN_KEYS[task]))
        if derivation.value != configured:
            raise PhysverifyError(
                f"the {task} margin from the published table is {derivation.value}, "
                f"[h3] {MARGIN_KEYS[task]} is {configured}"
            )
    return derived


def dev_arms(
    prompts: Sequence[PromptRecord],
    base: Sequence[CandidateScore],
    ft: Sequence[CandidateScore],
    k_dev: int,
) -> list[PromptArms]:
    dev = [p for p in prompts if p.split == DEV_SPLIT]
    return build_arms(dev, base, ft, k_dev, k_dev)


def dev_mde(
    arms: Sequence[PromptArms],
    k_dev: int,
    rescore: Rescore,
    pc: JudgeValues,
    n: int,
    cfg: ComparisonSettings,
) -> Mde:
    """sd of the per-prompt dev differences, FT against base on common seeds (estimated)."""
    verifier = sample_sd(list(crn_ft_vs_base(arms, k_dev, rescore, 1)[0].values()))

    def ft_pc(arm: PromptArms) -> float | None:
        values = [judged(pc, s) for s in arm.ft.values()]
        kept = [v for v in values if v is not None]
        return sum(kept) / len(kept) if kept else None

    pc_sd = sample_sd(list(judged_against_base(arms, pc, ft_pc, 1)[0].values())) if pc else None
    pc_mde = None if pc_sd is None else mde_paired(n, pc_sd, cfg.alpha, cfg.power)
    return Mde(
        n=n,
        alpha=cfg.alpha,
        power=cfg.power,
        verifier_sd=verifier,
        verifier_mde=None if verifier is None else mde_paired(n, verifier, cfg.alpha, cfg.power),
        pc_sd=pc_sd,
        pc_mde=pc_mde,
        pc_underpowered=None if pc_mde is None else pc_mde > 2 * cfg.noninferiority_margin_pc,
    )


def dev_usability(
    layout: ComparisonLayout,
    tags: Sequence[str],
    stamps: Mapping[str, str],
    cfg: ComparisonSettings,
) -> Usability | None:
    """None until every dev judge output exists under the current judge stamp of its task."""
    values: dict[str, list[float | None]] = {}
    for task in TASKS:
        outputs = [load_judge_output(layout.judge_output(DEV_SPLIT, tag, task)) for tag in tags]
        if any(o is None or o.stamp != stamps[task] for o in outputs):
            return None
        values[task] = [v for o in outputs for v in judge_values(o).values()]
    return usability(values, cfg.usability)


def dev_pc(
    layout: ComparisonLayout, tags: Sequence[str], stamp: str
) -> dict[JudgeKey, float | None]:
    """Dev PC values judged under the current PC stamp; outputs of another judge are skipped."""
    found: dict[JudgeKey, float | None] = {}
    for tag in tags:
        output = load_judge_output(layout.judge_output(DEV_SPLIT, tag, "pc"))
        if output is not None and output.stamp == stamp:
            found.update(keyed_values(output))
    return found


@dataclass(frozen=True)
class FreezeInputs:
    settings: Settings
    checks: Mapping[str, type[Check]]
    store: CampaignStore
    layout: ComparisonLayout
    cfg: ComparisonSettings
    finetune: FinetuneSettings
    scorer: AggregateScorer
    rescore: Rescore
    judge: Judge
    selection: Selection
    calibration_path: Path
    margin_rows: Sequence[Mapping[str, str]]
    margin_source: Path
    commit: str
    stop_rule: StopRuleState


def adapter_identity(selection: Selection) -> tuple[str | None, str | None]:
    """The adapter path and the tag of its current bytes (a changed file changes the tag)."""
    if selection.selected is None:
        return None, None
    path = Path(selection.selected.adapter)
    if not path.exists():
        raise MissingInputError(f"the selected adapter {path} does not exist")
    return str(path), adapter_tag(path)


def current_record(inputs: FreezeInputs, now: dt.datetime) -> ComparisonPreregistration:
    prompts = inputs.store.prompts()
    analysed = analysis_prompts(prompts, inputs.cfg.exclude_tags)
    rule_status, rule_label = rule_state(inputs.scorer)
    dataset = calibration_dataset(inputs.calibration_path)
    margins = derive_margins(inputs.margin_rows, inputs.cfg.gate.columns, inputs.cfg)
    adapter, tag = adapter_identity(inputs.selection)
    dev_tags = [BASE_TAG] if tag is None else [BASE_TAG, tag]
    stamps = {task: inputs.judge.stamp(task) for task in TASKS}
    return ComparisonPreregistration(
        commit=inputs.commit,
        section_hashes=section_hashes(inputs.settings.sections),
        prompts_sha256=sha256_if_file(inputs.store.prompts_path),
        eval_counts=eval_counts(prompts, analysed),
        selection_run_id=inputs.selection.run_id,
        adapter=adapter,
        adapter_tag=tag,
        calibration_artifact_sha256=sha256_if_file(inputs.calibration_path),
        rule=inputs.scorer.name,
        scorer_version=inputs.scorer.version,
        rule_label=rule_label,
        rule_status=rule_status,
        calibration_dataset=dataset,
        contaminated=contaminated(dataset),
        judge_stamps=stamps,
        gate_sha256=sha256_if_file(inputs.layout.gate),
        margins={task: d.value for task, d in margins.items()},
        margin_derivations=margins,
        margin_source_sha256=sha256_if_file(inputs.margin_source),
        bootstrap_seed=inputs.cfg.bootstrap_seed,
        mde=_mde(inputs, prompts, tag, len(analysed)),
        usability=dev_usability(inputs.layout, dev_tags, stamps, inputs.cfg),
        scipy_version=scipy.__version__,
        eval_candidates_at_freeze=candidate_snapshot(all_eval_records(inputs.store)),
        stop_rule=inputs.stop_rule,
        created_utc=now.astimezone(dt.UTC).isoformat(timespec="seconds"),
        exclusion_record_sha256=check_scope(inputs.settings, inputs.checks).record_sha256,
        secondary_tests=list(SECONDARY_TESTS),
    )


def require_exclusion_record(
    settings: Settings, current: ComparisonPreregistration, checks: Mapping[str, type[Check]]
) -> None:
    """A calibrated generator-comparison rule is frozen only on the decision record of its
    artifact."""
    require_record_for(
        check_scope(settings, checks),
        calibrated=current.rule_status == "calibrated",
        artifact_sha256=current.calibration_artifact_sha256,
    )


def _mde(inputs: FreezeInputs, prompts: Sequence[PromptRecord], tag: str | None, n: int) -> Mde:
    name = inputs.scorer.name
    base = inputs.store.scores(BASE_TAG, name)
    ft = [] if tag is None else inputs.store.scores(tag, name)
    k_dev = inputs.finetune.candidates.k_dev
    arms = dev_arms(prompts, base, ft, k_dev)
    tags = [BASE_TAG] if tag is None else [BASE_TAG, tag]
    pc = dev_pc(inputs.layout, tags, inputs.judge.stamp("pc"))
    return dev_mde(arms, k_dev, inputs.rescore, pc, n, inputs.cfg)
