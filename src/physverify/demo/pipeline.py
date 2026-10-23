"""End-to-end demo: prompt -> video (generated or given) -> extraction -> checks -> S -> report.

Every dependency (campaign store, generator, scoring stack, scorer) is injected; this module never
builds them. The per-check summaries and raw results are rule-independent and kept on disk, so the
clip score is always recomputed with the active scorer instead of trusting a stored stamp.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from physverify.core.atomic import write_atomic, write_text_atomic
from physverify.core.digests import file_sha256
from physverify.core.errors import UsageError
from physverify.demo.overlay import (
    OVERLAY_FILE,
    SIDE_BY_SIDE_FILE,
    overlay_frames,
    read_video,
    render_overlay,
    save_mp4,
    side_by_side,
)
from physverify.demo.report import (
    PROVISIONAL_BANNER,
    CandidateRef,
    RuleInfo,
    VideoReport,
    build_report,
    candidate_of,
    score_text,
    summaries_of,
    write_report,
)
from physverify.demo.settings import OverlayStyle
from physverify.finetune.generate import ADAPTER_HASH_CHARS, CandidateGenerator
from physverify.finetune.records import (
    BASE_TAG,
    CandidateRecord,
    CandidateScore,
    PromptRecord,
    write_record,
    write_records,
)
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scoring import ScoringDeps, score_candidates
from physverify.finetune.seeds import candidate_id
from physverify.finetune.store import CampaignStore
from physverify.schema.checks import CheckResult
from physverify.schema.io import read_jsonl

VIDEO_TAG_PREFIX: Final = "video-"
DEMO_K: Final = 0
SCENES_FILE: Final = "scenes.md"
Clock = Callable[[], float]


@dataclass(frozen=True)
class ReportDeps:
    """How a report is scored and drawn; ``max_frame_bytes`` is ``runtime.max_frame_bytes``."""

    scorer: AggregateScorer
    rule: RuleInfo
    style: OverlayStyle
    max_frame_bytes: int
    provenance: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DemoDeps:
    """``commit`` stamps the records of given videos."""

    store: CampaignStore
    scoring: ScoringDeps
    report: ReportDeps
    commit: str = ""
    clock: Clock = time.perf_counter


def register_prompt(store: CampaignStore, prompt: PromptRecord) -> None:
    prompts = store.prompts() if store.prompts_path.is_file() else []
    if all(p.prompt_id != prompt.prompt_id for p in prompts):
        write_records(store.prompts_path, [*prompts, prompt])


def video_tag(video: Path) -> tuple[str, str]:
    """The model tag of a given video (``video-<sha12>``) and its full sha256."""
    digest = file_sha256(video)
    return f"{VIDEO_TAG_PREFIX}{digest[:ADAPTER_HASH_CHARS]}", digest


def import_video(
    store: CampaignStore, prompt: PromptRecord, video: Path, commit: str
) -> CandidateRecord:
    """Copies ``video`` into the campaign's ``local_video`` subset as candidate k = 0; a stored
    copy with other content (a model-tag prefix collision) is refused, never overwritten."""
    if not video.is_file():
        raise UsageError(f"video {video} does not exist")
    tag, digest = video_tag(video)
    target = store.video_path(tag, prompt.prompt_id, DEMO_K)
    if target.is_file():
        _require_same(target, digest)
    else:
        write_atomic(target, lambda temp: shutil.copyfile(video, temp))
    record = CandidateRecord(
        candidate_id=candidate_id(prompt.prompt_id, DEMO_K),
        prompt_id=prompt.prompt_id,
        split=prompt.split,
        k=DEMO_K,
        seed=0,
        model_tag=tag,
        video_path=str(target),
        latents_path=None,
        embeds_path=str(store.embeds_path(prompt.prompt_id)),
        commit=commit,
        config_hash=digest,
    )
    write_record(store.meta_path(tag, prompt.prompt_id, DEMO_K), record)
    store.write_manifest(tag)
    return record


def _require_same(target: Path, digest: str) -> None:
    if file_sha256(target) != digest:
        raise UsageError(
            f"{target} holds another video under the same model tag; refusing to overwrite it"
        )


def stored_candidate(store: CampaignStore, model_tag: str, prompt_id: str) -> CandidateRecord:
    for record in store.candidates(model_tag):
        if record.prompt_id == prompt_id and record.k == DEMO_K and store.is_complete(record):
            return record
    raise UsageError(
        f"no complete k = {DEMO_K} video of {prompt_id} under model tag {model_tag} in {store.root}"
    )


def generate(generator: CandidateGenerator, prompt: PromptRecord) -> None:
    plan = generator.plan([prompt], k=DEMO_K + 1)
    generator.run(plan, {prompt.prompt_id: prompt})


def obtain(
    prompt: PromptRecord,
    deps: DemoDeps,
    video: Path | None,
    generator: CandidateGenerator | None,
    model_tag: str,
) -> CandidateRecord:
    if video is not None:
        return import_video(deps.store, prompt, video, deps.commit)
    if generator is not None:
        generate(generator, prompt)
        model_tag = generator.model_tag
    return stored_candidate(deps.store, model_tag, prompt.prompt_id)


def score_of(record: CandidateRecord, prompt: PromptRecord, deps: DemoDeps) -> CandidateScore:
    scorer = deps.report.scorer
    scores = score_candidates(deps.store, record.model_tag, [prompt], scorer, deps.scoring)
    for score in scores:
        if score.candidate_id == record.candidate_id:
            return score
    raise UsageError(f"{record.candidate_id} was not scored")


def report_of(
    prompt: PromptRecord,
    score: CandidateScore,
    record: CandidateRecord,
    deps: ReportDeps,
    timings: Mapping[str, float],
) -> VideoReport:
    results = read_jsonl(CheckResult, score.results_path)
    candidate = CandidateRef(
        model_tag=record.model_tag,
        candidate_id=record.candidate_id,
        video_path=record.video_path,
        results_path=score.results_path,
        motion=score.motion,
        objects_ok=score.objects_ok,
    )
    return build_report(
        prompt,
        candidate,
        score.per_check,
        results,
        deps.scorer.score(results),
        deps.rule,
        deps.provenance,
        timings,
    )


def report_dir(out: Path, report: VideoReport) -> Path:
    return out / f"{report.prompt.prompt_id}__{report.model_tag}"


def publish(report: VideoReport, out: Path, deps: ReportDeps) -> Path:
    """Writes report.json, report.md and overlay.mp4 under ``out/<prompt_id>__<model_tag>/``."""
    directory = report_dir(out, report)
    write_report(directory, report)
    render_overlay(report, directory / OVERLAY_FILE, deps.style, deps.max_frame_bytes)
    return directory


def run_demo(
    prompt: PromptRecord,
    deps: DemoDeps,
    *,
    out: Path,
    video: Path | None = None,
    generator: CandidateGenerator | None = None,
    model_tag: str = BASE_TAG,
) -> VideoReport:
    """``video`` is scored as given; else ``generator`` makes k = 0; else the stored k = 0 of
    ``model_tag`` is used."""
    register_prompt(deps.store, prompt)
    start = deps.clock()
    record = obtain(prompt, deps, video, generator, model_tag)
    obtained = deps.clock()
    score = score_of(record, prompt, deps)
    scored = deps.clock()
    timings = {"video_s": obtained - start, "score_s": scored - obtained}
    report = report_of(prompt, score, record, deps.report, timings)
    publish(report, out, deps.report)
    return report


def rebuild_report(old: VideoReport, deps: ReportDeps, out: Path) -> VideoReport:
    """The report of stored results under the active scorer (e.g. once calibration arrives)."""
    candidate = candidate_of(old)
    checks = read_jsonl(CheckResult, candidate.results_path)
    report = build_report(
        old.prompt,
        candidate,
        summaries_of(old),
        checks,
        deps.scorer.score(checks),
        deps.rule,
        {**old.provenance, **deps.provenance},
        old.timings,
    )
    publish(report, out, deps)
    return report


def write_side_by_side(left: VideoReport, right: VideoReport, path: Path, deps: ReportDeps) -> Path:
    clips = []
    fps = 0.0
    for report in (left, right):
        frames, fps = read_video(Path(report.video_path), deps.max_frame_bytes)
        clips.append(overlay_frames(frames, report, deps.style))
    labels = (left.model_tag, right.model_tag)
    return save_mp4(side_by_side(clips[0], clips[1], labels, deps.style), path, fps)


def _verdicts(report: VideoReport) -> dict[str, str]:
    return {row.check_id: row.verdict for row in report.checks}


def scene_lines(base: VideoReport, tuned: VideoReport) -> list[str]:
    old, new = _verdicts(base), _verdicts(tuned)
    changed = [
        f"{c}: {old.get(c, '-')} -> {new.get(c, '-')}"
        for c in sorted({*old, *new})
        if old.get(c) != new.get(c)
    ]
    return [
        f"## {base.prompt.prompt_id}",
        "",
        base.prompt.text,
        "",
        f"- {base.model_tag}: {score_text(base.clip)}, coverage {base.clip.coverage:.2f}",
        f"- {tuned.model_tag}: {score_text(tuned.clip)}, coverage {tuned.clip.coverage:.2f}",
        f"- Verdict changes: {'; '.join(changed) or 'none'}",
        f"- Side by side: `{base.prompt.prompt_id}/{SIDE_BY_SIDE_FILE}`",
        "",
    ]


def render_scenes(pairs: Sequence[tuple[VideoReport, VideoReport]], rule: RuleInfo) -> str:
    banner = [] if rule.calibrated else [f"> **{PROVISIONAL_BANNER}.**", ""]
    head = [
        "# Demo scenes: base vs fine-tuned",
        "",
        *banner,
        f"Rule: {rule.scorer} ({rule.status}), label `{rule.label}`.",
        "",
    ]
    return "\n".join([*head, *(line for pair in pairs for line in scene_lines(*pair))])


def run_scenes(
    prompts: Sequence[PromptRecord], deps: DemoDeps, tuned_tag: str, out: Path
) -> list[tuple[VideoReport, VideoReport]]:
    """Base and fine-tuned k = 0 of each prompt, already generated in the store."""
    pairs = []
    for prompt in prompts:
        reports = tuple(
            run_demo(prompt, deps, out=out, model_tag=tag) for tag in (BASE_TAG, tuned_tag)
        )
        base, tuned = reports
        path = out / prompt.prompt_id / SIDE_BY_SIDE_FILE
        write_side_by_side(base, tuned, path, deps.report)
        pairs.append((base, tuned))
    write_text_atomic(out / SCENES_FILE, render_scenes(pairs, deps.report.rule))
    return pairs
