"""``baselines score`` and ``baselines compare``: an external judge's scores of the final
run's sets, then, under the look's freeze, every clip's label beside its B1 and its judge score.

``compare`` orders its reads: the ledger line, then B1 of the calibration set's
clean clips (development data) and the cut u_B1 written to ``cuts.json``, and only then B1 and
the scores of the looked-at clips.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from physverify.app.calibration import calibration_session
from physverify.app.context import AppContext
from physverify.app.sealing import Selection, guard_sealed, own_looks, refuse_sealed_clips
from physverify.app.services import dataset_for, external_judge, verifier_b1
from physverify.baselines.labels import labels_of
from physverify.baselines.records import CUTS_FILE, PART_FILES, ClipRow, Cuts, PartRows, SetRows
from physverify.baselines.score_cache import ClipVideo, ScoreCache, read_scores, score_file
from physverify.baselines.settings import (
    JudgeSpec,
    Part,
    VlmComparisonSettings,
    load_vlm_comparison,
)
from physverify.baselines.stats import clean_cut
from physverify.baselines.videos import judge_video
from physverify.calibration.artifact_io import Calibration, default_artifact_path
from physverify.calibration.clip_values import ClipValueOf
from physverify.calibration.session import Session, artifact_at, fit_clips
from physverify.calibration.table import is_perturbed
from physverify.core.atomic import write_text_atomic
from physverify.core.digests import file_sha256
from physverify.core.errors import ConfigError, MissingInputError, UsageError
from physverify.core.output import Echo
from physverify.datasets._base import DatasetSource, select_clips
from physverify.diagnosis.settings import diagnosis_settings
from physverify.generator_comparison.judges._base import Judge
from physverify.generator_comparison.settings import load_comparison
from physverify.schema.checks import CheckResult, InputLevel
from physverify.schema.clip import ClipManifest

WORK_DIR: Final = "work"
ENCODED_DIR: Final = "videos"


@dataclass(frozen=True)
class ScoreRequest:
    judge: str
    dataset: str
    subset: str | None
    scores: Path
    autoeval_root: Path | None = None
    limit: int | None = None


@dataclass(frozen=True)
class CompareRequest:
    part: str
    artifact: Path
    dataset: str | None
    subset: str | None
    real_datasets: tuple[str, ...]
    scores: Path
    out: Path
    judge: str
    clips: tuple[str, ...] = ()
    """``--part kubric`` only: these clips of a development set (refused on a sealed one)."""


def score_clips(context: AppContext, request: ScoreRequest, *, echo: Echo) -> None:
    """Fill the judge's score cache of a dataset; prints the cache path, never a score."""
    settings = context.settings
    config = load_vlm_comparison(settings)
    spec = config.judge(request.judge)
    caption = config.set_of(request.dataset).caption
    home = request.scores / request.judge
    judge = _judge(context, spec, request.autoeval_root, home / WORK_DIR)
    source = dataset_for(context, request.dataset)
    subset = request.subset or diagnosis_settings(settings).subset
    clips = _limited(source.clips(subset), request.limit)
    data_dir, encoded = settings.paths.data_dir, home / ENCODED_DIR
    videos = [_clip_video(source, clip, data_dir, encoded, caption) for clip in clips]
    cache = ScoreCache(
        score_file(request.scores, request.judge, request.dataset),
        judge,
        spec.task,
        spec.clips_per_call,
    )
    echo(f"scores: {cache.path} ({cache.fill(videos)} clips)")


def compare_part(
    context: AppContext, command: str, request: CompareRequest, *, echo: Echo, err: Echo
) -> None:
    """Write ``cuts.json`` and the part's per-clip rows; prints their paths, never a number."""
    settings = context.settings
    config = load_vlm_comparison(settings)
    spec = config.judge(request.judge)
    part = _part(request.part)
    _require_frozen_artifact(default_artifact_path(settings), request.artifact)
    named = _named_sets(config, part, request, diagnosis_settings(settings).subset)
    refuse_sealed_clips(context, [name for name, _ in named], request.clips, err=err)
    files = {name: score_file(request.scores, request.judge, name) for name, _ in named}
    absent = sorted(str(path) for path in files.values() if not path.is_file())
    if absent:
        raise MissingInputError(f"no judge scores at {', '.join(absent)}: run `baselines score`")
    session = calibration_session(context, echo, err)
    calibration = artifact_at(session, request.artifact)
    sources = {name: dataset_for(context, name) for name, _ in named}
    chosen = {
        name: tuple(select_clips(sources[name].clips(subset), request.clips))
        for name, subset in named
    }
    b1 = verifier_b1(context)
    selections = [Selection(name, subset, chosen[name]) for name, subset in named]
    guard_sealed(
        context,
        command,
        selections,
        err=err,
        artifacts={"calibration": calibration.sha256},
        run_dir=session.run.run_dir,
    )
    value_of = b1.under(calibration.stratum(InputLevel.PERCEIVED, settings.profile))
    cuts = _cuts(session, calibration, value_of, config.clean_fpr, b1.excluded)
    cuts_path = request.out / CUTS_FILE
    write_text_atomic(cuts_path, cuts.to_json())
    echo(f"cuts: {cuts_path}")
    sets = tuple(
        _set_rows(session, part, sources[name], subset, chosen[name], value_of, files[name])
        for name, subset in named
    )
    freeze, looks = own_looks(context, command, session.run.run_dir)
    rows = PartRows(
        part=part,
        judge=request.judge,
        cut=spec.cut,
        cuts=cuts,
        cuts_sha256=file_sha256(cuts_path),
        freeze=freeze,
        looks=looks,
        sets=sets,
    )
    path = request.out / PART_FILES[part]
    write_text_atomic(path, rows.to_json())
    echo(f"comparison: {path}")


def _judge(context: AppContext, spec: JudgeSpec, root: Path | None, workdir: Path) -> Judge:
    """The judge of ``spec`` under ``[h3] autoeval`` (its root replaced by ``root``), refused
    when it cannot run or its prompt is not the pinned one."""
    comparison = load_comparison(context.settings)
    autoeval = comparison.autoeval
    if root is not None:
        autoeval = autoeval.model_copy(update={"root": str(root)})
    built = external_judge(
        context,
        comparison.model_copy(update={"external_judge": spec.plugin, "autoeval": autoeval}),
        workdir,
    )
    missing = built.unavailable()
    if missing is not None:
        raise ConfigError(f"judge {spec.plugin}: {missing}")
    prompt = autoeval.path(spec.prompt_file)
    if prompt is None or not prompt.is_file():
        raise ConfigError(f"judge {spec.plugin}: no prompt file {spec.prompt_file}")
    if file_sha256(prompt) != spec.prompt_sha256:
        raise ConfigError(f"judge {spec.plugin}: {prompt} is not the pinned prompt")
    return built


def _limited(clips: Sequence[ClipManifest], limit: int | None) -> Sequence[ClipManifest]:
    """The ``limit`` clips of smallest sha256(clip_id) (a rehearsal's fixed sample), or all."""
    if limit is None:
        return clips
    return sorted(clips, key=lambda c: hashlib.sha256(c.clip_id.encode()).hexdigest())[:limit]


def _clip_video(
    source: DatasetSource, clip: ClipManifest, data_dir: Path, encoded: Path, caption: str
) -> ClipVideo:
    video = judge_video(source, clip, data_dir, encoded)
    return ClipVideo(clip.clip_id, video, file_sha256(video), caption)


def _require_frozen_artifact(frozen: Path, given: Path) -> None:
    if given.resolve() != frozen.resolve():
        raise UsageError(f"--artifact must be the frozen artifact {frozen}, not {given}")


def _part(value: str) -> Part:
    for part in PART_FILES:
        if part == value:
            return part
    raise UsageError(f"--part is one of {sorted(PART_FILES)}, not {value!r}")


def _named_sets(
    config: VlmComparisonSettings, part: Part, request: CompareRequest, real_subset: str
) -> list[tuple[str, str | None]]:
    if part == "kubric":
        if request.dataset is None or request.real_datasets:
            raise UsageError("--part kubric takes --dataset (and --subset), no --real-dataset")
        config.set_of(request.dataset, "kubric")
        return [(request.dataset, request.subset)]
    if request.dataset is not None or request.clips or not request.real_datasets:
        raise UsageError("--part real takes --real-dataset (repeatable), no --dataset or --clips")
    for name in request.real_datasets:
        config.set_of(name, "real")
    return [(name, real_subset) for name in dict.fromkeys(request.real_datasets)]


def _cuts(
    session: Session,
    calibration: Calibration,
    value_of: ClipValueOf,
    clean_fpr: float,
    excluded: frozenset[str],
) -> Cuts:
    """u_B1 from B1 of the calibration set's clean clips."""
    cal = session.config.calibration
    source = session.dataset(cal.dataset)
    clean = [c for c in fit_clips(source, cal.subset, ()) if not is_perturbed(c.perturbation)]
    values = [v for v in _b1_by_clip(session, source, clean, value_of).values() if v is not None]
    return Cuts(
        u_b1=clean_cut(values, clean_fpr),
        clean_fpr=clean_fpr,
        n_clean=len(values),
        calibration_dataset=source.name,
        calibration_sha256=calibration.sha256,
        excluded_checks=tuple(sorted(excluded)),
    )


def _set_rows(
    session: Session,
    part: Part,
    source: DatasetSource,
    subset: str | None,
    clips: Sequence[ClipManifest],
    value_of: ClipValueOf,
    scores: Path,
) -> SetRows:
    b1 = _b1_by_clip(session, source, clips, value_of)
    read = read_scores(scores)
    rows = tuple(
        ClipRow(
            clip_id=label.clip_id,
            scene=label.scene,
            perturbed=label.perturbed,
            b1=b1.get(label.clip_id),
            judge=None if label.clip_id not in read else read[label.clip_id].value,
        )
        for label in labels_of(part, clips)
    )
    return SetRows(
        dataset=source.name,
        subset=subset,
        scores_sha256=file_sha256(scores),
        stamps=tuple(sorted({read[r.clip_id].stamp for r in rows if r.clip_id in read})),
        rows=rows,
    )


def _b1_by_clip(
    session: Session, source: DatasetSource, clips: Sequence[ClipManifest], value_of: ClipValueOf
) -> dict[str, float | None]:
    report = session.raw_runner().run(session.input_source(source, InputLevel.PERCEIVED), clips)
    by_clip: dict[str, list[CheckResult]] = {c.clip_id: [] for c in clips}
    for result in report.results:
        by_clip.setdefault(result.clip_id, []).append(result)
    return {clip_id: value_of(results) if results else None for clip_id, results in by_clip.items()}
