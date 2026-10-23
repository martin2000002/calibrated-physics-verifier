"""``physverify size``: fetch Objectron boxes, fit the size model, measure it (ADR 0010).

Network but not heavy: ``fetch`` reads only the leading ``Sequence.objects`` bytes of each
annotation through ranged GETs. Every figure written is an estimate with its measured error."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any

import typer

from physverify.app.context import AppContext
from physverify.app.services import pipeline_deps, resolve_dataset
from physverify.cli.commands._runs import write_json
from physverify.cli.context import app_context
from physverify.config.models import Settings, SizePriorTable
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import MissingInputError
from physverify.core.markdown import number
from physverify.datasets._base import DatasetSource
from physverify.datasets.local_video import NAME as LOCAL_VIDEO
from physverify.datasets.objectron_static import VIDEO_FILE
from physverify.diagnosis.cases import stage_output
from physverify.diagnosis.chain import ChainRunner
from physverify.diagnosis.plans import Variant
from physverify.extraction.providers.scale_size_prior import SIZE_PRIORS_SECTION
from physverify.runtime.runs import new_run
from physverify.schema.clip import ClipManifest
from physverify.schema.perception import MaskTrack
from physverify.schema.scale import ScaleEstimate
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable
from physverify.sizing import e2e, objectron
from physverify.sizing.draws import TEST, TRAIN, e2e_draw, objectron_dir, selection
from physverify.sizing.evaluate import MODEL, evaluate
from physverify.sizing.fit import conformal_factor, fit_model, render_toml
from physverify.sizing.model import SECTION as MODEL_SECTION
from physverify.sizing.model import SizeModelTable
from physverify.sizing.samples import SizeSample, samples_of
from physverify.sizing.settings import EndToEndSettings, SizingSettings, sizing_settings
from physverify.verification.inputs import refuse_heavy

NAME = "size"
MODEL_FILE = "size_model.toml"
SCALE_METHODS = ("size_prior", "size_model")

app = typer.Typer(help="Object size model learned on Objectron (fetch, fit, eval).")


def load_samples(
    settings: Settings, sizing: SizingSettings, split: str, reader: objectron.PrefixReader
) -> list[SizeSample]:
    heads = objectron_dir(settings) / "heads"
    samples: list[SizeSample] = []
    missing = []
    for ref in selection(settings, sizing, split, reader):
        path = objectron.head_path(heads, ref)
        if not path.is_file():
            missing.append(ref.video_id)
            continue
        samples += samples_of(ref, objectron.load_head(path), split, sizing.min_extent_m)
    if missing:
        raise MissingInputError(f"{len(missing)} annotations not fetched; run `size fetch`")
    return samples


@app.command("fetch", help="Download the box sizes of the selected Objectron videos.")
def fetch() -> None:
    context = app_context()
    settings = context.settings
    sizing = sizing_settings(settings)
    reader = objectron.UrllibPrefixReader()
    plan = objectron.HeadFetch(
        sizing.bucket_url, objectron_dir(settings) / "heads", sizing.head_bytes, sizing.retries
    )
    for split in objectron.SPLITS:
        refs = selection(settings, sizing, split, reader)
        stored = objectron.fetch_heads(reader, plan, refs, sizing.workers)
        typer.echo(f"{split}: {len(refs)} videos, {stored} new bytes stored")


@app.command("fit", help="Fit the per-class model on train; writes configs/size_model.toml.")
def fit() -> None:
    context = app_context()
    settings = context.settings
    sizing = sizing_settings(settings)
    train = load_samples(settings, sizing, TRAIN, objectron.UrllibPrefixReader())
    source = f"objectron train ({sizing.license}), {len(train)} objects, seed {sizing.seed}"
    model = fit_model(
        train,
        folds=sizing.folds,
        seed=sizing.seed,
        coverage=sizing.coverage_target,
        source=source,
    )
    target = settings.config_dir / MODEL_FILE
    write_text_atomic(target, render_toml(model))
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    write_json(run_dir / "fit.json", _fit_record(model, train, sizing))
    typer.echo(f"model: {target} ({len(model.classes)} classes, factor {model.sigma_factor:.4f})")
    typer.echo(f"run: {run_dir}")


def _fit_record(
    model: SizeModelTable, train: Sequence[SizeSample], sizing: SizingSettings
) -> dict[str, Any]:
    """Fit evidence; the test sigma is measured by ``size eval`` (eval.json), never here."""
    return {
        "estimated": True,
        "model": model.model_dump(mode="json"),
        "n_objects": len(train),
        "n_videos": len({s.video_id for s in train}),
        "folds": sizing.folds,
        "factor_at_coverage": {
            str(q): conformal_factor(train, sizing.folds, sizing.seed, q) for q in (0.683, 0.954)
        },
        "test_sigma": "see eval.json (size eval)",
        "truth_limit": "box sizes come from ARKit; their own error is not documented",
    }


@app.command("eval", help="Error on the official test split against B0 and B1.")
def evaluate_command() -> None:
    context = app_context()
    settings = context.settings
    sizing = sizing_settings(settings)
    model = settings.section(MODEL_SECTION, SizeModelTable)
    if not model.classes:
        raise MissingInputError("configs/size_model.toml is empty; run `size fit` first")
    test = load_samples(settings, sizing, TEST, objectron.UrllibPrefixReader())
    priors = settings.section(SIZE_PRIORS_SECTION, SizePriorTable).root
    report = evaluate(
        test,
        model,
        priors[sizing.baseline_prior],
        n_boot=sizing.bootstrap,
        seed=sizing.seed,
        ci=sizing.ci,
    )
    report["test_sigma_log"] = report["total"][MODEL]["rms_log"]
    report["truth_limit"] = "box sizes come from ARKit; their own error is not documented"
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    write_json(run_dir / "eval.json", report)
    for line in eval_lines(report):
        typer.echo(line)
    typer.echo(f"run: {run_dir}")


def eval_lines(report: dict[str, Any]) -> list[str]:
    total = report["total"]
    lines = [f"test: {report['n_objects']} objects, {report['n_videos']} videos (estimates)"]
    for name in (MODEL, *(k for k in total if k != MODEL and not k.startswith("rms_log_"))):
        m = total[name]
        lines.append(
            f"  {name}: median|L/L-1| {_v(m['median_abs_rel'])}  rms log {_v(m['rms_log'])}  "
            f"cover 1s {_v(m['cover_1s'])}  2s {_v(m['cover_2s'])}"
        )
    for key in (k for k in total if k.startswith("rms_log_")):
        lines.append(f"  {key}: {_v(total[key])}")
    return lines


def _v(entry: dict[str, Any]) -> str:
    low, high = (_number(bound) for bound in entry["ci"])
    return f"{entry['value']:.3f} [{low}, {high}]"


def _number(value: float | None) -> str:
    return number(value, ".3f", "n/a")


@app.command("e2e-prepare", help="Download and decode a few Objectron test videos (network).")
def e2e_prepare(
    videos: Annotated[int | None, typer.Option(help="Override [sizing.e2e] videos.")] = None,
) -> None:
    context = app_context()
    settings = context.settings
    sizing = sizing_settings(settings)
    plan = sizing.e2e if videos is None else sizing.e2e.model_copy(update={"videos": videos})
    reader = objectron.UrllibPrefixReader()
    prepare = partial(_prepare_clip, settings, sizing, plan)
    draw = e2e_draw(settings, sizing, plan, reader, prepare)
    manifest = _e2e_frames_root(settings, plan) / "manifest.json"
    write_text_atomic(manifest, json.dumps(_manifest(draw.taken, plan), indent=1))
    typer.echo(
        f"{len(draw.taken)} clips prepared, {len(draw.skipped)} videos skipped (> max_video_mb)"
    )
    typer.echo(f"manifest: {manifest}")


def _e2e_frames_root(settings: Settings, plan: EndToEndSettings) -> Path:
    return settings.paths.raw_dir / LOCAL_VIDEO / plan.subset


def _truth_path(settings: Settings, clip_id: str) -> Path:
    return objectron_dir(settings) / "e2e" / f"{clip_id}.json"


def _prepare_clip(
    settings: Settings, sizing: SizingSettings, plan: EndToEndSettings, ref: objectron.VideoRef
) -> bool:
    """Frames and truth of one video; the MOV is deleted once decoded. False if too large."""
    clip_id = e2e.clip_id_of(ref)
    truth_path = _truth_path(settings, clip_id)
    if truth_path.is_file():
        return True
    video_url = f"{sizing.bucket_url}/{objectron.video_prefix(ref)}{VIDEO_FILE}"
    if e2e.remote_size(video_url) > plan.max_video_mb * 1e6:
        return False
    work = objectron_dir(settings) / "e2e_download" / clip_id
    work.mkdir(parents=True, exist_ok=True)
    annotation = work / "annotation.pbdata"
    e2e.download(objectron.annotation_url(sizing.bucket_url, ref), annotation)
    e2e.download(video_url, work / VIDEO_FILE)
    step = e2e.step_of(plan.fps)
    written = e2e.decode_frames(
        work / VIDEO_FILE,
        _e2e_frames_root(settings, plan) / clip_id,
        step=step,
        n_frames=round(plan.seconds * plan.fps),
        max_side=plan.max_side,
    )
    buf = annotation.read_bytes()
    parsed = objectron.sequence_objects(buf, complete=True)
    if parsed is None:
        raise objectron.ObjectronError(f"{ref.video_id}: annotation has no objects")
    truth = e2e.clip_truth(
        ref, objectron.frame_records(buf[parsed[1] :]), step=step, n_frames=written
    )
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(truth_path, truth.to_json())
    shutil.rmtree(work)
    return True


def _manifest(refs: Sequence[objectron.VideoRef], plan: EndToEndSettings) -> dict[str, Any]:
    return {
        "clips": [
            {
                "video": f"{e2e.clip_id_of(ref)}/",
                "real_fps": plan.fps,
                "queries": [{"phrase": ref.label.replace("_", " "), "expected_count": 1}],
            }
            for ref in refs
        ]
    }


@app.command("e2e-eval", help="Perceived scale against ARKit depth on the prepared clips.")
def e2e_eval() -> None:
    context = app_context()
    settings = context.settings
    sizing = sizing_settings(settings)
    source = resolve_dataset(context, LOCAL_VIDEO)
    chain = _e2e_chain(context, source)
    rows: list[e2e.ScaleRow] = []
    per_clip: list[dict[str, Any]] = []
    for clip in source.clips(sizing.e2e.subset):
        record, clip_rows = _e2e_clip(settings, chain, clip, sizing.e2e.min_box_iou)
        per_clip.append(record)
        rows += clip_rows
    excluded: dict[str, int] = {}
    for record in per_clip:
        if "excluded" in record:
            excluded[record["excluded"]] = excluded.get(record["excluded"], 0) + 1
    total = e2e.summarize(rows, n_boot=sizing.bootstrap, seed=sizing.seed, ci=sizing.ci)
    report = {
        "estimated": True,
        "n_clips": len(per_clip),
        "excluded": excluded,
        "min_box_iou": sizing.e2e.min_box_iou,
        "total": total,
        "per_clip": per_clip,
        "camera": "assumed FOV (the true-FOV attribution is not run)",
        "truth_limit": "ARKit depth of the box center; its own error is not documented",
    }
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    write_json(run_dir / "e2e.json", report)
    for method, m in total.items():
        typer.echo(
            f"{method}: n {m['n']}  median|s/s-1| {_v(m['median_abs_rel'])}  "
            f"rms log {_v(m['rms_log'])}  cover 1s {_v(m['cover_1s'])}"
        )
    typer.echo(f"excluded: {excluded}")
    typer.echo(f"run: {run_dir}")


def _e2e_chain(context: AppContext, source: DatasetSource) -> ChainRunner:
    deps = pipeline_deps(context, None, partial(refuse_heavy, LOCAL_VIDEO))
    return ChainRunner(context.settings, context.registry("providers"), deps, source)


def _scale_outputs(
    chain: ChainRunner, clip: ClipManifest
) -> tuple[MaskTrack, ObjectStateTable, dict[str, ScaleEstimate]] | None:
    """Track and lift (shared upstream) and the scale of each method; None if any failed."""
    track, lift, estimates = None, None, {}
    for method in SCALE_METHODS:
        variant = Variant(method, frozenset(), MappingProxyType({Stage.SCALE: method}))
        out = chain.outputs(clip, variant)
        if out is None:
            return None
        track = stage_output(out.outputs, Stage.TRACK, MaskTrack)
        lift = stage_output(out.outputs, Stage.LIFT, ObjectStateTable)
        scale = stage_output(out.outputs, Stage.SCALE, ScaleEstimate)
        if track is None or lift is None or scale is None:
            return None
        estimates[method] = scale
    return (track, lift, estimates) if track is not None and lift is not None else None


def _e2e_clip(
    settings: Settings, chain: ChainRunner, clip: ClipManifest, min_box_iou: float
) -> tuple[dict[str, Any], list[e2e.ScaleRow]]:
    record: dict[str, Any] = {"clip": clip.clip_id}
    path = _truth_path(settings, clip.clip_id)
    if not path.is_file():
        return {**record, "excluded": "no_truth"}, []
    truth = e2e.ClipTruth.from_json(path.read_text(encoding="utf-8"))
    loaded = _scale_outputs(chain, clip)
    if loaded is None:
        return {**record, "excluded": "chain_failed"}, []
    track, lift, estimates = loaded
    match = e2e.match_object(truth, track.masks, track.sampling.frame_indices)
    if match is None or match[1] < min_box_iou:
        iou = None if match is None else match[1]
        return {**record, "excluded": "low_box_iou", "iou": iou}, []
    object_id = str(track.object_ids[match[0]])
    n = lift.object_ids.tolist().index(object_id)
    s_true = e2e.true_scale(truth, lift.position[:, n, 2], lift.frame_index)
    record |= {"label": truth.label, "iou": match[1], "s_true": s_true}
    rows = [
        e2e.ScaleRow(clip.clip_id, method, s_true, est.meters_per_unit, est.sigma)
        for method, est in estimates.items()
        if est.meters_per_unit is not None and math.isfinite(s_true)
    ]
    record["rows"] = [asdict(r) for r in rows]
    return record, rows
