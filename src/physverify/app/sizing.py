"""Wiring of the size model's end-to-end check from the application context."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

from physverify.app.context import AppContext
from physverify.app.services import chain_runner, dataset_for, pipeline_deps
from physverify.config.models import SIZE_PRIORS_SECTION, Settings, SizePriorTable
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import MissingInputError
from physverify.core.output import Echo
from physverify.datasets._base import DatasetSource
from physverify.datasets.local_video import NAME as LOCAL_VIDEO
from physverify.diagnosis.chain_runner import ChainRunner
from physverify.diagnosis.chain_variants import Variant
from physverify.diagnosis.measurement_cases import stage_output
from physverify.extraction.providers.scale_size_model import SizeModelScale
from physverify.extraction.providers.scale_size_prior import SizePriorScale
from physverify.runtime.runs import new_run, write_json
from physverify.schema.clip import ClipManifest
from physverify.schema.perception import MaskTrack
from physverify.schema.scale import ScaleEstimate
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable
from physverify.sizing import end_to_end, objectron
from physverify.sizing.evaluate import MODEL, evaluate
from physverify.sizing.fit import fit_model, fit_record, render_toml
from physverify.sizing.model import SECTION as MODEL_SECTION
from physverify.sizing.model import SizeModelTable
from physverify.sizing.objectron_draws import (
    TEST,
    TRAIN,
    EndToEndDraw,
    load_samples,
    objectron_dir,
    selection,
)
from physverify.sizing.settings import EndToEndSettings, sizing_settings
from physverify.verification.inputs import refuse_heavy

SCALE_METHODS = (SizePriorScale.name, SizeModelScale.name)
MODEL_FILE = "verifier/size_model.toml"


def fetch_heads(settings: Settings, *, echo: Echo) -> None:
    """Store the leading box bytes of every selected Objectron video, one line per split."""
    sizing = sizing_settings(settings)
    reader = objectron.UrllibPrefixReader()
    plan = objectron.HeadFetch(
        sizing.bucket_url, objectron_dir(settings) / "heads", sizing.head_bytes, sizing.retries
    )
    for split in objectron.SPLITS:
        refs = selection(settings, sizing, split, reader)
        stored = objectron.fetch_heads(reader, plan, refs, sizing.workers)
        echo(f"{split}: {len(refs)} videos, {stored} new bytes stored")


def fit_and_write(context: AppContext) -> tuple[SizeModelTable, Path, Path]:
    """Fit the model on train and write it to the config directory; returns the model, its file
    and the run directory holding ``fit.json``."""
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
    write_json(run_dir / "fit.json", fit_record(model, train, sizing))
    return model, target, run_dir


def evaluate_test_split(context: AppContext) -> tuple[dict[str, Any], Path]:
    """Error on the official test split against B0 and B1; writes the run's ``eval.json``."""
    settings = context.settings
    sizing = sizing_settings(settings)
    model = settings.section(MODEL_SECTION, SizeModelTable)
    if not model.classes:
        raise MissingInputError(f"configs/{MODEL_FILE} is empty; run `size fit` first")
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
    return report, run_dir


def e2e_frames_root(settings: Settings, plan: EndToEndSettings) -> Path:
    return settings.paths.raw_dir / LOCAL_VIDEO / plan.subset


def prepare_end_to_end(settings: Settings, videos: int | None) -> tuple[EndToEndDraw, Path]:
    """Download and decode the drawn Objectron test videos as ``local_video`` clips."""
    sizing = sizing_settings(settings)
    plan = sizing.e2e if videos is None else sizing.e2e.model_copy(update={"videos": videos})
    reader = objectron.UrllibPrefixReader()
    return end_to_end.prepare(settings, sizing, plan, e2e_frames_root(settings, plan), reader)


def evaluate_end_to_end(context: AppContext) -> tuple[dict[str, Any], Path]:
    """Perceived scale of each method against ARKit depth; writes the run's ``e2e.json``."""
    settings = context.settings
    sizing = sizing_settings(settings)
    source = dataset_for(context, LOCAL_VIDEO)
    chain = e2e_chain(context, source)
    clips = source.clips(sizing.e2e.subset)
    report = end_to_end.evaluate_clips(settings, sizing, clips, partial(scale_outputs, chain))
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    write_json(run_dir / "e2e.json", report)
    return report, run_dir


def e2e_chain(context: AppContext, source: DatasetSource) -> ChainRunner:
    deps = pipeline_deps(context, None, partial(refuse_heavy, LOCAL_VIDEO))
    return chain_runner(context, deps, source)


def scale_outputs(
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
