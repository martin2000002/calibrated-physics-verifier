"""Warm the chain cache, build the pair image table, collect a learned variant's training
samples, then train it.

    PYTHONPATH=src python tools/train_learned.py warm --dataset D --subset S [--truth-share 0.5]
    PYTHONPATH=src python tools/train_learned.py images --dataset D --subset S \
        --out pairs.npz [--truth-share 0.5] [--single-frame] [--dry-run] [--set key=value ...]
    PYTHONPATH=src python tools/train_learned.py samples --check R1 --dataset D --subset S \
        --images pairs.npz --out samples.npz [--set key=value ...]
    PYTHONPATH=src python tools/train_learned.py train --check R1 --samples samples.npz \
        --dataset D --subset S --encoder-sha SHA --evaluated MANIFEST... --out DIR
    PYTHONPATH=src python tools/train_learned.py plan --check R1 --dataset D --subset S \
        --out plan.json [--truth-share 0.5] [--evaluated MANIFEST...]
    PYTHONPATH=src python tools/train_learned.py gates --check R1 --dataset D --subset S \
        --artifact A.npz --oracle TRUTH.npz --perceived PERCEIVED.npz --single-frame SINGLE.npz \
        [--hdri-dataset D --hdri-subset S] --out gates.json
    PYTHONPATH=src python tools/train_learned.py gates --dry-run --check R1 --dataset D \
        --subset S [--hdri-dataset D --hdri-subset S] --out populations.json

``warm`` runs each clip's chain under its drawn substitution arm; it is the only command that
may run a heavy stage, the others read the cache and fail a clip whose stage is not cached.
``images`` reads each clip's masks and depth under its drawn arm and its frames from the video,
and encodes every pair's crop with the VDA-S encoder (refused unless its checkpoint hash
matches; ``--single-frame``: each row reads the middle sampled frame alone, for the single-frame
control); ``--dry-run`` stops before the encoder and writes the counts of what it would read as
JSON instead. ``samples`` reads the same drawn arm. Every command but ``train`` refuses a sealed
dataset and takes ``--truth-share`` (1: every stage from ground truth, 0: none). ``train``
trains on split=train clips only, refuses any evaluated manifest that shares a perturbation seed
with the training clips, then writes ``<sha256>.npz`` (the artifact, named by its hash) and
``training_log.json`` into ``--out``. ``plan`` is the dry run of ``samples`` and ``train`` for one
check: clips by arm, role and validation split, clips without cached inputs, the leakage guard
and the gate and control populations, as JSON; it reads no sample value and trains nothing.
``gates`` scores the internal validation clips of the training selection with the artifact, from
samples of the ground-truth arm (``--truth-share 1``), the perceived arm (``--truth-share 0``) and
the perceived arm's single-frame rows (C-hdri splits those validation clips by whether training
saw their background; ``--hdri-dataset/--hdri-subset``, when given, must name the training
selection itself); it writes every gate and control with its interval, population, unscored
clips per side and pass or fail as JSON. ``--dry-run`` writes the populations from the
manifests alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from physverify.app.context import AppContext, GlobalOptions, Invocation
from physverify.app.learned import (
    collect,
    collect_images,
    fit,
    learned_check,
    plan_images,
    plan_training,
    training_clips,
    training_seeds,
    warm,
)
from physverify.app.learned_gates import GateFiles, GateSelections, gate_report, plan_gates
from physverify.app.services import dataset_for
from physverify.core.feature_table import FeatureTable
from physverify.extraction.contact_crops import CropSettings
from physverify.extraction.crop_encoder import vda_crop_encoder
from physverify.extraction.pair_features import dense_columns
from physverify.learned.samples import SampleSet
from physverify.learned.train import TrainSettings
from physverify.synth.seed_audit import manifest_seeds

REPO = Path(__file__).resolve().parents[1]
ARM_SEED = 0
TRUTH_SHARE = 0.5


def _context(args: argparse.Namespace) -> AppContext:
    options = GlobalOptions(REPO / "configs", profile=args.profile, overrides=tuple(args.set))
    return AppContext(Invocation(tuple(sys.argv), os.environ), options)


def _stderr(line: str) -> None:
    print(line, file=sys.stderr)


def _warm(args: argparse.Namespace) -> int:
    app = _context(args)
    failures = warm(app, (args.dataset, args.subset), (ARM_SEED, args.truth_share), err=_stderr)
    for failure in failures:
        _stderr(f"{failure.clip_id} [{failure.variant}] {failure.stage}: {failure.error}")
    print(f"{len(failures)} clip failures")
    return 0


def _images(args: argparse.Namespace) -> int:
    app = _context(args)
    selection = (args.dataset, args.subset)
    arm_draw = (ARM_SEED, args.truth_share)
    if args.dry_run:
        plan = plan_images(app, selection, arm_draw, err=_stderr)
        args.out.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
        print(f"{plan.clips} clips, {plan.pairs} pairs, {len(plan.failed)} without inputs")
        return 0
    encoder = vda_crop_encoder(app.settings.paths)
    found = collect_images(
        app,
        selection,
        arm_draw,
        (encoder, CropSettings()),
        err=_stderr,
        single_frame=args.single_frame,
    )
    found.table.save(args.out)
    print(f"{len(found.table.keys)} pair rows, {len(found.failed)} clips without inputs")
    return 0


def _samples(args: argparse.Namespace) -> int:
    app = _context(args)
    table = FeatureTable.load(args.images)
    dense = dense_columns()
    tokens = tuple(c for c in table.columns if c not in dense)
    found = collect(
        app,
        (args.dataset, args.subset),
        args.check,
        (table, dense, tokens),
        (ARM_SEED, args.truth_share),
        err=_stderr,
    )
    found.samples.to_table().save(args.out)
    print(f"{len(found.samples.samples)} samples, {len(found.failed)} clips without outputs")
    return 0


def _train(args: argparse.Namespace) -> int:
    app = _context(args)
    samples = SampleSet.from_table(FeatureTable.load(args.samples))
    clips = training_clips(dataset_for(app, args.dataset).clips(args.subset))
    unit = type(learned_check(app, args.check)).magnitude_unit
    evaluated = [frozenset(manifest_seeds(path)) for path in args.evaluated]
    trained = fit(
        samples,
        (args.check, unit, args.encoder_sha),
        training_seeds(clips),
        evaluated,
        TrainSettings(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    staged = args.out / "artifact.npz"
    sha = trained.artifact.save(staged)
    staged.rename(args.out / f"{sha}.npz")
    log = {"artifact_sha256": sha, "settings": asdict(TrainSettings()), **asdict(trained.log)}
    (args.out / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"artifact {sha}, best epoch {trained.log.best_epoch}")
    return 0


def _plan(args: argparse.Namespace) -> int:
    app = _context(args)
    evaluated = [frozenset(manifest_seeds(path)) for path in args.evaluated]
    selection = (args.dataset, args.subset)
    plan = plan_training(
        app, selection, args.check, (ARM_SEED, args.truth_share), evaluated, err=_stderr
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
    print(
        f"{args.check}: {plan.clips} clips, roles {plan.roles}, validation "
        f"{plan.validation_roles}, {len(plan.without_inputs)} without inputs, "
        f"{plan.leakage_checked} manifests disjoint"
    )
    return 0


def _gates(args: argparse.Namespace) -> int:
    app = _context(args)
    hdri = None if args.hdri_dataset is None else (args.hdri_dataset, args.hdri_subset)
    selections = GateSelections((args.dataset, args.subset), hdri)
    if args.dry_run:
        found = plan_gates(app, selections, args.check, err=_stderr)
    else:
        missing = [
            n for n in ("artifact", "oracle", "perceived", "single_frame") if not getattr(args, n)
        ]
        if missing:
            raise SystemExit(
                f"gates needs {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        files = GateFiles(args.artifact, args.oracle, args.perceived, args.single_frame)
        found = gate_report(app, selections, args.check, files, err=_stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found, indent=2), encoding="utf-8")
    print(f"{args.check} populations: {json.dumps(found['populations'])}")
    if "verdict" in found:
        print(f"{args.check} verdict: {json.dumps(found['verdict'])}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--set", action="append", default=[])
    commands = parser.add_subparsers(dest="command", required=True)
    warm_cmd = commands.add_parser("warm")
    images = commands.add_parser("images")
    samples = commands.add_parser("samples")
    train = commands.add_parser("train")
    plan = commands.add_parser("plan")
    gates = commands.add_parser("gates")
    for sub in (warm_cmd, images, samples, plan):
        sub.add_argument("--truth-share", type=float, default=TRUTH_SHARE)
    for sub in (warm_cmd, images):
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--subset", default=None)
    images.add_argument("--out", required=True, type=Path)
    images.add_argument("--dry-run", action="store_true")
    images.add_argument("--single-frame", action="store_true")
    for sub in (samples, train, plan, gates):
        sub.add_argument("--check", required=True)
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--subset", default=None)
        sub.add_argument("--out", required=True, type=Path)
    samples.add_argument("--images", required=True, type=Path)
    train.add_argument("--samples", required=True, type=Path)
    train.add_argument("--encoder-sha", required=True)
    train.add_argument("--evaluated", nargs="+", required=True, type=Path)
    plan.add_argument("--evaluated", nargs="*", default=[], type=Path)
    gates.add_argument("--dry-run", action="store_true")
    for name in ("--artifact", "--oracle", "--perceived", "--single-frame"):
        gates.add_argument(name, type=Path, default=None)
    gates.add_argument("--hdri-dataset", default=None)
    gates.add_argument("--hdri-subset", default=None)
    args = parser.parse_args(argv)
    commands_by_name = {
        "warm": _warm,
        "images": _images,
        "samples": _samples,
        "train": _train,
        "plan": _plan,
        "gates": _gates,
    }
    run = commands_by_name[args.command]
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
