"""Collect training samples of a learned variant, then train its artifact.

    PYTHONPATH=src python tools/train_learned.py samples --check R1 --dataset D --subset S \
        --images pairs.npz --out samples.npz [--set key=value ...]
    PYTHONPATH=src python tools/train_learned.py train --check R1 --samples samples.npz \
        --dataset D --subset S --encoder-sha SHA --evaluated MANIFEST... --out DIR

``samples`` reads the pipeline cache under the clip's drawn substitution arm and refuses a
sealed dataset. ``train`` refuses any evaluated manifest that shares a perturbation seed with the
training clips, then writes ``<sha256>.npz`` (the artifact, named by its hash) and
``training_log.json`` into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from physverify.app.context import AppContext, GlobalOptions, Invocation
from physverify.app.learned import collect, fit, learned_check, training_seeds
from physverify.app.services import dataset_for
from physverify.core.feature_table import FeatureTable
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
        (ARM_SEED, TRUTH_SHARE),
        err=lambda line: print(line, file=sys.stderr),
    )
    found.samples.to_table().save(args.out)
    print(f"{len(found.samples.samples)} samples, {len(found.failed)} clips without outputs")
    return 0


def _train(args: argparse.Namespace) -> int:
    app = _context(args)
    samples = SampleSet.from_table(FeatureTable.load(args.samples))
    clips = dataset_for(app, args.dataset).clips(args.subset)
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--set", action="append", default=[])
    commands = parser.add_subparsers(dest="command", required=True)
    samples = commands.add_parser("samples")
    train = commands.add_parser("train")
    for sub in (samples, train):
        sub.add_argument("--check", required=True)
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--subset", default=None)
        sub.add_argument("--out", required=True, type=Path)
    samples.add_argument("--images", required=True, type=Path)
    train.add_argument("--samples", required=True, type=Path)
    train.add_argument("--encoder-sha", required=True)
    train.add_argument("--evaluated", nargs="+", required=True, type=Path)
    args = parser.parse_args(argv)
    return _samples(args) if args.command == "samples" else _train(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
