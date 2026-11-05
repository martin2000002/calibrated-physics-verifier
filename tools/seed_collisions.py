"""Seeds of a planned synthetic subset that an already rendered set also uses.

    PYTHONPATH=src python tools/seed_collisions.py --profile precise --subset r1dyn DIR [DIR ...]

The planned seeds are the dataset seed, every scene seed and the seed of every perturbation
attempt of the subset as the configuration (files, ``PHYSVERIFY__*``, ``--set``) plans it. The
rendered seeds are every integer ``seed`` in each ``DIR/manifest.jsonl``. Prints a JSON summary;
exits 1 when a seed collides.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.loader import default_config_dir, load_settings
from physverify.synth.generate import plan_subset
from physverify.synth.plan import NONE
from physverify.synth.seeds import perturbation_seed, scene_seed
from physverify.synth.settings import (
    DatasetSettings,
    load_dataset_settings,
    perturbation_params,
)

MANIFEST = "manifest.jsonl"


def planned_seeds(
    ds: DatasetSettings,
    families: Mapping[str, Any],
    perturbations: Mapping[str, Any],
    subset: str,
) -> dict[int, str]:
    """Seed -> what draws it."""
    plan = plan_subset(ds, families, perturbations, subset)
    seeds = {ds.seed: "dataset"}
    for scene in plan.scenes:
        seeds[scene_seed(ds.seed, scene.family, scene.index)] = scene.scene_id
    for variant in plan.variants:
        if variant.perturbation == NONE:
            continue
        attempts = perturbation_params(perturbations[variant.perturbation], ds).max_attempts
        for attempt in range(attempts):
            seed = perturbation_seed(
                ds.seed, variant.family, variant.index, variant.perturbation, variant.level, attempt
            )
            seeds[seed] = f"{variant.clip_id} attempt {attempt}"
    return seeds


def seeds_in(value: object) -> Iterator[int]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "seed" and isinstance(item, int) and not isinstance(item, bool):
                yield item
            else:
                yield from seeds_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from seeds_in(item)


def rendered_seeds(directory: Path) -> set[int]:
    with (directory / MANIFEST).open(encoding="utf-8") as handle:
        return {seed for line in handle if line.strip() for seed in seeds_in(json.loads(line))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", default=None)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    settings = load_settings(
        default_config_dir(os.environ),
        environ=os.environ,
        profile=args.profile,
        overrides=args.overrides,
    )
    ds = load_dataset_settings(settings)
    planned = planned_seeds(
        ds, PLUGIN_KINDS["families"](), PLUGIN_KINDS["perturbations"](), args.subset
    )
    rendered = {str(d): rendered_seeds(d) for d in args.dirs}
    collisions = sorted(
        f"{planned[seed]} in {d}" for d, seeds in rendered.items() for seed in set(planned) & seeds
    )
    summary = {
        "version": ds.version,
        "dataset_seed": ds.seed,
        "planned": len(planned),
        "rendered": {d: len(seeds) for d, seeds in rendered.items()},
        "collisions": collisions,
    }
    print(json.dumps(summary, indent=1))
    return 1 if collisions else 0


if __name__ == "__main__":
    sys.exit(main())
