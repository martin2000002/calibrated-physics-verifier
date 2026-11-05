"""Refuse a subset whose seeds an existing dataset already drew.

    PYTHONPATH=src python tools/seed_collisions.py --profile precise --subset SUBSET MANIFEST...

Settings come from ``configs/`` and ``PHYSVERIFY__*`` like any command, so the dataset seed and
version a launcher exports are the ones checked. Prints the seed and one line per collision; exit 1
when there is any.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.loader import load_settings
from physverify.synth.generate import plan_subset
from physverify.synth.seed_audit import collisions, planned_seeds
from physverify.synth.settings import load_dataset_settings

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--subset", required=True)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args(argv)
    settings = load_settings(REPO / "configs", environ=os.environ, profile=args.profile)
    ds = load_dataset_settings(settings)
    perturbations = PLUGIN_KINDS["perturbations"]()
    plan = plan_subset(ds, PLUGIN_KINDS["families"](), perturbations, args.subset)
    planned = planned_seeds(plan, perturbations, ds)
    found = collisions(planned, args.manifests)
    print(
        f"dataset seed {ds.seed}, version {ds.version}: {len(plan.variants)} clips, "
        f"{len(planned)} attempt seeds, {len(args.manifests)} manifests, {len(found)} collisions"
    )
    for hit in found:
        print(f"collision: seed {hit.seed} of {hit.clip_id} in {hit.manifest}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
