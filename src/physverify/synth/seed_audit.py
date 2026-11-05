"""Seed disjointness of a planned subset against datasets already built: every attempt seed a
planned clip can draw is compared with the ``perturbation.seed`` of every manifest row."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from physverify.synth.perturb import seed_stream
from physverify.synth.plan import DatasetPlan
from physverify.synth.seeds import perturbation_seed
from physverify.synth.settings import DatasetSettings, PerturbationClass, perturbation_params


@dataclass(frozen=True)
class Collision:
    seed: int
    clip_id: str
    manifest: str


def planned_seeds(
    plan: DatasetPlan, perturbations: Mapping[str, PerturbationClass], ds: DatasetSettings
) -> dict[int, str]:
    """Each seed any attempt of a planned clip may draw, with the clip that draws it."""
    out: dict[int, str] = {}
    for v in plan.variants:
        plugin = perturbations[v.perturbation]
        stream = seed_stream(plugin)
        for attempt in range(perturbation_params(plugin, ds).max_attempts):
            seed = perturbation_seed(ds.seed, v.family, v.index, stream, v.level, attempt)
            out.setdefault(seed, v.clip_id)
    return out


def manifest_seeds(path: Path) -> Iterator[int]:
    """The ``perturbation.seed`` of every row of a ``manifest.jsonl`` (other fields unread)."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                yield int(row["data"]["perturbation"]["seed"])


def collisions(planned: Mapping[int, str], manifests: Iterable[Path]) -> list[Collision]:
    found: list[Collision] = []
    for path in manifests:
        found.extend(
            Collision(seed, planned[seed], str(path))
            for seed in sorted(set(manifest_seeds(path)))
            if seed in planned
        )
    return found
