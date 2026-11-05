"""tools/seed_collisions.py: the seeds a subset plans, and the ones a rendered set already uses."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from physverify.synth.seeds import perturbation_seed, scene_seed
from physverify.synth.settings import DatasetSettings, SubsetSettings

TOOL = Path(__file__).resolve().parents[2] / "tools" / "seed_collisions.py"
SUBSET = SubsetSettings(
    scenes_per_family=3,
    levels=(1, 2, 3, 4),
    level_mode="cycle",
    perturbations={"f5_contact": ("r1_pass_through",)},
)
ATTEMPTS = 8


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("seed_collisions_tool", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def subset_ds(ds: DatasetSettings) -> DatasetSettings:
    return ds.model_copy(update={"subsets": {"r1dyn": SUBSET}, "calib_fraction": 1.0})


def test_every_scene_and_attempt_seed_is_planned(
    tool: ModuleType, subset_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    planned = tool.planned_seeds(subset_ds, families, perturbations, "r1dyn")
    seed = subset_ds.seed
    assert planned[seed] == "dataset"
    assert len(planned) == 1 + 3 + 3 * ATTEMPTS
    for index in range(3):
        assert scene_seed(seed, "f5_contact", index) in planned
        last = perturbation_seed(seed, "f5_contact", index, "r1_pass_through", index % 4 + 1, 7)
        assert planned[last].endswith("attempt 7")


def test_a_rendered_seed_is_found_at_any_depth(tool: ModuleType, tmp_path: Path) -> None:
    lines = [{"seed": 3, "record": {"params": {"seed": 9}}}, {"seeds": [{"seed": 4}]}]
    (tmp_path / tool.MANIFEST).write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert tool.rendered_seeds(tmp_path) == {3, 4, 9}
