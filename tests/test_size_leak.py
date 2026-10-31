"""F2: the verifier's size knowledge never comes from the synthetic generator's size table."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest

from physverify.config.loader import load_settings
from physverify.config.models import SIZE_PRIORS_SECTION, SizePrior, SizePriorTable
from physverify.sizing.model import SECTION as SIZE_MODEL_SECTION
from physverify.sizing.model import SizeModelTable, class_prior
from physverify.sizing.settings import sizing_settings
from physverify.synth.settings import SIZE_DISTRIBUTION_SECTION, load_size_distribution

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "physverify"
GENERATOR = PACKAGE / "synth"
CONTRACT = "The estimator never reads the synthetic generator"
ESTIMATOR_TABLES = ("SIZE_PRIORS_SECTION", f'section("{SIZE_PRIORS_SECTION}"', "physverify.sizing")


def _sources(root: Path) -> Iterator[Path]:
    return (path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _mentions(path: Path, *needles: str) -> bool:
    text = path.read_text(encoding="utf-8")
    return any(needle in text for needle in needles)


def _estimator_sizes() -> dict[str, SizePrior]:
    settings = load_settings(REPO / "configs", environ={})
    model = settings.section(SIZE_MODEL_SECTION, SizeModelTable)
    fixed = sizing_settings(settings).fixed
    return {
        **settings.section(SIZE_PRIORS_SECTION, SizePriorTable).root,
        **{name: class_prior(size, model.sigma_factor) for name, size in model.classes.items()},
        **{name: SizePrior(mean_m=f.mean_m, rel_spread=f.rel_spread) for name, f in fixed.items()},
    }


def test_the_estimator_import_ban_is_a_contract() -> None:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert CONTRACT in text


def test_only_the_generator_names_its_size_table() -> None:
    outside = [
        str(path.relative_to(REPO))
        for path in _sources(PACKAGE)
        if not path.is_relative_to(GENERATOR)
        and _mentions(path, SIZE_DISTRIBUTION_SECTION, "SIZE_DISTRIBUTION_SECTION")
    ]
    assert outside == []


def test_the_generator_never_reads_the_estimator_tables() -> None:
    """Family params keep their field name ``size_priors`` (dataset plans hash it); what fills it
    is ``[size_distribution]``."""
    inside = [
        str(path.relative_to(REPO))
        for path in _sources(GENERATOR)
        if _mentions(path, *ESTIMATOR_TABLES)
    ]
    assert inside == []


@pytest.mark.parametrize("field", ["mean_m", "rel_spread"])
def test_no_estimator_size_equals_a_generator_size(field: str) -> None:
    settings = load_settings(REPO / "configs", environ={})
    generator = {getattr(p, field) for p in load_size_distribution(settings).values()}
    shared = {
        name: getattr(prior, field)
        for name, prior in _estimator_sizes().items()
        if any(math.isclose(getattr(prior, field), value, rel_tol=1e-3) for value in generator)
    }
    assert shared == {}
