"""AC1, AC2 and AC19: the real provider registry, its execution contract and lazy heavy imports."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from cli_runner import CliRunner

from physverify.cli.context import AppContext, GlobalOptions, Invocation
from physverify.extraction._base import StageProvider, validate_provider
from physverify.extraction.cache import reads_own_error
from physverify.schema.stages import Stage, check_upstream_order

REPO = Path(__file__).resolve().parents[2]
PROVIDERS_DIR = REPO / "src" / "physverify" / "extraction" / "providers"
LAZY_MODULES = ("torch", "transformers", "huggingface_hub", "timm")
EXPECTED = {
    "detect/grounding_dino_tiny",
    "detect/grounding_dino_tiny_per_query",
    "detect/grounding_dino_base",
    "track/edgetam",
    "track/sam2_1_hiera_tiny",
    "depth/vda_small",
    "depth/vda_metric_small",
    "camera/assumed_fov",
    "lift/backproject",
    "scale/size_prior",
    "scale/size_model",
    "scale/free_fall",
    "kinematics/savgol",
    "contacts/proximity",
    *(f"{stage.value}/ground_truth" for stage in Stage),
}
HEAVY = {
    "detect/grounding_dino_tiny",
    "detect/grounding_dino_tiny_per_query",
    "detect/grounding_dino_base",
    "track/edgetam",
    "track/sam2_1_hiera_tiny",
    "depth/vda_small",
    "depth/vda_metric_small",
}


@pytest.fixture(scope="module")
def registry() -> dict[str, type[StageProvider]]:
    context = AppContext(
        Invocation(argv=("physverify",), environ={}), GlobalOptions(config_dir=REPO / "configs")
    )
    return dict(context.registry("providers"))


def test_registry_has_exactly_the_twenty_two_providers(
    registry: dict[str, type[StageProvider]],
) -> None:
    assert len(EXPECTED) == 22
    assert set(registry) == EXPECTED


def test_every_provider_validates_and_only_perception_models_are_heavy(
    registry: dict[str, type[StageProvider]],
) -> None:
    for provider in registry.values():
        validate_provider(provider)
        check_upstream_order(provider.stage, provider.error_profile_reads)
    assert {key for key, p in registry.items() if p.heavy} == HEAVY


def test_only_backproject_reads_its_own_profile_entry(
    registry: dict[str, type[StageProvider]],
) -> None:
    lift = registry["lift/backproject"]
    assert lift.error_profile_reads == (Stage.TRACK, Stage.DEPTH)
    assert reads_own_error(lift)
    others = [
        key for key, p in registry.items() if key != "lift/backproject" and reads_own_error(p)
    ]
    assert others == []


@pytest.mark.parametrize("command", ["run", "fetch-weights"])
def test_extract_help_goes_through_the_real_registry(cli: CliRunner, command: str) -> None:
    result = cli("extract", command, "--help")
    assert result.code == 0, result.err


def test_building_the_registry_imports_no_heavy_library() -> None:
    script = (
        "import sys\n"
        "from physverify.cli.context import PLUGIN_KINDS\n"
        "import physverify.extraction.weights, physverify.extraction.pipeline\n"
        "assert len(PLUGIN_KINDS['providers']()) == 22\n"
        f"print(sorted(m for m in {LAZY_MODULES!r} if m in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert done.stdout.strip() == "[]"


def _module_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _sources() -> list[Path]:
    extra = [PROVIDERS_DIR.parent / "weights.py", PROVIDERS_DIR.parent / "pipeline.py"]
    return sorted(PROVIDERS_DIR.glob("*.py")) + extra


def test_heavy_libraries_are_imported_only_inside_functions() -> None:
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not _module_level_imports(tree) & set(LAZY_MODULES), path.name


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _is_attr(node: ast.AST, base: str, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == base
    )


def test_cpu_literal_appears_only_as_a_device_comparison() -> None:
    for path in sorted(PROVIDERS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = _parents(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "cpu":
                parent = parents[node]
                assert isinstance(parent, ast.Compare), path.name
                assert _is_attr(parent.left, "ctx", "device"), path.name


def test_frames_are_requested_with_the_context_sampling() -> None:
    """``ctx.video.frames(s.start, <stop>, s.step, s.max_side)``; <stop> is ``s.stop`` except in
    DINO, which reads only the first ``search_frames`` sampled frames."""
    calls = 0
    for path in sorted(PROVIDERS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "frames":
                continue
            calls += 1
            assert _is_attr(node.func.value, "ctx", "video"), path.name
            start, stop, step, max_side = node.args
            assert _is_attr(start, "s", "start") and _is_attr(step, "s", "step"), path.name
            assert _is_attr(max_side, "s", "max_side"), path.name
            dino = path.name == "detect_grounding_dino.py"
            assert _is_attr(stop, "s", "stop") or (dino and isinstance(stop, ast.Name)), path.name
    assert calls == 3
