"""Names that reach disk: schema ids and versions, record and error class names, file names,
run.json fields and the sealed-ledger command labels."""

from __future__ import annotations

import ast
import builtins
import datetime as dt
import json
from collections.abc import Iterator
from functools import cache
from pathlib import Path

from physverify.calibration.devscore import DEVSCORE_FILE, DEVSCORE_SCHEMA, DEVSCORE_VERSION
from physverify.calibration.exclusion_record import RECORD_SCHEMA, RECORD_VERSION
from physverify.calibration.schema import (
    CALIBRATION_SCHEMA,
    CALIBRATION_VERSION,
    EVALUATION_SCHEMA,
)
from physverify.cli.commands._sealed import GUARDED
from physverify.reporting.render import RESULTS_SCHEMA, RESULTS_VERSION
from physverify.runtime.runs import RunContext
from physverify.schema.io import SCHEMA_VERSION as RECORD_FORMAT_VERSION
from physverify.synth.plan import GT_MIGRATIONS, GT_REVISION
from physverify.timing.records import SCHEMA_VERSION as TIMING_SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[2]
CODE_ROOTS = (REPO / "src", REPO / "scripts")

RECORD_CLASSES = frozenset(
    {
        "CameraModel",
        "CameraSpec",
        "CheckResult",
        "ClipManifest",
        "ContactRecord",
        "ContactTable",
        "DepthMap",
        "Detections",
        "EffectiveScale",
        "ErrorEntry",
        "ErrorProfile",
        "FrameSampling",
        "IdMap",
        "MaskTrack",
        "ObjectQuery",
        "ObjectSpec",
        "ObjectStateTable",
        "PerturbationRecord",
        "RawCollisions",
        "RenderLayers",
        "SamplingSpec",
        "ScaleEstimate",
        "SceneSpec",
        "StageArtifactMeta",
    }
)

ERROR_CLASSES = frozenset(
    {
        "BudgetError",
        "CalibrationDataError",
        "CalibrationError",
        "ChecksumError",
        "ConfigError",
        "DuplicatePluginError",
        "ExclusionInputError",
        "ExistingRunError",
        "GenerationError",
        "IncompleteDownloadError",
        "InputsError",
        "InvalidPluginError",
        "KSigmaMismatchError",
        "LedgerError",
        "LockBusyError",
        "MissingInputError",
        "MissingPluginsError",
        "MixedRowStepError",
        "NotImplementedCommandError",
        "ObjectronError",
        "PerturbationNotApplicableError",
        "PhysverifyError",
        "PluginError",
        "QueryError",
        "RejectedSampleError",
        "ResourceBudgetError",
        "ResultsInputError",
        "ScaleBindingError",
        "ScheduleError",
        "SchemaError",
        "SchemaVersionError",
        "SealedSetError",
        "StaleCalibrationError",
        "TimingError",
        "UnitMismatchError",
        "UnreachableTargetError",
        "UnusableCalibrationError",
        "UnusableStratumError",
        "UsageError",
        "WireError",
        "_HookError",
    }
)

FILE_NAMES = (
    ".head",
    ".latents.safetensors",
    ".meta.json",
    ".mp4",
    ".part",
    ".queries.json",
    ".safetensors",
    ".safetensors.index.json",
    ".test_comparisons.jsonl",
    "_z",
    "{:06d}.png",
    "abstention.json",
    "abstention_report.json",
    "beta_grid.json",
    "calibration.json",
    "calibration.md",
    "camera.json",
    "checks.jsonl",
    "checks.toml",
    "checks_hash.txt",
    "clip_info.json",
    "closest.npz",
    "config.json",
    "contacts_camera.json",
    "contacts_provisional.json",
    "contacts_world.json",
    "curve.csv",
    "curves.csv",
    "devscore.json",
    "e2e.json",
    "errors.csv",
    "eval.json",
    "evaluation.json",
    "evaluation.md",
    "events.jsonl",
    "extraction.jsonl",
    "failed.json",
    "failures.jsonl",
    "fit.json",
    "gate.json",
    "gt_meta.json",
    "h3_report.json",
    "h3_report.md",
    "index.json",
    "input.csv",
    "items.csv",
    "kubric_meta.json",
    "layers.npz",
    "local.toml",
    "manifest.json",
    "manifest.jsonl",
    "meta.json",
    "metrics.csv",
    "metrics.json",
    "model.safetensors",
    "output.csv",
    "overlay.mp4",
    "pairs.jsonl",
    "params.json",
    "per_check.csv",
    "per_generator.csv",
    "perturbation.json",
    "pilot_review.json",
    "plan.json",
    "preregistration.json",
    "prompts.jsonl",
    "pytorch_lora_weights.safetensors",
    "raw_collisions.npz",
    "record.json",
    "report.json",
    "report.md",
    "results.json",
    "results.md",
    "run.json",
    "samples.csv",
    "scenes.md",
    "scores.jsonl",
    "selected.json",
    "selection.json",
    "side_by_side.mp4",
    "sim_meta.json",
    "size_model.toml",
    "skipped.json",
    "spec.json",
    "specificity.json",
    "stage_errors.json",
    "stage_errors.recalibrated.json",
    "stage_errors.{profile}.json",
    "stage_sigma.json",
    "state.env",
    "states.npz",
    "states_camera.npz",
    "states_variant.npz",
    "states_world.npz",
    "stats.csv",
    "status.json",
    "steps.tsv",
    "stop_rule.json",
    "summary.md",
    "sweep_matrix.csv",
    "table.csv",
    "thresholds.json",
    "trainer.pt",
    "val.csv",
    "val_pairs.jsonl",
    "val_report.json",
    "video.MOV",
)

RUN_RECORD_FIELDS = (
    "run_id",
    "run_dir",
    "argv",
    "device",
    "profile",
    "config_hash",
    "commit",
    "created_utc",
)

LEDGER_LABELS = frozenset(
    {
        "abstain curve",
        "abstain fit",
        "calibrate devscore",
        "calibrate evaluate",
        "calibrate fit",
        "calibrate stage-sigma",
        "diagnose attribution",
        "diagnose end-to-end",
        "diagnose real",
        "diagnose stages",
        "scene-stats per-clip",
        "verify",
    }
)


def _modules() -> Iterator[ast.Module]:
    for root in CODE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield ast.parse(path.read_text(encoding="utf-8"))


@cache
def _class_bases() -> dict[str, frozenset[str]]:
    bases: dict[str, set[str]] = {}
    for module in _modules():
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                names = {_base_name(base) for base in node.bases}
                bases.setdefault(node.name, set()).update(n for n in names if n)
    return {name: frozenset(found) for name, found in bases.items()}


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _descendants(roots: frozenset[str]) -> frozenset[str]:
    found = set(roots)
    bases = _class_bases()
    while True:
        new = {name for name, parents in bases.items() if parents & found} - found
        if not new:
            return frozenset(found - roots)
        found |= new


@cache
def _string_constants() -> frozenset[str]:
    return frozenset(
        node.value
        for module in _modules()
        for node in ast.walk(module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def _module_constant(name: str) -> set[str]:
    return {
        node.value.value
        for module in _modules()
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    }


def test_schema_ids_and_versions_are_unchanged() -> None:
    assert (DEVSCORE_SCHEMA, DEVSCORE_VERSION, DEVSCORE_FILE) == (
        "physverify.devscore",
        "1.1.0",
        "devscore.json",
    )
    assert (RECORD_SCHEMA, RECORD_VERSION) == ("physverify.exclusions", "1.0.0")
    assert (CALIBRATION_SCHEMA, CALIBRATION_VERSION, EVALUATION_SCHEMA) == (
        "physverify.calibration",
        "1.6.0",
        "physverify.calibration.evaluation",
    )
    assert (RESULTS_SCHEMA, RESULTS_VERSION) == ("physverify.results", "1.0.0")
    assert RECORD_FORMAT_VERSION == "1.0"
    assert TIMING_SCHEMA_VERSION == "physverify.timing/1"


def test_the_freeze_manifest_schema_is_unchanged() -> None:
    assert _module_constant("FREEZE_SCHEMA") == {"physverify/freeze"}
    assert _module_constant("FREEZE_VERSION") == {"1.0.0"}


def test_the_ground_truth_revision_is_unchanged() -> None:
    assert GT_REVISION == 2
    assert GT_MIGRATIONS == ("floor-crossing", "pair-collision")


def test_record_class_names_are_unchanged() -> None:
    assert _descendants(frozenset({"Record"})) == RECORD_CLASSES


def test_error_class_names_are_unchanged() -> None:
    builtin_errors = frozenset(
        name
        for name in dir(builtins)
        if isinstance(getattr(builtins, name), type)
        and issubclass(getattr(builtins, name), BaseException)
    )
    assert _descendants(builtin_errors) == ERROR_CLASSES


def test_persisted_file_names_still_occur_in_the_code() -> None:
    constants = _string_constants()
    missing = [name for name in FILE_NAMES if not any(name in text for text in constants)]
    assert missing == []


def test_run_json_fields_are_unchanged(tmp_path: Path) -> None:
    run = RunContext.create(
        tmp_path,
        argv=("verify",),
        device="cpu",
        profile="fast",
        config_hash="0" * 64,
        commit="",
        now=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
    )
    record = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
    assert tuple(record) == RUN_RECORD_FIELDS


def test_sealed_ledger_command_labels_are_unchanged() -> None:
    assert frozenset(GUARDED) == LEDGER_LABELS
