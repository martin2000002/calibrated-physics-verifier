"""Field names, in order, of the models that are written to disk or hashed: artifacts, reports,
records, the sealed ledger, the dataset plan and settings, and every check and provider Params."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.artifact import CalibrationArtifact, EvaluationReport
from physverify.calibration.dev_score import DevScoreReport
from physverify.calibration.sealed_ledger import LedgerEntry
from physverify.synth.plan import DatasetPlan
from physverify.synth.settings import DatasetSettings
from physverify.timing.settings import GenerationSettings
from physverify.verification.exclusion_record import ExclusionRecord

PINNED = Path(__file__).with_name("data") / "persisted_fields.json"

MODELS: Mapping[str, type[BaseModel]] = {
    "calibration_artifact": CalibrationArtifact,
    "evaluation_report": EvaluationReport,
    "exclusion_record": ExclusionRecord,
    "dev_score_report": DevScoreReport,
    "ledger_entry": LedgerEntry,
    "dataset_settings": DatasetSettings,
    "generation_settings": GenerationSettings,
}
PARAMS_KINDS = ("checks", "providers")

DATASET_PLAN = {
    "version": "v1",
    "config_hash": "c" * 64,
    "gt_revision": 2,
    "scenes": [
        {"scene_id": "f1_projectile_s000", "family": "f1_projectile", "index": 0, "split": "calib"}
    ],
    "variants": [
        {
            "clip_id": "f1_projectile_s000_none_0",
            "scene_id": "f1_projectile_s000",
            "sim_id": "f1_projectile_s000",
            "family": "f1_projectile",
            "index": 0,
            "perturbation": "none",
            "level": 0,
            "target_magnitude": 0.0,
            "split": "calib",
        }
    ],
    "subsets": {"smoke": ["f1_projectile_s000_none_0"]},
}


def field_paths(
    node: Mapping[str, Any], defs: Mapping[str, Any], prefix: str = "", seen: tuple[str, ...] = ()
) -> list[str]:
    """Dotted field paths of a JSON schema, depth first in declaration order."""
    ref = node.get("$ref")
    if ref is not None:
        name = ref.rsplit("/", 1)[-1]
        if name in seen:
            return []
        node, seen = defs[name], (*seen, name)
    found: list[str] = []
    for key, sub in node.get("properties", {}).items():
        found += [prefix + key, *field_paths(sub, defs, f"{prefix}{key}.", seen)]
    for group in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for sub in node.get(group, []):
            found += field_paths(sub, defs, prefix, seen)
    for key in ("items", "additionalProperties"):
        if isinstance(node.get(key), Mapping):
            found += field_paths(node[key], defs, prefix, seen)
    return list(dict.fromkeys(found))


def model_paths(model: type[BaseModel]) -> list[str]:
    schema = model.model_json_schema(mode="serialization")
    return field_paths(schema, schema.get("$defs", {}))


@pytest.fixture(scope="module")
def pinned() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(PINNED.read_text(encoding="utf-8"))
    return data


@pytest.mark.parametrize("name", list(MODELS))
def test_persisted_model_fields_are_unchanged(pinned: dict[str, Any], name: str) -> None:
    assert model_paths(MODELS[name]) == pinned["models"][name]


@pytest.mark.parametrize("kind", PARAMS_KINDS)
def test_plugin_params_fields_are_unchanged(pinned: dict[str, Any], kind: str) -> None:
    registry = PLUGIN_KINDS[kind]()
    found = {key: model_paths(plugin.Params) for key, plugin in sorted(registry.items())}
    assert found == pinned[kind]


def test_the_dataset_plan_json_is_unchanged() -> None:
    written = DatasetPlan.from_json(json.loads(json.dumps(DATASET_PLAN))).to_json()
    assert written == DATASET_PLAN
    assert list(written) == list(DATASET_PLAN)
    assert list(written["variants"][0]) == list(DATASET_PLAN["variants"][0])
