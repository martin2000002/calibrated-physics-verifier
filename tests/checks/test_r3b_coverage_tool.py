"""`tools/r3b_coverage.py` counts clips with an in-scope event over clips with an event."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.io import write_jsonl

TOOL = Path(__file__).resolve().parents[2] / "tools" / "r3b_coverage.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("r3b_coverage_tool", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result(
    check_id: str,
    clip_id: str,
    subject: str = "a",
    level: InputLevel = InputLevel.PERCEIVED,
    reason: AbstainReason | None = None,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        clip_id=clip_id,
        subject=(subject,),
        frame_span=(0, 3),
        magnitude=0.1 if reason is None else None,
        sigma=0.0,
        unit="u",
        verdict=Verdict.UNSCORED if reason is None else Verdict.ABSTAIN,
        input_level=level,
        abstain_reason=reason,
    )


def outside(clip_id: str, subject: str) -> CheckResult:
    found = result("R3b", clip_id, subject, reason=AbstainReason.NOT_APPLICABLE)
    details = {
        "not_applicable": "two_sided_spheres_isolated",
        "not_applicable_predicates": ["spheres", "isolated"] if subject == "a" else ["two_sided"],
    }
    return replace(found, details=details)


RESULTS = [
    outside("c0", "a"),
    result("R3b", "c0", "b"),
    outside("c1", "a"),
    outside("c1", "b"),
    result("R3b", "c2", reason=AbstainReason.TOO_FEW_FRAMES),
    result("R3b", "c3", level=InputLevel.ORACLE),
    result("R1", "c4"),
]


def test_a_clip_is_in_scope_when_one_of_its_events_is() -> None:
    table = _load_tool().coverage(RESULTS, {"c0": "positive", "c1": "positive", "c2": "clean"})
    assert table["perceived"]["positive"] == {
        "clips_with_event": 2,
        "clips_in_scope": 1,
        "coverage": 0.5,
        "failed": {"isolated": 2, "spheres": 2, "two_sided": 1},
    }
    assert table["perceived"]["clean"]["clips_in_scope"] == 1
    assert table["oracle"] == {
        "all": {"clips_with_event": 1, "clips_in_scope": 1, "coverage": 1.0, "failed": {}}
    }


def test_the_command_reads_checks_jsonl_and_roles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checks, roles = tmp_path / "checks.jsonl", tmp_path / "roles.csv"
    write_jsonl(checks, RESULTS)
    roles.write_text("clip_id,role\nc0,positive\nc1,positive\n", encoding="utf-8")
    _load_tool().main([str(checks), "--roles", str(roles)])
    printed = json.loads(capsys.readouterr().out)
    assert printed["perceived"]["positive"]["clips_in_scope"] == 1
    assert printed["perceived"]["all"]["clips_with_event"] == 1
