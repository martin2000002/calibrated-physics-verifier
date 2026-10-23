"""The files the results report reads, loaded once and checked to belong to one artifact.

Every number is later read from a ``Doc`` by JSON pointer (RFC 6901), so the value printed and
the ``path#pointer`` it cites cannot drift apart. The typed loaders (``parse_calibration``,
``EvaluationReport``, ``parse_record``) only validate and give the sha256 to compare.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from physverify.calibration.artifact import EvaluationReport
from physverify.calibration.artifact_io import (
    Calibration,
    CalibrationError,
    parse_calibration,
)
from physverify.core import jsonio
from physverify.core.digests import text_sha256
from physverify.human_agreement.clip_decision import CALIBRATED_NAME
from physverify.human_agreement.metrics import METRICS_FILE
from physverify.human_agreement.report import REPORT_DIR
from physverify.reporting.chain import (
    EVALUATION_KEY,
    RECORD_KEY,
    ChainState,
    Missing,
    ResultsInputError,
    abstain_curve_run,
)
from physverify.schema.checks import InputLevel
from physverify.selective_risk.settings import ABSTENTION_FILE, REPORT_FILE
from physverify.verification.exclusion_record import ExclusionInputError, parse_record
from physverify.verification.scope import CheckScope, record_path

FIT_STEPS = (5, 6, 7)
DECIDE_STEP = 9
CONSUMED_LEVEL = InputLevel.PERCEIVED
POINTER_SEPARATOR = "#"


@dataclass(frozen=True)
class Doc:
    path: Path
    data: Any
    sha256: str

    def at(self, pointer: str) -> Any:
        return walk(self.data, pointer)

    def source(self, pointer: str) -> str:
        return f"{self.path}{POINTER_SEPARATOR}{pointer}"


def pointer(*parts: str | int) -> str:
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def walk(data: Any, path: str) -> Any:
    node = data
    for raw in path.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node


def resolve(source: str) -> Any:
    """What a ``path#pointer`` source names, read again from disk."""
    path, _, where = source.partition(POINTER_SEPARATOR)
    return walk(jsonio.read_json(Path(path), ResultsInputError), where)


def read_doc(path: Path) -> Doc:
    if not path.is_file():
        raise ResultsInputError(f"no file at {path}")
    text = path.read_text(encoding="utf-8")
    data = jsonio.loads(text, path, ResultsInputError)
    return Doc(path=path, data=data, sha256=text_sha256(text))


@dataclass(frozen=True)
class Request:
    """``check_scope``: ``[check_scope]``, whose record is resolved against the artifact's
    directory by ``verification.scope.record_path``."""

    chain_dir: Path
    calibration_evaluation: Path | None = None
    agreement_run: Path | None = None
    abstention_run: Path | None = None
    check_scope: CheckScope = CheckScope()


@dataclass(frozen=True)
class Inputs:
    chain: ChainState
    profile: str
    artifact: Doc
    calibration: Calibration
    evaluation: Doc
    heldout_evaluation: Doc
    abstention: Doc | Missing
    abstention_report: Doc | Missing
    record: Doc | Missing
    metrics: Doc | None


def collect_inputs(request: Request) -> Inputs:
    chain = ChainState.load(request.chain_dir)
    if request.agreement_run is not None and chain.rehearsal:
        raise ResultsInputError(
            f"--agreement-run with the {chain.device} chain: "
            "the human-agreement study runs only on the CPU calibration"
        )
    chain.require_steps(FIT_STEPS)
    evaluation, report = _evaluation(Path(chain.required(EVALUATION_KEY)))
    artifact, calibration = _artifact(report)
    heldout_evaluation, calibration_report = (
        (evaluation, report)
        if request.calibration_evaluation is None
        else _evaluation(request.calibration_evaluation)
    )
    _require_same_artifact(calibration_report, calibration, heldout_evaluation.path)
    require_artifact_tau(calibration_report, calibration, heldout_evaluation.path)
    calibration_dir = artifact.path.parent
    curve = request.abstention_run or abstain_curve_run(chain)
    profile = evaluation_profile(report)
    record = record_path(request.check_scope, calibration_dir)
    return Inputs(
        chain=chain,
        profile=profile,
        artifact=artifact,
        calibration=calibration,
        evaluation=evaluation,
        heldout_evaluation=heldout_evaluation,
        abstention=_abstention(calibration_dir / ABSTENTION_FILE, calibration),
        abstention_report=(
            curve if isinstance(curve, Missing) else _abstention(curve / REPORT_FILE, calibration)
        ),
        record=_record(chain, record, calibration, evaluation),
        metrics=(
            None
            if request.agreement_run is None
            else _metrics(request.agreement_run, calibration, profile)
        ),
    )


def _evaluation(path: Path) -> tuple[Doc, EvaluationReport]:
    doc = read_doc(path)
    try:
        return doc, EvaluationReport.model_validate(doc.data)
    except ValidationError as exc:
        raise ResultsInputError(f"{path} is not an evaluation report:\n{exc}") from exc


def _artifact(report: EvaluationReport) -> tuple[Doc, Calibration]:
    path = Path(report.artifact_path)
    doc = read_doc(path)
    try:
        calibration = parse_calibration(path.read_text(encoding="utf-8"), path=path)
    except CalibrationError as exc:
        raise ResultsInputError(str(exc)) from exc
    _require_same_artifact(report, calibration, path)
    return doc, calibration


def _require_same_artifact(report: EvaluationReport, calibration: Calibration, label: Path) -> None:
    if report.artifact_sha256 != calibration.sha256:
        raise ResultsInputError(
            f"{label}: evaluated artifact sha256 {report.artifact_sha256[:12]}, but "
            f"{calibration.path} has {calibration.sha256[:12]} (refitted since?)"
        )
    if report.artifact_run_id != calibration.run_id:
        raise ResultsInputError(
            f"{label}: evaluated artifact run {report.artifact_run_id}, not {calibration.run_id}"
        )


def require_artifact_tau(report: EvaluationReport, calibration: Calibration, label: Path) -> None:
    """The all-scenes pooled tau_b of the evaluation must be the artifact's (same rows, same
    bootstrap); a check the evaluation leaves out must have no pair in the artifact either."""
    for stratum in report.strata:
        if not stratum.monotonicity:
            continue
        fitted = calibration.stratum(stratum.level, stratum.profile).model.checks
        pooled = {m.check_id: m.monotonicity.to_json() for m in stratum.monotonicity}
        for check_id, check in fitted.items():
            expected = check.monotonicity.to_json() if check.monotonicity.n_pairs else None
            if pooled.get(check_id) != expected:
                raise ResultsInputError(
                    f"{label}: the pooled tau_b of {stratum.level.value}/{stratum.profile} "
                    f"{check_id} differs from the artifact's (the calib rows changed?)"
                )


def evaluation_profile(report: EvaluationReport) -> str:
    profiles = {s.profile for s in report.strata}
    if len(profiles) != 1:
        raise ResultsInputError(f"the evaluation has strata of profiles {sorted(profiles)}")
    return profiles.pop()


def _abstention(path: Path, calibration: Calibration) -> Doc | Missing:
    """A table fitted by abstain on this artifact: the perceived table's ref and source."""
    if not path.is_file():
        return Missing(f"no {path}")
    doc = read_doc(path)
    table = doc.data.get("thresholds", {}).get(CONSUMED_LEVEL.value, {})
    ref = table.get("artifact") or {}
    source = str(table.get("source", ""))
    if ref.get("sha256") != calibration.sha256 or not source.endswith(f"@{calibration.run_id}"):
        raise ResultsInputError(
            f"{path} used the table {source!r} (artifact sha256 {str(ref.get('sha256'))[:12]}), "
            f"not the artifact {calibration.run_id} ({calibration.sha256[:12]})"
        )
    return doc


def _record(
    chain: ChainState,
    path: Path | None,
    calibration: Calibration,
    evaluation: Doc,
) -> Doc | Missing:
    if not chain.completed(DECIDE_STEP):
        return Missing(f"step {DECIDE_STEP} (exclusions decide) has not completed")
    if path is None:
        return Missing("[check_scope].decision_record is unset")
    if not path.is_file():
        return Missing(f"no decision record at {path}")
    doc = read_doc(path)
    try:
        record = parse_record(path.read_text(encoding="utf-8"), str(path))
    except ExclusionInputError as exc:
        raise ResultsInputError(str(exc)) from exc
    expected = {
        "the chain's RECORD_SHA": (record.sha256, chain.required(RECORD_KEY)),
        "the artifact sha256": (record.inputs.artifact_sha256, calibration.sha256),
        "the evaluation sha256": (record.inputs.evaluation_sha256, evaluation.sha256),
    }
    for what, (found, wanted) in expected.items():
        if found != wanted:
            raise ResultsInputError(f"{path} does not match {what} ({found[:12]} != {wanted[:12]})")
    return doc


def _metrics(run: Path, calibration: Calibration, profile: str) -> Doc:
    """A calibrated rule must be the stratum this report covers: ``CalibratedRule.label`` ends
    with the stratum source ``#<level>/<profile>@<run_id>`` (``loader.Calibration._wrap``)."""
    doc = read_doc(run / REPORT_DIR / METRICS_FILE)
    rule = str(doc.data.get("rule", ""))
    expected = f"#{CONSUMED_LEVEL.value}/{profile}@{calibration.run_id}"
    if is_calibrated(rule) and not rule.endswith(expected):
        raise ResultsInputError(
            f"{doc.path}: rule {rule!r} is not the {CONSUMED_LEVEL.value}/{profile} stratum of "
            f"the artifact {calibration.run_id}"
        )
    return doc


def is_calibrated(rule: str) -> bool:
    return rule.startswith(f"{CALIBRATED_NAME}:")


def stratum_index(
    strata: Sequence[Mapping[str, Any]], level: InputLevel, profile: str
) -> int | None:
    return _index(strata, lambda s: s.get("level") == level.value and s.get("profile") == profile)


def check_index(checks: Sequence[Mapping[str, Any]], check_id: str) -> int | None:
    return _index(checks, lambda c: c.get("check_id") == check_id)


def _index(
    items: Sequence[Mapping[str, Any]], match: Callable[[Mapping[str, Any]], bool]
) -> int | None:
    return next((i for i, item in enumerate(items) if match(item)), None)
