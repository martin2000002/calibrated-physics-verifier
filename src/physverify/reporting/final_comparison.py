"""The comparison table of the final run: B1 against the external judge on the fresh Kubric
set (Part 1), B1 at u_B1 on real and Kubric video (Part 2), and the verifier against the VLM on
VideoPhy-2 test (Part 3). Every number is computed here from the rows ``baselines compare`` and
``agreement compare`` wrote."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from physverify.baselines.records import ClipRow, PartRows
from physverify.baselines.rules import NOT_EVALUABLE, Verdict, p1, part2, part3
from physverify.baselines.settings import BootstrapSettings, Part, VlmComparisonSettings
from physverify.baselines.stats import (
    Estimate,
    Flag,
    Score,
    b1_flag,
    b1_score,
    false_alarms,
    judge_flag,
    judge_score,
    paired_pairs,
    scene_detection,
    scene_pairs,
    unread_share,
)
from physverify.core.digests import file_sha256
from physverify.core.jsonio import read_model
from physverify.core.markdown import table
from physverify.reporting import final_agreement as agreement
from physverify.reporting.final_rows import Row, ci_text, fmt
from physverify.stats.intervals import Rate

Outcome = Verdict | Literal["descriptive"]
DESCRIPTIVE: Final[Outcome] = "descriptive"
B1_NOTE: Final = (
    "B1 is the verifier's clip value with the scope's exclusions, not the calibrated score; "
    "u_B1 is the clean quantile of B1 on the calibration set itself (in-sample)"
)
R2_PREFIX: Final = "R2.fpr"


@dataclass(frozen=True)
class ComparisonInputs:
    kubric: Path | None = None
    real: Path | None = None
    agreement_items: Path | None = None
    final_table: Path | None = None


class _FinalTable(BaseModel):
    """``final_table.json``, of which only its rows are read."""

    model_config = ConfigDict(frozen=True)
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Line:
    id: str
    claim: str
    value: str
    ci: str
    rule: str
    verdict: Outcome
    source: str
    note: str = ""


@dataclass(frozen=True)
class PairAccuracies:
    """Pair accuracy of B1, of the judge and their difference (``b1``, ``judge``, ``delta``) over
    a part's clean-perturbed pairs, and the share of its clips the judge left unread."""

    estimates: Mapping[str, Estimate]
    unread: float


def pair_accuracies(part: PartRows, boot: BootstrapSettings) -> PairAccuracies:
    """The one computation behind Part 1 of the table and the pair accuracy figure."""
    rows = clip_rows(part)
    return PairAccuracies(paired_pairs(scene_pairs(rows), boot), unread_share(rows))


def clip_rows(part: PartRows) -> list[ClipRow]:
    return [row for found in part.sets for row in found.rows]


def comparison_lines(inputs: ComparisonInputs, config: VlmComparisonSettings) -> list[Line]:
    kubric = _part(inputs.kubric, "kubric")
    real = _part(inputs.real, "real")
    return [
        *p1_lines(kubric, config, _src(inputs.kubric)),
        *part2_lines(kubric, real, config, _src(inputs.kubric), _src(inputs.real)),
        *judge_lines(kubric, real, config, _src(inputs.kubric), _src(inputs.real)),
        *r2_lines(inputs.final_table),
        *part3_lines(inputs.agreement_items, config),
    ]


def p1_lines(kubric: PartRows | None, config: VlmComparisonSettings, src: str) -> list[Line]:
    rule = "PA(B1) - PA(judge): CI low > 0 confirms"
    if kubric is None:
        return [_missing("part1.delta", "pair accuracy, B1 minus judge (fresh Kubric)", rule)]
    pairs = pair_accuracies(kubric, config.pairs)
    found, unread = pairs.estimates, pairs.unread
    verdict = p1(found.get("delta"), unread, config.max_uninterpretable)
    note = f"judge unread {fmt(unread)} (limit {config.max_uninterpretable}); {_looks(kubric)}"
    return [
        _estimate(
            "part1.pa_b1", "pair accuracy of B1 (fresh Kubric)", found["b1"], "", DESCRIPTIVE, src
        ),
        _estimate(
            "part1.pa_judge",
            f"pair accuracy of {kubric.judge} (fresh Kubric)",
            found["judge"],
            "",
            DESCRIPTIVE,
            src,
        ),
        _estimate(
            "part1.delta",
            "pair accuracy, B1 minus judge (fresh Kubric)",
            found["delta"],
            rule,
            verdict,
            src,
            note,
        ),
    ]


def part2_lines(
    kubric: PartRows | None,
    real: PartRows | None,
    config: VlmComparisonSettings,
    kubric_src: str,
    real_src: str,
) -> list[Line]:
    rule = (
        f"every real set FPR <= {config.real_fpr} and CP upper <= {config.real_fpr_upper}; "
        "fresh Kubric detection CI low > 0"
    )
    cut = _cut(kubric, real)
    if cut is None:
        return [_missing("part2", "B1 at u_B1: quiet on real video, detects on Kubric", rule)]
    flag = b1_flag(cut)
    fprs = {} if real is None else _fprs(real, flag, b1_score, config)
    detected = (
        None if kubric is None else scene_detection(clip_rows(kubric), flag, b1_score, config.pairs)
    )
    lines = [
        _rate(f"part2.fpr.{name}", f"FPR of B1 > u_B1 on {name}", r, real_src)
        for name, r in fprs.items()
    ]
    if detected is not None:
        lines.append(
            _estimate(
                "part2.detection",
                "TPR - FPR of B1 > u_B1 (fresh Kubric)",
                detected,
                "",
                DESCRIPTIVE,
                kubric_src,
            )
        )
    verdict = part2(fprs, detected, config.real_fpr, config.real_fpr_upper)
    lines.append(
        Line(
            "part2",
            "B1 at u_B1: quiet on real video, detects on Kubric",
            f"u_B1 {fmt(cut)}",
            "",
            rule,
            verdict,
            real_src,
            B1_NOTE,
        )
    )
    return lines


def judge_lines(
    kubric: PartRows | None,
    real: PartRows | None,
    config: VlmComparisonSettings,
    kubric_src: str,
    real_src: str,
) -> list[Line]:
    """The judge read the published way (score below its cut), described, judged by no rule."""
    lines: list[Line] = []
    if real is not None:
        flag = judge_flag(real.cut)
        for name, rate in _fprs(real, flag, judge_score, config).items():
            claim = f"FPR of {real.judge} < {fmt(real.cut)} on {name}"
            lines.append(_rate(f"J.fpr.{name}", claim, rate, real_src))
    if kubric is not None:
        found = scene_detection(
            clip_rows(kubric), judge_flag(kubric.cut), judge_score, config.pairs
        )
        claim = f"TPR - FPR of {kubric.judge} < {fmt(kubric.cut)} (fresh Kubric)"
        lines.append(_estimate("J.detection", claim, found, "", DESCRIPTIVE, kubric_src))
    return lines


def r2_lines(final_table: Path | None) -> list[Line]:
    """The final table's per-check real FPR rows, printed beside Part 2."""
    if final_table is None or not final_table.is_file():
        return [_missing(R2_PREFIX, "final table: real video FPR per deciding check", "")]
    rows = read_model(_FinalTable, final_table).rows
    return [
        Line(
            row.id,
            f"final table: {row.criterion}",
            row.value,
            row.ci,
            row.threshold,
            DESCRIPTIVE,
            str(final_table),
            f"final table status {row.status}; {row.note}",
        )
        for row in rows
        if row.id.startswith(R2_PREFIX)
    ]


def part3_lines(items_path: Path | None, config: VlmComparisonSettings) -> list[Line]:
    rule = "BA(verifier) - BA(VLM): CI high < 0 confirms, CI low > 0 refutes, else no difference"
    claim = "balanced accuracy, verifier minus VLM (VideoPhy-2 test)"
    if items_path is None or not items_path.is_file():
        return [_missing("part3.delta_ba", claim, rule)]
    items = agreement.partition(agreement.read_items(items_path), config.agreement_partition)
    boot = config.agreement
    found = agreement.paired(
        items,
        agreement.balanced_accuracy_difference,
        agreement.Bootstrap(boot.n, boot.seed, boot.ci),
    )
    share = agreement.decided_fraction(items, boot.ci)
    src = str(items_path)
    return [
        Line(
            "part3.delta_ba",
            claim,
            fmt(found.value),
            ci_text(found.low, found.high),
            rule,
            part3(found.low, found.high),
            src,
            f"n {found.n}; decided fraction {fmt(share.value)} {ci_text(share.low, share.high)}",
        )
    ]


def markdown(lines: Sequence[Line]) -> str:
    out = ["# Comparison table", ""]
    out += table(
        ("id", "claim", "value", "CI", "rule", "verdict", "source", "note"),
        ([x.id, x.claim, x.value, x.ci, x.rule, x.verdict, x.source, x.note] for x in lines),
    )
    return "\n".join(out) + "\n"


def as_json(lines: Sequence[Line], inputs: ComparisonInputs) -> str:
    paths = {k: v for k, v in asdict(inputs).items() if v is not None}
    shas = {k: file_sha256(Path(v)) for k, v in paths.items() if Path(v).is_file()}
    data = {"lines": [asdict(x) for x in lines], "inputs": {k: str(v) for k, v in paths.items()}}
    return json.dumps({**data, "input_sha256": shas}, indent=2, sort_keys=True) + "\n"


def _part(path: Path | None, part: Part) -> PartRows | None:
    if path is None or not path.is_file():
        return None
    found = read_model(PartRows, path)
    return found if found.part == part else None


def _cut(kubric: PartRows | None, real: PartRows | None) -> float | None:
    """u_B1 of the parts read; None when none is read or two parts disagree on it."""
    cuts = {p.cuts.u_b1 for p in (kubric, real) if p is not None}
    return cuts.pop() if len(cuts) == 1 else None


def _fprs(
    part: PartRows, flag: Flag, score: Score, config: VlmComparisonSettings
) -> dict[str, Rate]:
    return {s.dataset: false_alarms(s.rows, flag, score, config.pairs.ci) for s in part.sets}


def _looks(part: PartRows) -> str:
    freeze = "development" if part.freeze is None else part.freeze[:12]
    return f"freeze {freeze}, looks {part.looks}"


def _src(path: Path | None) -> str:
    return "" if path is None else str(path)


def _missing(id_: str, claim: str, rule: str) -> Line:
    return Line(id_, claim, "n/a", "", rule, NOT_EVALUABLE, "", "missing input")


def _estimate(
    id_: str, claim: str, found: Estimate, rule: str, verdict: Outcome, src: str, note: str = ""
) -> Line:
    text = f"n {found.n}" + (f"; {note}" if note else "")
    return Line(
        id_, claim, fmt(found.value), ci_text(found.ci.low, found.ci.high), rule, verdict, src, text
    )


def _rate(id_: str, claim: str, found: Rate, src: str) -> Line:
    value = f"{fmt(found.value)} ({found.k}/{found.n})"
    return Line(
        id_,
        claim,
        value,
        ci_text(found.ci.low, found.ci.high),
        "",
        DESCRIPTIVE,
        src,
        "Clopper-Pearson",
    )
