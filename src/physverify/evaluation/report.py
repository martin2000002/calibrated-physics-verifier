"""``h3_report.json`` and ``h3_report.md``: every pre-registered number, the verdict and the audit
(ADR 0013). Rendering only; the numbers come from ``analysis`` and ``verdict``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from pydantic import BaseModel, ConfigDict

from physverify.agreement.prereg import FreezeAudit
from physverify.core.atomic import write_text_atomic
from physverify.evaluation.analysis import ArmSummary, GuardResult, H3Analysis
from physverify.evaluation.layout import H3Layout
from physverify.evaluation.margins import MarginDerivation
from physverify.evaluation.prereg import Mde
from physverify.evaluation.stats import TestResult
from physverify.evaluation.verdict import H3Verdict

ESTIMAND: Final = (
    "Estimand: the per-prompt Hodges-Lehmann shift of S(fine-tuned) - S(best-of-8), on the "
    "checks decided in both videos, over the analysed eval prompts; one fine-tuned sample is "
    "set against a selection that costs eight generations and eight verifications."
)
CIRCULARITY: Final = (
    "Circularity: the verifier is the reward of the fine-tuning, the selector of best-of-8 and "
    "the primary judge. A verifier-only result says the model learned what the verifier rewards; "
    "only agreement with the external judge (VideoPhy-2 AutoEval, a weak judge: Pearson PC 0.37 "
    "with humans) supports a physical improvement. The winner's curse favours best-of-8; reward "
    "hacking favours the fine-tuned model."
)
NONE: Final = "-"


class H3Report(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    campaign: str
    rule_label: str | None
    rule_status: str
    ft_model_tag: str | None
    scipy_version: str
    method: str
    analysis: H3Analysis
    verdict: H3Verdict
    margins: dict[str, MarginDerivation]
    mde: Mde
    audit: FreezeAudit
    estimand: str = ESTIMAND
    circularity: str = CIRCULARITY


def _num(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return NONE
    return str(value) if isinstance(value, int) else f"{value:.{digits}g}"


def _table(header: list[str], rows: Iterable[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return [*lines, *("| " + " | ".join(row) + " |" for row in rows)]


TEST_HEADER: Final = ["test", "n", "dropped", "p", "HL", "mean", "CI of mean"]


def _test_row(name: str, t: TestResult | None) -> list[str]:
    if t is None:
        return [name, NONE, NONE, "not run", NONE, NONE, NONE]
    ci = f"[{_num(t.ci_lo)}, {_num(t.ci_hi)}]"
    return [name, str(t.n), str(t.dropped), _num(t.p), _num(t.hl), _num(t.mean), ci]


def _arm_row(name: str, a: ArmSummary) -> list[str]:
    values = (a.n, a.n_scored, a.abstention, a.mean_score, a.median_motion, a.objects_ok_rate)
    return [name, *(_num(v) for v in values), _num(a.mean_pc), _num(a.mean_sa)]


def _guard_row(name: str, g: GuardResult) -> list[str]:
    return [name, g.state, _num(g.value), _num(g.limit), g.detail]


def _margin_row(task: str, d: MarginDerivation) -> list[str]:
    return [
        task,
        d.column,
        f"{d.lo_generator} {_num(d.lo)}",
        f"{d.hi_generator} {_num(d.hi)}",
        _num(d.spread),
        _num(d.se),
        _num(d.value),
    ]


def _verdict(r: H3Report) -> list[str]:
    v = r.verdict
    post_hoc = " POST HOC" if v.post_hoc else ""
    return [
        f"# H3 comparison: {r.campaign}",
        "",
        f"**Verdict: `{v.verdict}`{post_hoc}** ({'; '.join(v.reasons)})",
        "",
        f"- rule: `{r.rule_label}` ({r.rule_status}); fine-tuned model: `{r.ft_model_tag}`",
        f"- external status: `{v.external_status}`; external outcome: `{v.external_outcome}`",
        f"- method: {r.method}; scipy {r.scipy_version}",
        "",
        r.estimand,
        "",
    ]


def _arms(a: H3Analysis) -> list[str]:
    header = ["arm", "n", "scored", "abstention", "mean S", "median motion", "objects_ok"]
    rows = [_arm_row(name, arm) for name, arm in a.arms.items()]
    fallbacks = ", ".join(f"{k}: {n}" for k, n in sorted(a.fallbacks.items())) or NONE
    return [
        "## Arms",
        "",
        *_table([*header, "mean PC", "mean SA"], rows),
        "",
        f"Best-of-8 picks by fallback: {fallbacks}",
        "",
    ]


def _tests(a: H3Analysis) -> list[str]:
    holm_rows = [[*_test_row(name, e.test), _num(e.p_adjusted)] for name, e in a.holm.items()]
    return [
        "## Primary and external (fixed sequence)",
        "",
        *_table(
            TEST_HEADER,
            [
                _test_row("primary: S(ft) - S(bo8)", a.primary),
                _test_row("external: PC(ft) - PC(bo8)", a.external),
            ],
        ),
        "",
        "## Secondary tests (Holm, m = 3)",
        "",
        *_table([*TEST_HEADER, "Holm p"], holm_rows),
        "",
        f"Ranking claim (PC of best-of-8 above a random base pick): {a.ranking_claim}",
        "",
        "Descriptive only (order-statistic tautology):",
        "",
        *_table(
            TEST_HEADER, [_test_row("verifier: S(bo8) - mean S(base)", a.bo8_vs_base_verifier)]
        ),
        "",
    ]


def _guards(a: H3Analysis) -> list[str]:
    rows = [_guard_row(name, g) for name, g in a.guards.items()]
    return [
        "## Guards",
        "",
        *_table(["guard", "state", "value", "limit", "detail"], rows),
        "",
        f"G5 prompts left out (best-of-8 without a score): {a.g5_left_out}",
        "",
    ]


def _sensitivities(a: H3Analysis) -> list[str]:
    return [
        "## Sensitivities and strata",
        "",
        *_table(TEST_HEADER, [_test_row(n, t) for n, t in a.sensitivities.items()]),
        "",
        *_table(TEST_HEADER, [_test_row(n, t) for n, t in a.strata.items()]),
        "",
    ]


def _power(r: H3Report) -> list[str]:
    m = r.mde
    flag = " (underpowered: above twice the margin)" if m.pc_underpowered else ""
    return [
        "## Minimum detectable effect and margins",
        "",
        f"- n = {m.n}, alpha {m.alpha}, power {m.power}; sd estimated on dev",
        f"- verifier: sd {_num(m.verifier_sd)}, MDE {_num(m.verifier_mde)} (estimated)",
        f"- PC: sd {_num(m.pc_sd)}, MDE {_num(m.pc_mde)} (estimated){flag}",
        "",
        *_table(
            ["task", "column", "lowest", "highest", "spread", "SE", "margin"],
            [_margin_row(task, d) for task, d in r.margins.items()],
        ),
        "",
    ]


def _audit(r: H3Report) -> list[str]:
    a = r.audit
    deviations = [f"- {d}" for d in a.deviations] or ["- none"]
    return [
        "## Freeze audit",
        "",
        f"- pre-registration {a.prereg_sha256[:12]} created {a.created_utc}, replaces "
        f"{(a.replaces_sha256 or NONE)[:12]}; {a.versions} version(s), first frozen "
        f"{a.first_created_utc}; post hoc: {a.post_hoc}",
        f"- earliest eval file {a.earliest_eval_utc or NONE}; first freeze before it: "
        f"{a.blind}; current freeze after it: {a.frozen_after_eval} "
        "(file times are informational; the binding is by content hash)",
        f"- code commit at the freeze {a.frozen_commit[:12]}, "
        f"at this report {a.report_commit[:12]}",
        "",
        "Deviations:",
        "",
        *deviations,
        "",
        "## Circularity",
        "",
        r.circularity,
        "",
    ]


def render_markdown(report: H3Report) -> str:
    sections = (
        _verdict(report),
        _arms(report.analysis),
        _tests(report.analysis),
        _guards(report.analysis),
        _sensitivities(report.analysis),
        _power(report),
        _audit(report),
    )
    return "\n".join(line for section in sections for line in section)


def write_report(layout: H3Layout, report: H3Report) -> None:
    write_text_atomic(layout.report_json, report.model_dump_json(indent=2) + "\n")
    write_text_atomic(layout.report_md, render_markdown(report))
