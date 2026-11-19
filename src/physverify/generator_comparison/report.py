"""``h3_report.json`` and ``h3_report.md``: every pre-registered number, the verdict and the audit.
Rendering only; the numbers come from ``analysis`` and ``verdict``."""

from __future__ import annotations

from typing import Final

from physverify.core.atomic import write_text_atomic
from physverify.core.markdown import SIG4, table
from physverify.core.models import FrozenModel
from physverify.generator_comparison.analysis_inputs import ArmSummary
from physverify.generator_comparison.guards import GuardResult
from physverify.generator_comparison.layout import ComparisonLayout
from physverify.generator_comparison.margins import MarginDerivation
from physverify.generator_comparison.prereg import Mde
from physverify.generator_comparison.preregistered_numbers import ComparisonAnalysis
from physverify.generator_comparison.stats import TestResult
from physverify.generator_comparison.verdict import ComparisonVerdict
from physverify.human_agreement.prereg import FreezeAudit

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


class ComparisonReport(FrozenModel):
    campaign: str
    rule_label: str | None
    rule_status: str
    ft_model_tag: str | None
    scipy_version: str
    method: str
    analysis: ComparisonAnalysis
    verdict: ComparisonVerdict
    margins: dict[str, MarginDerivation]
    mde: Mde
    audit: FreezeAudit
    estimand: str = ESTIMAND
    circularity: str = CIRCULARITY


def _num(value: float | int | None) -> str:
    return str(value) if isinstance(value, int) else SIG4(value)


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


def _verdict(r: ComparisonReport) -> list[str]:
    v = r.verdict
    post_hoc = " POST HOC" if v.post_hoc else ""
    return [
        f"# Generator comparison: {r.campaign}",
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


def _arms(a: ComparisonAnalysis) -> list[str]:
    header = ["arm", "n", "scored", "abstention", "mean S", "median motion", "objects_ok"]
    rows = [_arm_row(name, arm) for name, arm in a.arms.items()]
    fallbacks = ", ".join(f"{k}: {n}" for k, n in sorted(a.fallbacks.items())) or NONE
    return [
        "## Arms",
        "",
        *table([*header, "mean PC", "mean SA"], rows),
        "",
        f"Best-of-8 picks by fallback: {fallbacks}",
        "",
    ]


def _tests(a: ComparisonAnalysis) -> list[str]:
    holm_rows = [[*_test_row(name, e.test), _num(e.p_adjusted)] for name, e in a.holm.items()]
    return [
        "## Primary and external (fixed sequence)",
        "",
        *table(
            TEST_HEADER,
            [
                _test_row("primary: S(ft) - S(bo8)", a.primary),
                _test_row("external: PC(ft) - PC(bo8)", a.external),
            ],
        ),
        "",
        "## Secondary tests (Holm, m = 3)",
        "",
        *table([*TEST_HEADER, "Holm p"], holm_rows),
        "",
        f"Ranking claim (PC of best-of-8 above a random base pick): {a.ranking_claim}",
        "",
        "Descriptive only (order-statistic tautology):",
        "",
        *table(TEST_HEADER, [_test_row("verifier: S(bo8) - mean S(base)", a.bo8_vs_base_verifier)]),
        "",
    ]


def _guards(a: ComparisonAnalysis) -> list[str]:
    rows = [_guard_row(name, g) for name, g in a.guards.items()]
    return [
        "## Guards",
        "",
        *table(["guard", "state", "value", "limit", "detail"], rows),
        "",
        f"G5 prompts left out (best-of-8 without a score): {a.g5_left_out}",
        "",
    ]


def _sensitivities(a: ComparisonAnalysis) -> list[str]:
    return [
        "## Sensitivities and strata",
        "",
        *table(TEST_HEADER, [_test_row(n, t) for n, t in a.sensitivities.items()]),
        "",
        *table(TEST_HEADER, [_test_row(n, t) for n, t in a.strata.items()]),
        "",
    ]


def _power(r: ComparisonReport) -> list[str]:
    m = r.mde
    flag = " (underpowered: above twice the margin)" if m.pc_underpowered else ""
    return [
        "## Minimum detectable effect and margins",
        "",
        f"- n = {m.n}, alpha {m.alpha}, power {m.power}; sd estimated on dev",
        f"- verifier: sd {_num(m.verifier_sd)}, MDE {_num(m.verifier_mde)} (estimated)",
        f"- PC: sd {_num(m.pc_sd)}, MDE {_num(m.pc_mde)} (estimated){flag}",
        "",
        *table(
            ["task", "column", "lowest", "highest", "spread", "SE", "margin"],
            [_margin_row(task, d) for task, d in r.margins.items()],
        ),
        "",
    ]


def _audit(r: ComparisonReport) -> list[str]:
    a = r.audit
    deviations = [f"- {d}" for d in a.deviations] or ["- none"]
    return [
        "## Freeze audit",
        "",
        (
            f"- pre-registration {a.prereg_sha256[:12]} created {a.created_utc}, replaces "
            f"{(a.replaces_sha256 or NONE)[:12]}; {a.versions} version(s), first frozen "
            f"{a.first_created_utc}; post hoc: {a.post_hoc}"
        ),
        (
            f"- earliest eval file {a.earliest_eval_utc or NONE}; first freeze before it: "
            f"{a.blind}; current freeze after it: {a.frozen_after_eval} "
            "(file times are informational; the binding is by content hash)"
        ),
        (
            f"- code commit at the freeze {a.frozen_commit[:12]}, "
            f"at this report {a.report_commit[:12]}"
        ),
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


def render_markdown(report: ComparisonReport) -> str:
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


def write_report(layout: ComparisonLayout, report: ComparisonReport) -> None:
    write_text_atomic(layout.report_json, report.model_dump_json(indent=2) + "\n")
    write_text_atomic(layout.report_md, render_markdown(report))
