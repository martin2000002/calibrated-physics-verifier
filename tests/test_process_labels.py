"""No process labels (plan items, review codes, banners) in code, configs, scripts or docs.

A label survives only where something reads it back: a persisted field or file name, a parsed
stdout line, a directory that already exists on the H200. Each survivor is listed below with its
path, the exact fragment, how many label hits that fragment covers and its reader, so a new use
fails and so does an entry nothing uses any more.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ("src", "scripts", "tests", "configs", "tools")
SCANNED_FILES = ("README.md", "CLAUDE.md")
EXCLUDED_SUFFIXES = frozenset({".jsonl", ".npz", ".pyc"})
EXCLUDED_NAMES = frozenset({"uv.lock", "local.toml", "py.typed"})
EXCLUDED_PARTS = frozenset({"golden", "__pycache__", ".pytest_cache"})
EXCLUDED_PREFIXES = (
    "tests/synth/data/",
    "tests/contracts/data/pre_split_configs.json",
    "tools/equivalence/expected/",
)
SELF = Path(__file__).resolve()

JARGON = re.compile(
    r"\bADRs?\b|\bADR[ -]?[0-9]{1,4}\b|\bA([3-9]|1[0-3])\b|\bA8 ?v2\b|\ba8v2\b|\bH[123]\b"
    r"|\bJuicio\b|\bJ[0-9]{1,2}\b|\bplan-it[0-9]\b|\bit[1-5]\b|\bD[0-9]{1,2}(\.[0-9]+)?\b"
    r"|\bAC-[0-9]+\b|\bT[0-9]{1,2}\b|[Pp]lan[- ]congelado|[Ee]nmienda|\bC_F\b|\bC_V1\b"
    r"|\b(?i:wp)-?[0-9]+[a-z]?\b|\bWP-[A-Z0-9]+\b|\bP[1-5]\b|§|\b20[0-9]{2}-[01][0-9]-[0-3][0-9]\b"
    r"|(?<![-\w])[Aa]udit [A-Za-z]?[0-9]|\b[Jj]udge change|(?<![-\w])[Aa]mendments?\b"
    r"|\bKnowledge/"
)
IDENT = re.compile(r"(^|_)(h[123]|a8v2|a([3-9]|1[0-3])|it[1-5]|d9)(_|$)", re.IGNORECASE)
CAMEL = re.compile(r"(?<![A-Z])(H[123]|A(?:[3-9]|1[0-3])|A8V2|It[1-5])(?=[A-Z][a-z]|$)")
TEST_NAME = re.compile(
    r"^test_(.*_)?(h[123]|a8v2|a[0-9]{1,2}|it[1-5]|d[0-9]{1,2}|j[0-9]{1,2}|p[1-5]|t[0-9]{1,2}"
    r"|ac[0-9]+|adr[0-9]*)(_|$)"
)
BANNER = re.compile(r"^\s*#\s*(-{3,}|={3,})|^\s*#.*(-{5,}|={5,})\s*$")
SHELL_NAME = re.compile(r"\$\{?([A-Za-z_]\w*)|\b([A-Za-z_]\w*)(?==)")
PROCESS_ATTRS = re.compile(r"\b(OWNER|IMPLEMENTED|not_implemented|NotImplementedCommandError)\b")

COMMENT = "comment_or_docstring"
LITERAL = "literal"
IDENTIFIER = "identifier"
TEST_NAME_HITS = "test_name"
BANNER_HITS = "banner"
PROCESS = "process_attr"
TEXT = "config_or_script_text"
FORBIDDEN = (COMMENT, BANNER_HITS, PROCESS, TEST_NAME_HITS)


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    category: str
    context: str
    span: tuple[int, int]

    @property
    def label(self) -> str:
        return self.context[self.span[0] : self.span[1]]


@dataclass(frozen=True)
class Kept:
    """A fragment that may hold labels in one file: ``count`` label hits fall inside it."""

    path: str
    fragment: str
    count: int
    reader: str


SEALED_SCOPE_NOTE = "SCOPE_NOTE is written to devscore.json (scope_note), pinned by a golden blob"
PRIMARY_HEADING = "evaluation.md heading that scripts/h200/run_query.py:33 finds the section by"
AGREEMENT_LINE = "workflow.py stdout line that scripts/h200/run_human_agreement.sh:385 parses"
AGREEMENT_SED = "the sed expression of scripts/h200/run_human_agreement.sh:385"
CHAIN_DIR = "on-disk calibration-chain directory on the H200 ($H200_ROOT/a8v2, _common.sh:9)"
ARM_IDS = "arm ids of perception_arms.tsv: existing iteration directories and CHAIN_ARM values"
ITERATION_IDS = "fixture: iteration directory ids as they exist on the H200"

KEPT_FRAGMENTS = (
    Kept(
        "README.md",
        "`--a8v2-dir`",
        1,
        "renamed-names table: the old `results` option that stored launchers may still pass",
    ),
    Kept("scripts/h200/_common.sh", '"$H200_ROOT/a8v2"', 1, CHAIN_DIR),
    Kept("scripts/h200/perception_arms.tsv", "it3-r", 1, ARM_IDS),
    Kept("scripts/h200/perception_arms.tsv", "it3-ak", 1, ARM_IDS),
    Kept("scripts/h200/perception_arms.tsv", "it3-ap", 1, ARM_IDS),
    Kept("scripts/h200/perception_arms.tsv", "it3-c", 1, ARM_IDS),
    Kept("scripts/h200/run_human_agreement.sh", "sed -n '/^H2: /{p;q}'", 1, AGREEMENT_LINE),
    Kept("scripts/h200/run_query.py", '"## H1 primary"', 1, PRIMARY_HEADING),
    Kept(
        "src/physverify/calibration/dev_score.py",
        "(calibrate stage-sigma, D10) and H1 (Kendall",
        2,
        SEALED_SCOPE_NOTE,
    ),
    Kept("src/physverify/calibration/report.py", "## H1 primary (scenes ", 1, PRIMARY_HEADING),
    Kept(
        "src/physverify/freeze/selection.py",
        "(plan-congelado J4)",
        2,
        "J4_RULE is persisted in freeze.json (selection.rule)",
    ),
    Kept("src/physverify/human_agreement/workflow.py", "H2: ", 2, AGREEMENT_LINE),
    Kept(
        "tests/calibration/test_monotonicity_primary.py",
        "## H1 primary (scenes s000, s011-s039) vs all scenes",
        1,
        PRIMARY_HEADING,
    ),
    Kept("tests/calibration/test_monotonicity_primary.py", '"H1 primary"', 1, PRIMARY_HEADING),
    Kept("tests/contracts/test_parsed_output.py", "'/^H2: /{p;q}'", 1, AGREEMENT_SED),
    Kept("tests/contracts/test_parsed_output.py", '"H2: "', 1, AGREEMENT_LINE),
    Kept(
        "tests/fixtures/videophy_world.py",
        '"2026-10-17"',
        1,
        "fixture date of the reviewed-subset file the tests build",
    ),
    Kept("tests/human_agreement/test_cli.py", '"/^H2: /{p;q}"', 1, AGREEMENT_SED),
    Kept("tests/human_agreement/test_cli.py", "H2: ", 2, AGREEMENT_LINE),
    Kept(
        "tests/reporting/reporting_cases.py",
        '"a8v2"',
        1,
        "chain directory of the results scenario; the harness pairs output trees by path and "
        "the base side writes it",
    ),
    Kept(
        "tests/scripts/human_agreement/test_human_agreement_script.py",
        '"H2: supported"',
        1,
        AGREEMENT_LINE,
    ),
    Kept("tests/scripts/human_agreement/test_human_agreement_script.py", '"a8v2"', 1, CHAIN_DIR),
    Kept(
        "tests/scripts/human_agreement/test_human_agreement_script.py",
        '"- H2: supported"',
        1,
        "the summary line the script builds from the parsed H2 line",
    ),
    Kept("tests/scripts/test_freeze_query.py", "it1-fixes", 2, ITERATION_IDS),
    Kept("tests/scripts/test_freeze_query.py", "it2-state", 9, ITERATION_IDS),
    Kept("tests/scripts/test_freeze_query.py", "it2-x", 1, ITERATION_IDS),
    Kept("tests/scripts/test_freeze_query.py", "it3-arms", 1, ITERATION_IDS),
    Kept("tests/scripts/test_freeze_query.py", "it4-selfrel", 3, ITERATION_IDS),
    Kept("tests/scripts/test_perception_iteration_script.py", '"a8v2"', 1, CHAIN_DIR),
    Kept("tests/scripts/test_perception_iteration_script.py", '"a8v2-cuda/it"', 1, CHAIN_DIR),
    Kept("tests/scripts/test_perception_iteration_script.py", "it3-r", 3, ARM_IDS),
    Kept("tests/scripts/test_perception_iteration_script.py", "it3-ak", 2, ARM_IDS),
    Kept("tests/scripts/test_perception_iteration_script.py", "it3-ap", 1, ARM_IDS),
    Kept("tests/scripts/test_perception_iteration_script.py", "it3-c", 6, ARM_IDS),
    Kept("tests/scripts/test_perception_iteration_script.py", "it3-c-cuda", 1, ARM_IDS),
    Kept("tests/scripts/test_perception_iteration_script.py", "it3-arms-cpu", 1, ARM_IDS),
    Kept("tests/scripts/test_perception_iteration_script.py", "it3-r-it3-ak", 4, ARM_IDS),
    Kept("tests/scripts/test_run_freeze_script.py", "it1-fixes", 2, ITERATION_IDS),
    Kept("tests/scripts/test_run_freeze_script.py", "it2-state", 5, ITERATION_IDS),
    Kept("tests/scripts/test_run_freeze_script.py", '"a8v2"', 1, CHAIN_DIR),
    Kept("tests/scripts/test_run_query.py", "## H1 primary", 2, PRIMARY_HEADING),
    Kept(
        "tools/equivalence/driver.py",
        '"--a8v2-dir"',
        1,
        "results option of the base commit 9eb5888, which the harness runs",
    ),
)

PERSISTED_FIELD = "field name persisted in a written report"
AUTOEVAL_ROOT = "PHYSVERIFY__H3__AUTOEVAL__ROOT"
IDENTIFIER_READERS = {
    "h1": "CheckDevScore.h1, a key of devscore.json",
    "h1_pass": PERSISTED_FIELD,
    "meets_h1": PERSISTED_FIELD,
    "heldout_h1": f"{PERSISTED_FIELD}, and the function that fills it",
    "h1_primary": PERSISTED_FIELD,
    "h1_scope": "the [h1_scope] config section and its loader",
    "h2_verdict": "metrics.json and results.json key, and the function that builds it",
    "h2_group": "metrics.json key naming the primary group, a field of its typed model",
    "h3_families_without_target": "field of the sealed exclusion decision record",
    AUTOEVAL_ROOT: "environment override of root in the persisted "
    "[h3.autoeval] config section, read by the settings loader",
}
KEPT_IDENTIFIERS: Mapping[str, Mapping[str, int]] = {
    "scripts/h200/run_generator_comparison.sh": {AUTOEVAL_ROOT: 2},
    "scripts/h200/setup_autoeval.sh": {AUTOEVAL_ROOT: 1},
    "scripts/h200/smoke_autoeval.sh": {AUTOEVAL_ROOT: 2},
    "src/physverify/calibration/artifact.py": {
        "h1_pass": 1,
        "meets_h1": 1,
        "heldout_h1": 1,
        "h1_primary": 1,
    },
    "src/physverify/calibration/dev_score.py": {"h1": 2, "meets_h1": 5, "h1_pass": 1},
    "src/physverify/calibration/evaluate.py": {"heldout_h1": 3, "h1_primary": 1},
    "src/physverify/calibration/exclusion_decision.py": {"h3_families_without_target": 2},
    "src/physverify/calibration/heldout_monotonicity.py": {
        "heldout_h1": 1,
        "meets_h1": 4,
        "h1_pass": 1,
    },
    "src/physverify/calibration/monotonicity.py": {"h1_pass": 1},
    "src/physverify/calibration/report.py": {"heldout_h1": 2, "h1_primary": 2, "h1_pass": 1},
    "src/physverify/calibration/session.py": {"h1_scope": 2},
    "src/physverify/calibration/summary.py": {"h1": 3, "meets_h1": 1},
    "src/physverify/human_agreement/comparison.py": {"h2_verdict": 3, "h2_group": 1},
    "src/physverify/human_agreement/metrics.py": {"h2_verdict": 1, "h2_group": 2},
    "src/physverify/human_agreement/policy_metrics.py": {"h2_verdict": 1},
    "src/physverify/human_agreement/report.py": {"h2_verdict": 1},
    "src/physverify/human_agreement/workflow.py": {"h2_verdict": 1},
    "src/physverify/reporting/numbers.py": {"h2_verdict": 2},
    "src/physverify/synth/scene_scope.py": {"h1_scope": 1},
    "src/physverify/verification/exclusion_record.py": {"h3_families_without_target": 1},
    "tests/calibration/exclusion_cases.py": {"h1_pass": 1, "h3_families_without_target": 1},
    "tests/calibration/stratum_cases.py": {"h1_pass": 1},
    "tests/calibration/test_devscore.py": {"h1": 2, "meets_h1": 1, "h1_pass": 1},
    "tests/calibration/test_exclusions.py": {"h3_families_without_target": 1},
    "tests/calibration/test_fit_fixture.py": {"h1_pass": 1},
    "tests/calibration/test_heldout_monotonicity.py": {
        "heldout_h1": 16,
        "h1_primary": 6,
        "meets_h1": 2,
    },
    "tests/calibration/test_monotonicity.py": {"h1_pass": 2},
    "tests/calibration/test_monotonicity_by_dataset.py": {"h1_scope": 3},
    "tests/calibration/test_monotonicity_primary.py": {
        "h1_scope": 4,
        "h1_primary": 8,
        "h1_pass": 1,
    },
    "tests/human_agreement/test_stats.py": {"h2_verdict": 7},
    "tests/reporting/reporting_cases.py": {"h3_families_without_target": 1},
    "tests/reporting/test_results.py": {"h1_pass": 1},
    "tests/scripts/test_generator_comparison_scripts.py": {AUTOEVAL_ROOT: 1},
}


def _is_scanned(path: Path) -> bool:
    relative = path.relative_to(REPO).as_posix()
    return (
        path.is_file()
        and path != SELF
        and path.suffix not in EXCLUDED_SUFFIXES
        and path.name not in EXCLUDED_NAMES
        and not EXCLUDED_PARTS.intersection(path.parts)
        and not relative.startswith(EXCLUDED_PREFIXES)
    )


def _scanned_files() -> list[Path]:
    paths = [path for folder in SCANNED_DIRS for path in (REPO / folder).rglob("*")]
    paths += [REPO / name for name in SCANNED_FILES]
    return sorted(path for path in paths if _is_scanned(path))


def _bare_string_starts(tree: ast.AST) -> set[tuple[int, int]]:
    """Start of every bare string statement, attribute docstrings included."""
    return {
        (node.lineno, node.col_offset)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _matches(
    pattern: re.Pattern[str], path: str, line: int, category: str, text: str
) -> Iterator[Hit]:
    for match in pattern.finditer(text):
        yield Hit(path, line, category, text, match.span())


def _token_category(token: tokenize.TokenInfo, bare_strings: set[tuple[int, int]]) -> str | None:
    if token.type == tokenize.COMMENT:
        return COMMENT
    if token.type == tokenize.STRING:
        return COMMENT if token.start in bare_strings else LITERAL
    if token.type == tokenize.FSTRING_MIDDLE:
        return LITERAL
    return None


def _token_hits(
    token: tokenize.TokenInfo, bare_strings: set[tuple[int, int]], path: str
) -> Iterator[Hit]:
    line, text = token.start[0], token.string
    yield from _matches(PROCESS_ATTRS, path, line, PROCESS, text)
    category = _token_category(token, bare_strings)
    if category is not None:
        yield from _matches(JARGON, path, line, category, text)
    if token.type == tokenize.COMMENT:
        yield from _matches(BANNER, path, line, BANNER_HITS, text)
    if token.type == tokenize.NAME and (IDENT.search(text) or CAMEL.search(text)):
        yield Hit(path, line, IDENTIFIER, text, (0, len(text)))


def _python_hits(source: str, path: str) -> Iterator[Hit]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and TEST_NAME.search(
            node.name
        ):
            yield Hit(path, node.lineno, TEST_NAME_HITS, node.name, (0, len(node.name)))
    bare_strings = _bare_string_starts(tree)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        yield from _token_hits(token, bare_strings, path)


def _shell_name_hits(line: str, path: str, number: int) -> Iterator[Hit]:
    for match in SHELL_NAME.finditer(line):
        group = 1 if match.group(1) else 2
        if IDENT.search(match.group(group)):
            yield Hit(path, number, IDENTIFIER, line, match.span(group))


def _text_hits(source: str, path: str) -> Iterator[Hit]:
    for number, line in enumerate(source.splitlines(), start=1):
        yield from _matches(JARGON, path, number, TEXT, line)
        yield from _matches(BANNER, path, number, BANNER_HITS, line)
        yield from _matches(PROCESS_ATTRS, path, number, PROCESS, line)
        if path.endswith(".sh"):
            yield from _shell_name_hits(line, path, number)


def file_hits(path: Path) -> list[Hit]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    relative = path.relative_to(REPO).as_posix()
    hits = _python_hits(source, relative) if path.suffix == ".py" else _text_hits(source, relative)
    return list(hits)


def all_hits() -> list[Hit]:
    return [hit for path in _scanned_files() for hit in file_hits(path)]


def totals(hits: list[Hit]) -> dict[str, int]:
    return dict(sorted(Counter(hit.category for hit in hits).items()))


def _covering_fragment(hit: Hit, fragments: list[str]) -> str | None:
    """The longest kept fragment that contains the hit's label at the hit's position."""
    start, end = hit.span
    covering = [
        fragment
        for fragment in fragments
        if any(
            hit.context.startswith(fragment, offset)
            for offset in range(max(0, end - len(fragment)), start + 1)
        )
    ]
    return max(covering, key=len, default=None)


def _fragment_counts(hits: list[Hit]) -> tuple[Counter[tuple[str, str]], list[Hit]]:
    fragments: dict[str, list[str]] = {}
    for kept in KEPT_FRAGMENTS:
        fragments.setdefault(kept.path, []).append(kept.fragment)
    counts: Counter[tuple[str, str]] = Counter()
    uncovered = []
    for hit in hits:
        fragment = _covering_fragment(hit, fragments.get(hit.path, []))
        if fragment is None:
            uncovered.append(hit)
        else:
            counts[hit.path, fragment] += 1
    return counts, uncovered


def _describe(hits: list[Hit]) -> str:
    return "\n".join(f"{hit.path}:{hit.line} {hit.category} {hit.label!r}" for hit in hits)


def _mismatches(found: Counter[tuple[str, str]], kept: Mapping[tuple[str, str], int]) -> list[str]:
    keys = sorted(set(found) | set(kept))
    return [
        f"{path} {fragment!r}: found {found[path, fragment]}, kept {kept.get((path, fragment), 0)}"
        for path, fragment in keys
        if found[path, fragment] != kept.get((path, fragment), 0)
    ]


def test_no_labels_in_comments_banners_test_names_or_process_attributes() -> None:
    forbidden = [hit for hit in all_hits() if hit.category in FORBIDDEN]
    assert not forbidden, "process labels:\n" + _describe(forbidden)


def test_label_identifiers_are_only_the_kept_names() -> None:
    found = Counter((hit.path, hit.label) for hit in all_hits() if hit.category == IDENTIFIER)
    kept = {
        (path, name): n for path, names in KEPT_IDENTIFIERS.items() for name, n in names.items()
    }
    assert set(IDENTIFIER_READERS) == {name for _, name in kept}
    mismatches = _mismatches(found, kept)
    assert not mismatches, "label identifiers differ from KEPT_IDENTIFIERS:\n" + "\n".join(
        mismatches
    )


def test_label_literals_and_text_are_only_the_kept_fragments() -> None:
    hits = [hit for hit in all_hits() if hit.category in (LITERAL, TEXT)]
    counts, uncovered = _fragment_counts(hits)
    assert not uncovered, "labels outside KEPT_FRAGMENTS:\n" + _describe(uncovered)
    kept = {(entry.path, entry.fragment): entry.count for entry in KEPT_FRAGMENTS}
    assert len(kept) == len(KEPT_FRAGMENTS), "duplicate KEPT_FRAGMENTS entry"
    mismatches = _mismatches(counts, kept)
    assert not mismatches, "kept fragments differ from KEPT_FRAGMENTS:\n" + "\n".join(mismatches)


def test_no_labels_in_file_or_directory_names() -> None:
    names = {
        part
        for folder in SCANNED_DIRS
        for path in (REPO / folder).rglob("*")
        if not EXCLUDED_PARTS.intersection(path.parts)
        for part in path.relative_to(REPO).parts
    }
    labelled = sorted(
        name for name in names if JARGON.search(re.sub(r"[-_.]", " ", name)) or IDENT.search(name)
    )
    assert not labelled, f"process labels in names: {labelled}"


@pytest.mark.parametrize(
    "text",
    [
        "audit A3 found",
        "the audit 2 notes",
        "judge change",
        "per amendment 7",
        "Knowledge/x.md",
        "since WP7",
        "wp9-c-help-text.diff",
        "WP-X",
        "WP4a",
    ],
)
def test_process_references_are_labels(text: str) -> None:
    assert JARGON.search(text)


@pytest.mark.parametrize(
    "text", ["re-audit", "--audit", "subset --audit 3", "audit_lines", "audit(dataset, 2)"]
)
def test_command_words_are_not_labels(text: str) -> None:
    assert not JARGON.search(text)


@pytest.mark.parametrize("name", ["h3_dir", "H3_DIR", "KEYS_BEFORE_IT3", "PHYSVERIFY__H3__ROOT"])
def test_snake_names_with_labels_are_hits(name: str) -> None:
    assert IDENT.search(name)


@pytest.mark.parametrize(
    ("line", "names"),
    [
        ("H2_LINK=h2-latest", ["H2_LINK"]),
        ('ln -s x "$dir/$H2_LINK" "${A4_ROOT:-}"', ["H2_LINK", "A4_ROOT"]),
        ('chain_dir="$H200_ROOT/a8v2"', []),
    ],
)
def test_shell_variable_names_with_labels_are_hits(line: str, names: list[str]) -> None:
    assert [hit.label for hit in _shell_name_hits(line, "x.sh", 1)] == names


if __name__ == "__main__":
    print(totals(all_hits()))
