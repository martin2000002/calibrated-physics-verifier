"""Process labels (plan items, review codes, banners) must not grow in code, comments or configs.

The baseline in ``contracts/data/process_labels_baseline.json`` holds today's counts per file and
category; counts may only fall. Regenerate it with ``python tests/test_process_labels.py``.
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "tests" / "contracts" / "data" / "process_labels_baseline.json"
SCANNED_DIRS = ("src", "scripts", "tests", "configs", "tools")
EXCLUDED_SUFFIXES = frozenset({".json", ".jsonl", ".npz", ".pyc"})
EXCLUDED_NAMES = frozenset({"uv.lock", "local.toml", "py.typed"})
EXCLUDED_PARTS = frozenset({"golden", "__pycache__", ".pytest_cache"})
EXCLUDED_PREFIXES = ("tests/synth/data/", "tools/equivalence/expected/")
SELF = Path(__file__).resolve()

JARGON = re.compile(
    r"\bADRs?\b|\bADR[ -]?[0-9]{1,4}\b|\bA([3-9]|1[0-3])\b|\bA8 ?v2\b|\ba8v2\b|\bH[123]\b"
    r"|\bJuicio\b|\bJ[0-9]{1,2}\b|\bplan-it[0-9]\b|\bit[1-5]\b|\bD[0-9]{1,2}(\.[0-9]+)?\b"
    r"|\bAC-[0-9]+\b|\bT[0-9]{1,2}\b|[Pp]lan[- ]congelado|[Ee]nmienda|\bC_F\b|\bC_V1\b"
    r"|\bWP-[A-Z0-9]+\b|\bP[1-5]\b|§|\b20[0-9]{2}-[01][0-9]-[0-3][0-9]\b"
)
IDENT = re.compile(r"(^|_)(h[123]|a8v2|a([3-9]|1[0-3])|it[1-5]|d9)(_|$)")
CAMEL = re.compile(r"(?<![A-Z])(H[123]|A(?:[3-9]|1[0-3])|A8V2|It[1-5])(?=[A-Z][a-z]|$)")
TEST_NAME = re.compile(
    r"^test_(.*_)?(h[123]|a8v2|a[0-9]{1,2}|it[1-5]|d[0-9]{1,2}|j[0-9]{1,2}|p[1-5]|t[0-9]{1,2}"
    r"|ac[0-9]+|adr[0-9]*)(_|$)"
)
BANNER = re.compile(r"^\s*#\s*(-{3,}|={3,})|^\s*#.*(-{5,}|={5,})\s*$")
PROCESS_ATTRS = re.compile(r"\b(OWNER|IMPLEMENTED|not_implemented|NotImplementedCommandError)\b")
ALLOWED_IDENTIFIERS = frozenset(
    {
        "h1_scope",
        "h1_pass",
        "meets_h1",
        "heldout_h1",
        "h1_primary",
        "h2_verdict",
        "h2_group",
        "h2_dev",
        "h2_test",
        "h3",
        "h3_report",
        "h3_families_without_target",
        "j3",
        "j13",
    }
)
ALLOWED_IDENTIFIER_PREFIX = "PHYSVERIFY__H3__"
DEV_SCORE_FIELD = frozenset({"h1"})
ALLOWED_IDENTIFIERS_BY_PATH = {
    "src/physverify/calibration/dev_score.py": DEV_SCORE_FIELD,
    "src/physverify/calibration/summary.py": DEV_SCORE_FIELD,
    "tests/calibration/test_devscore.py": DEV_SCORE_FIELD,
}

COMMENT = "comment_or_docstring"
LITERAL = "literal"
IDENTIFIER = "identifier"
CAMEL_HITS = "camel_identifier"
TEST_NAME_HITS = "test_name"
BANNER_HITS = "banner"
PROCESS = "process_attr"
TEXT = "config_or_script_text"

Counts = dict[str, dict[str, int]]


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
    paths = (path for folder in SCANNED_DIRS for path in (REPO / folder).rglob("*"))
    return sorted(path for path in paths if _is_scanned(path))


def _docstring_starts(tree: ast.AST) -> set[tuple[int, int]]:
    """Start of every bare string statement, attribute docstrings included."""
    return {
        (node.lineno, node.col_offset)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _test_name_hits(tree: ast.AST) -> int:
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    return sum(
        isinstance(node, functions) and bool(TEST_NAME.search(node.name)) for node in ast.walk(tree)
    )


def _is_allowed_identifier(name: str, allowed: frozenset[str]) -> bool:
    return (
        name in ALLOWED_IDENTIFIERS or name in allowed or name.startswith(ALLOWED_IDENTIFIER_PREFIX)
    )


def _token_hits(
    token: tokenize.TokenInfo, docstrings: set[tuple[int, int]], allowed: frozenset[str]
) -> Counter[str]:
    text = token.string
    hits = Counter({PROCESS: len(PROCESS_ATTRS.findall(text))})
    if token.type == tokenize.COMMENT:
        hits[COMMENT] += len(JARGON.findall(text))
        hits[BANNER_HITS] += bool(BANNER.search(text))
    elif token.type == tokenize.STRING and token.start in docstrings:
        hits[COMMENT] += len(JARGON.findall(text))
    elif token.type in (tokenize.STRING, tokenize.FSTRING_MIDDLE):
        hits[LITERAL] += len(JARGON.findall(text))
    elif token.type == tokenize.NAME and not _is_allowed_identifier(text, allowed):
        hits[IDENTIFIER] += bool(IDENT.search(text))
        hits[CAMEL_HITS] += len(CAMEL.findall(text))
    return hits


def _python_hits(source: str, relative: str) -> Counter[str]:
    tree = ast.parse(source)
    docstrings = _docstring_starts(tree)
    allowed = ALLOWED_IDENTIFIERS_BY_PATH.get(relative, frozenset())
    hits: Counter[str] = Counter({TEST_NAME_HITS: _test_name_hits(tree)})
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        hits.update(_token_hits(token, docstrings, allowed))
    return hits


def _text_hits(source: str) -> Counter[str]:
    hits: Counter[str] = Counter()
    for line in source.splitlines():
        hits[TEXT] += len(JARGON.findall(line))
        hits[BANNER_HITS] += bool(BANNER.search(line))
        hits[PROCESS] += len(PROCESS_ATTRS.findall(line))
    return hits


def _file_hits(path: Path) -> Counter[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Counter()
    relative = path.relative_to(REPO).as_posix()
    hits = _python_hits(source, relative) if path.suffix == ".py" else _text_hits(source)
    return +hits


def current_counts() -> Counts:
    counts: Counts = {}
    for path in _scanned_files():
        hits = _file_hits(path)
        if hits:
            counts[path.relative_to(REPO).as_posix()] = dict(sorted(hits.items()))
    return counts


def _baseline() -> Counts:
    baseline: Counts = json.loads(BASELINE.read_text(encoding="utf-8"))
    return baseline


def _increases(current: Counts, baseline: Counts) -> list[str]:
    return [
        f"{path} {category}: {count} > {baseline.get(path, {}).get(category, 0)}"
        for path, hits in sorted(current.items())
        for category, count in hits.items()
        if count > baseline.get(path, {}).get(category, 0)
    ]


def _totals(counts: Counts) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for hits in counts.values():
        totals.update(hits)
    return dict(sorted(totals.items()))


def test_process_label_counts_never_increase() -> None:
    increases = _increases(current_counts(), _baseline())
    assert not increases, "new process labels:\n" + "\n".join(increases)


def test_process_label_totals() -> None:
    current = _totals(current_counts())
    print(f"process labels now {current}, baseline {_totals(_baseline())}")
    assert all(count >= 0 for count in current.values())


if __name__ == "__main__":
    BASELINE.write_text(json.dumps(current_counts(), indent=1, sort_keys=True) + "\n")
