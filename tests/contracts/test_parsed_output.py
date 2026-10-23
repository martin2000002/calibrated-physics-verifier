"""Output lines that scripts/h200 parses with sed, grep or awk: the scripts' patterns and the text
the commands print must stay in step."""

from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path

from cli_runner import CliRunner

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "h200"
CODE_ROOTS = (REPO / "src", REPO / "scripts")

PARSED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("s/^docker_image: //p", ("docker_image: ",)),
    ("^container_engine: $ENGINE", ("container_engine: ",)),
    ('"source: --set $name"', ("source: ", "--set ")),
    ('sed -n "s/^$1: //p" <<<"$DOCTOR"', ("config_dir: ", "config_hash: ", "paths.")),
    ("s/^record: .* sha256 \\([0-9a-f]*\\)$/\\1/p", ("record: ", " sha256 ")),
    ("'^real dataset .* skipped'", ("real dataset ", " skipped: ")),
    ("'/^evaluation: /{s///p;q}'", ("evaluation: ",)),
    ("'/^run: /{s///p;q}'", ("run: ",)),
    ("'/^preregistration: /{s///p;q}'", ("preregistration: ",)),
    ("'/^H2: /{p;q}'", ("H2: ",)),
    ('GATE_PASSED_LINE="  gate: passed"', ("  gate: ", "passed")),
    ("'s/.* scenes, \\([0-9]*\\) clips, .*/\\1/p'", (" scenes, ", " clips, ")),
    ("'^(note: |oracle/|perceived/|  R|selection )'", ("note: ", "selection (")),
    ("'s/^campaign \\([^:]*\\):.*/\\1/p'", ("campaign ",)),
    ("'s/.* prompts at //p'", (" prompts at ",)),
    ('"^$split: [1-9][0-9]* pairs"', (" pairs; ",)),
    ("'s/.*last checkpoint //p'", ("; last checkpoint ",)),
    ("'s/.*val implicit_acc .* kl_proxy //p'", ("val implicit_acc ", " kl_proxy ")),
)
DOCTOR_KEYS = (
    "profile",
    "device",
    "config_dir",
    "config_hash",
    "paths.raw_dir",
    "paths.cache_dir",
    "paths.runs_dir",
    "paths.calibration_dir",
    "paths.error_profile",
    "paths.thresholds",
    "docker_image",
    "container_engine",
    "heavy_lock",
)
NOT_FROM_PHYSVERIFY = frozenset({"GPU "})
_ANCHORED_PREFIX = re.compile(
    r"""(?:sed -n ['"](?:/\^|s/\^)|grep -[a-zA-Z]+ (?:-- )?['"]\^)([^'"/\\.*\[$(|]+)"""
)


@cache
def _script_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(SCRIPTS.glob("*.sh")))


@cache
def _string_constants() -> frozenset[str]:
    found: set[str] = set()
    for root in CODE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found.update(
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            )
    return frozenset(found)


def _printed(fragment: str) -> bool:
    return any(fragment in text for text in _string_constants())


def test_the_scripts_still_parse_these_patterns() -> None:
    text = _script_text()
    assert [pattern for pattern, _ in PARSED if pattern not in text] == []


def test_the_commands_still_print_the_parsed_fragments() -> None:
    fragments = [fragment for _, group in PARSED for fragment in group]
    assert [fragment for fragment in fragments if not _printed(fragment)] == []


def test_every_anchored_prefix_the_scripts_match_is_printed_by_the_code() -> None:
    prefixes = set(_ANCHORED_PREFIX.findall(_script_text())) - NOT_FROM_PHYSVERIFY
    assert sorted(prefix for prefix in prefixes if not _printed(prefix)) == []


def test_doctor_prints_the_keys_the_scripts_read(cli: CliRunner) -> None:
    result = cli("--set", "runtime.threads=2", "doctor")
    assert result.code == 0, result.err
    lines = result.out.splitlines()
    keys = {line.split(": ", 1)[0] for line in lines if ": " in line}
    assert set(DOCTOR_KEYS) <= keys
    assert "source: --set runtime.threads=2" in lines
    assert any(re.match(r"^docker_image: \S", line) for line in lines)
    assert any(re.match(r"^config_hash: [0-9a-f]{12}$", line) for line in lines)
