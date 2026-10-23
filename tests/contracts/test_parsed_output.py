"""Output lines that scripts/h200 parses with sed, grep or awk: the scripts' patterns and the text
the commands print must stay in step. The commands' own tests run the runners' parse functions
(tests/fixtures/steps_sh.py) on real output: verify and devscore (run:), calibrate evaluate
(evaluation:), exclusions show (record: ... sha256), agreement compare (H2:)."""

from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path

from cli_runner import CliRunner
from steps_sh import steps_read

from physverify.cli.commands import h3

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
    (
        "'^(note: |oracle/|perceived/|  R|selection )'",
        ("note: ", "selection (", "/{level.profile}: ", "  {check_id}: "),
    ),
    ("sed 's/ (.*)$//'", (" (free)", " (held by ")),
    ("'s/^campaign \\([^:]*\\):.*/\\1/p'", ("campaign ",)),
    ("'s/.* prompts at //p'", (" prompts at ",)),
    ('"^$split: [1-9][0-9]* pairs"', (" pairs; ",)),
    ("'s/.*last checkpoint //p'", ("; last checkpoint ",)),
    ("'s/.*val implicit_acc .* kl_proxy //p'", ("val implicit_acc ", " kl_proxy ")),
)
DOCTOR_READ_KEYS = ("profile", "device", "config_dir", "paths.cache_dir", "paths.raw_dir")
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
def _trees() -> tuple[ast.Module, ...]:
    paths = sorted(path for root in CODE_ROOTS for path in root.rglob("*.py"))
    return tuple(ast.parse(path.read_text(encoding="utf-8")) for path in paths)


@cache
def _string_constants() -> frozenset[str]:
    """Every literal text, f-strings with their {field} placeholders."""
    found: set[str] = set()
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(node.value)
            elif isinstance(node, ast.JoinedStr):
                found.add(ast.unparse(node)[2:-1])
    return frozenset(found)


@cache
def _line_starts() -> frozenset[str]:
    """The literal text a string or an f-string begins with (what a ^-anchored pattern sees)."""
    found: set[str] = set()
    for tree in _trees():
        joined = [node for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)]
        inner = {id(part) for node in joined for part in node.values}
        found.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in inner
        )
        found.update(
            node.values[0].value
            for node in joined
            if node.values
            and isinstance(node.values[0], ast.Constant)
            and isinstance(node.values[0].value, str)
        )
    return frozenset(found)


def _printed(fragment: str) -> bool:
    return any(fragment in text for text in _string_constants())


def _starts_a_line(prefix: str) -> bool:
    return any(text.startswith(prefix) for text in _line_starts())


def test_the_scripts_still_parse_these_patterns() -> None:
    text = _script_text()
    assert [pattern for pattern, _ in PARSED if pattern not in text] == []


def test_the_commands_still_print_the_parsed_fragments() -> None:
    fragments = [fragment for _, group in PARSED for fragment in group]
    assert [fragment for fragment in fragments if not _printed(fragment)] == []


def test_every_anchored_prefix_the_scripts_match_is_printed_by_the_code() -> None:
    prefixes = set(_ANCHORED_PREFIX.findall(_script_text())) - NOT_FROM_PHYSVERIFY
    assert prefixes
    assert sorted(prefix for prefix in prefixes if not _starts_a_line(prefix)) == []


def test_a_fragment_inside_a_line_does_not_count_as_its_start() -> None:
    assert _printed("sha256 ") and not _starts_a_line(" sha256 ")


def test_run_h3_waits_for_the_gate_line_h3_prints() -> None:
    assert h3.GATE_LINE + h3.GATE_PASSED == "  gate: passed"


def test_doctor_prints_the_keys_the_scripts_read(cli: CliRunner) -> None:
    result = cli("--set", "runtime.threads=2", "doctor")
    assert result.code == 0, result.err
    lines = result.out.splitlines()
    keys = {line.split(": ", 1)[0] for line in lines if ": " in line}
    assert set(DOCTOR_KEYS) <= keys
    assert "source: --set runtime.threads=2" in lines
    assert any(re.match(r"^docker_image: \S", line) for line in lines)
    assert any(re.match(r"^config_hash: [0-9a-f]{12}$", line) for line in lines)


def test_doctor_value_reads_the_printed_values_without_their_note(
    cli: CliRunner, tmp_path: Path
) -> None:
    result = cli("doctor")
    assert result.code == 0, result.err
    printed = dict(line.split(": ", 1) for line in result.out.splitlines() if ": " in line)
    env = {"DOCTOR": result.out}
    for key in DOCTOR_READ_KEYS:
        assert steps_read(tmp_path, "doctor_value", key, env=env) == printed[key]
    lock = steps_read(tmp_path, "doctor_value", "heavy_lock", env=env)
    assert printed["heavy_lock"].startswith(lock + " (") and " (" not in lock
