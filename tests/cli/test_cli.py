from __future__ import annotations

from types import ModuleType

import pytest
import typer
import typer.main
from cli_runner import CliRunner

from physverify.cli.context import COMMANDS_PACKAGE, check_command, command_name
from physverify.cli.main import build_app
from physverify.core.registry import discover_modules

EXPECTED = {"doctor", "synth", "extract", "verify", "diagnose", "download", "timing"}
COMMANDS = discover_modules(COMMANDS_PACKAGE, check_command, command_name)


def _leaves(module: ModuleType) -> list[tuple[str, ...]]:
    """Invocable paths of one command module: its subcommands, or itself when it has none."""
    root = typer.main.get_command(build_app(COMMANDS))
    group = root.commands[module.NAME]  # type: ignore[attr-defined]
    children = getattr(group, "commands", {})
    return [(module.NAME, name) for name in sorted(children)] or [(module.NAME,)]


LEAVES = [(module, leaf) for module in COMMANDS.values() for leaf in _leaves(module)]


def test_root_lists_the_scaffold_subcommands(cli: CliRunner) -> None:
    root = typer.main.get_command(build_app(COMMANDS))
    assert set(root.commands) >= EXPECTED  # type: ignore[attr-defined]
    result = cli("--help")
    assert result.code == 0
    assert all(name in result.out for name in EXPECTED)


@pytest.mark.parametrize(
    "leaf", sorted({leaf for _, leaf in LEAVES} | {(name,) for name in EXPECTED}), ids=" ".join
)
def test_help_exits_zero(cli: CliRunner, leaf: tuple[str, ...]) -> None:
    result = cli(*leaf, "--help")
    assert result.code == 0, result.err
    assert "Usage" in result.out


def test_expected_subcommands_exist() -> None:
    names = {leaf for _, leaf in LEAVES}
    for leaf in [
        ("synth", "pull-image"),
        ("synth", "export-gt"),
        ("synth", "list"),
        ("extract", "run"),
        ("extract", "fetch-weights"),
        ("verify",),
        ("diagnose", "stages"),
        ("diagnose", "end-to-end"),
        ("diagnose", "attribution"),
        ("diagnose", "real"),
        ("download", "fetch"),
        ("timing", "run"),
    ]:
        assert leaf in names


def test_implemented_fixture_only_needs_help(cli: CliRunner) -> None:
    result = cli("ready", "--help", commands_package="fx_commands")
    assert result.code == 0
    assert cli("ready", "go", commands_package="fx_commands").out.strip() == "ok"


def test_resource_budget_exits_2(cli: CliRunner) -> None:
    result = cli("budget", commands_package="fx_commands")
    assert result.code == 2
    assert "max_frame_bytes" in result.err


@pytest.mark.parametrize(
    ("argv", "line"),
    [
        (("--device", "cuda", "doctor"), "device: cuda"),
        (("--set", "runtime.device=cuda", "doctor"), "device: cuda"),
        (("--profile", "precise", "doctor"), "profile: precise"),
        (("doctor",), "profile: fast"),
    ],
)
def test_global_options_reach_doctor(cli: CliRunner, argv: tuple[str, ...], line: str) -> None:
    result = cli(*argv)
    assert result.code == 0, result.err
    assert line in result.out.splitlines()


def test_doctor_reports_sources_and_versions(cli: CliRunner) -> None:
    result = cli("--set", "runtime.threads=2", "doctor")
    assert result.code == 0
    assert "threads: 2" in result.out and "source: --set runtime.threads=2" in result.out
    assert any(line.startswith("torch: ") for line in result.out.splitlines())


def test_usage_errors_exit_2(cli: CliRunner) -> None:
    assert cli("--no-such-option").code == 2
    assert cli("no-such-command").code == 2
    bad = cli("--set", "runtime.threads=0", "doctor")
    assert bad.code == 2 and "[runtime] is invalid" in bad.err


@pytest.mark.parametrize(
    ("override", "line"),
    [
        (
            "runtime.threads='abc'",
            (
                "[runtime] threads: Input should be a valid integer, unable to parse string as"
                " an integer (got 'abc')"
            ),
        ),
        ("runtime.device='gpu'", "[runtime] device must be cpu, cuda or cuda:N, got 'gpu'"),
        ("runtime.nope=1", "[runtime] nope: Extra inputs are not permitted (got 1)"),
    ],
)
def test_a_config_error_is_one_line_per_field_without_urls(
    cli: CliRunner, override: str, line: str
) -> None:
    result = cli("--set", override, "doctor")
    assert result.code == 2
    assert result.err.splitlines() == ["error: [runtime] is invalid:", line]


def test_missing_config_dir_exits_2(cli: CliRunner) -> None:
    result = cli("--config-dir", "/nonexistent/physverify", "doctor")
    assert result.code == 2 and "config directory not found" in result.err
    assert "pass --config-dir or set PHYSVERIFY_CONFIG_DIR" in result.err


def test_log_level_is_validated_by_logging_setup(cli: CliRunner) -> None:
    assert cli("--log-level", "DEBUG", "doctor").code == 0


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("--log-level", "loud"), "Invalid value for '--log-level': must be one of DEBUG"),
        (("--device", "gpu"), "Invalid value for '--device': must be cpu, cuda or cuda:N"),
        (("--lock-timeout", "-1"), "Invalid value for '--lock-timeout': must be 0 or more"),
        (("--profile", "bogus"), "unknown profile 'bogus'; known: ['fast', 'precise']"),
    ],
)
def test_bad_global_option_values_are_usage_errors(
    cli: CliRunner, argv: tuple[str, ...], message: str
) -> None:
    result = cli(*argv, "doctor")
    assert result.code == 2
    assert message in result.err


def test_app_is_a_typer_instance() -> None:
    assert all(isinstance(module.app, typer.Typer) for module in COMMANDS.values())
