"""Size limits of the layout: thin CLI commands and modules small enough to review."""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "physverify"
COMMANDS = PACKAGE / "cli" / "commands"
MAX_COMMAND_LINES = 150
MAX_MODULE_LINES = 400


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _oversized(paths: list[Path], limit: int) -> list[str]:
    return [
        f"{path.relative_to(PACKAGE)}: {n}" for path in paths if (n := _line_count(path)) > limit
    ]


def test_cli_modules_are_thin() -> None:
    commands = sorted(COMMANDS.glob("*.py"))
    private = [
        path.name for path in commands if path.name.startswith("_") and path.name != "__init__.py"
    ]
    assert not private, f"private helpers belong below the CLI: {private}"
    oversized = _oversized(commands, MAX_COMMAND_LINES)
    assert not oversized, f"commands over {MAX_COMMAND_LINES} lines: {oversized}"


def test_modules_fit() -> None:
    oversized = _oversized(sorted(PACKAGE.rglob("*.py")), MAX_MODULE_LINES)
    assert not oversized, f"modules over {MAX_MODULE_LINES} lines: {oversized}"
