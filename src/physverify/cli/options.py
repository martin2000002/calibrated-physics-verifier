"""Option types and parsing shared by the subcommands."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.app.finetune import STOP_RULE_DEVIATION
from physverify.core.errors import UsageError

StopRuleDeviationOption = Annotated[
    str | None,
    typer.Option(
        STOP_RULE_DEVIATION,
        help="Go on although the pilot stop rule fired (recorded; H3 then has no verdict).",
    ),
]


def choice[T: str](value: str, known: tuple[T, ...], option: str) -> T:
    for candidate in known:
        if value == candidate:
            return candidate
    raise UsageError(f"{option} must be one of {list(known)}, got {value!r}")
