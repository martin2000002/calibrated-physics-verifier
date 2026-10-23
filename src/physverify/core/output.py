"""The sink a service writes its text lines to (``typer.echo``, ``list.append``, ...)."""

from __future__ import annotations

from collections.abc import Callable

type Echo = Callable[[str], None]
