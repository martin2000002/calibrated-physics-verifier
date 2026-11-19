"""What ``prompts build`` prints: counts per split and family and two sha256, never a text."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from physverify.core.digests import file_sha256, text_sha256
from physverify.finetune.records import PromptRecord, records_text
from physverify.finetune.settings import SPLITS, PromptSplit

EVAL: Final[PromptSplit] = "eval"


def build_lines(path: Path, records: Sequence[PromptRecord]) -> list[str]:
    """The file's sha256, the sha256 of its eval lines, and counts per split and family."""
    eval_lines = records_text(record for record in records if record.split == EVAL)
    lines = [f"prompts: {path} sha256 {file_sha256(path)}"]
    lines.append(f"  {EVAL} lines sha256 {text_sha256(eval_lines)}")
    for split in SPLITS:
        families = Counter(record.family for record in records if record.split == split)
        per_family = ", ".join(f"{name} {n}" for name, n in families.items())
        detail = f" ({per_family})" if families else ""
        lines.append(f"  {split}: {sum(families.values())}{detail}")
    return lines
