"""Seeds and shards that depend only on identifiers, never on order or on K (D5, D10)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from physverify.core.digests import text_sha256
from physverify.core.errors import UsageError

SEED_HEX_DIGITS: Final = 8
SHARD_PATTERN: Final = re.compile(r"^(\d+)/(\d+)$")


def _digest(text: str) -> int:
    return int(text_sha256(text)[:SEED_HEX_DIGITS], 16)


def candidate_seed(base_seed: int, prompt_id: str, k: int) -> int:
    """``int(sha256(f"{base_seed}:{prompt_id}:{k}")[:8], 16)``: adding prompts or candidates
    leaves every existing seed unchanged, and the fine-tuned model reuses the base seeds."""
    return _digest(f"{base_seed}:{prompt_id}:{k}")


def candidate_id(prompt_id: str, k: int) -> str:
    return f"{prompt_id}__k{k:02d}"


@dataclass(frozen=True)
class Shard:
    index: int = 0
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1 or not 0 <= self.index < self.count:
            raise UsageError(f"invalid shard {self.index}/{self.count}")

    def owns(self, key: str) -> bool:
        return self.count == 1 or _digest(key) % self.count == self.index


WHOLE: Final = Shard()


def parse_shard(text: str | None) -> Shard:
    if text is None:
        return WHOLE
    match = SHARD_PATTERN.match(text)
    if match is None:
        raise UsageError(f"--shard must look like i/n, got {text!r}")
    return Shard(int(match.group(1)), int(match.group(2)))
