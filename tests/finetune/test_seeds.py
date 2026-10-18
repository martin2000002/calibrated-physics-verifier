from __future__ import annotations

import pytest

from physverify.core.errors import UsageError
from physverify.finetune.seeds import WHOLE, Shard, candidate_id, candidate_seed, parse_shard


def test_a_seed_depends_only_on_its_identifiers() -> None:
    first = [candidate_seed(0, f"p{i}", k) for i in range(5) for k in range(4)]
    more = {(i, k): candidate_seed(0, f"p{i}", k) for i in reversed(range(8)) for k in range(8)}
    assert first == [more[(i, k)] for i in range(5) for k in range(4)]
    assert candidate_seed(1, "p0", 0) != candidate_seed(0, "p0", 0)


def test_seeds_do_not_collide_over_many_ids() -> None:
    seeds = {candidate_seed(0, f"fall-{i:06d}", k) for i in range(25_000) for k in range(4)}
    assert len(seeds) >= 100_000 - 2


def test_candidate_id_is_the_clip_id() -> None:
    assert candidate_id("fall-abc", 3) == "fall-abc__k03"


def test_shards_partition_the_ids() -> None:
    ids = [candidate_id(f"p{i}", 0) for i in range(200)]
    owners = [[s for s in range(3) if Shard(s, 3).owns(c)] for c in ids]
    assert all(len(o) == 1 for o in owners)
    assert all(WHOLE.owns(c) for c in ids)


def test_parse_shard() -> None:
    assert parse_shard(None) is WHOLE
    assert parse_shard("1/4") == Shard(1, 4)
    for bad in ("4/4", "x", "1-2"):
        with pytest.raises(UsageError):
            parse_shard(bad)
