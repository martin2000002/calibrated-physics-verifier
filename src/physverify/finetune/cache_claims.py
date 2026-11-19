"""Which candidate configuration owns each ``local_video`` clip id of one stage cache.

A clip id must name one video for as long as its cache entries live (why: ``layout``).
``finetune_claims.json`` at the cache root records the generation hash of every campaign that
scored into the cache and, per clip id, the candidate ``config_hash`` it was extracted for.
Scoring is refused while another configuration claims the cache: a campaign with another
generation hash (its own included, when the settings changed) or a clip id last extracted for
another ``config_hash``; ``purge`` drops those clips' cache entries instead. A change of
``store_latents`` alone also counts, although the video is the same. The file is replaced without
a lock, so two concurrent shards can lose each other's new claims: that weakens the guard and
never refuses a valid run. A cache filled before the claims existed is trusted as it is."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import Field

from physverify.core.errors import UsageError
from physverify.core.log import get_logger
from physverify.core.models import FrozenModel
from physverify.datasets.local_video import NAME as LOCAL_VIDEO
from physverify.extraction.cache import StageCache
from physverify.finetune.records import CandidateRecord, read_record, write_record

CLAIMS_FILE: Final = "finetune_claims.json"
PURGE_FLAG: Final = "--purge-stale-cache"
SHOWN: Final = 5

log = get_logger("finetune.cache_claims")


class ClipClaim(FrozenModel):
    campaign: str
    config_hash: str


class CacheClaims(FrozenModel):
    generation: dict[str, str] = Field(default_factory=dict)
    clips: dict[str, ClipClaim] = Field(default_factory=dict)


@dataclass(frozen=True)
class Conflicts:
    """``campaigns`` claim another generation hash; ``clashes`` are clip ids of this run last
    extracted for another ``config_hash``; ``stale`` every clip id whose entries are unusable."""

    campaigns: tuple[str, ...]
    clashes: tuple[str, ...]
    stale: tuple[str, ...]


def conflicts(
    claims: CacheClaims, generation_hash: str, records: Sequence[CandidateRecord]
) -> Conflicts:
    campaigns = sorted(c for c, h in claims.generation.items() if h != generation_hash)
    clashes = sorted(
        {
            r.clip_id
            for r in records
            if r.clip_id in claims.clips and claims.clips[r.clip_id].config_hash != r.config_hash
        }
    )
    orphaned = {clip for clip, claim in claims.clips.items() if claim.campaign in campaigns}
    return Conflicts(tuple(campaigns), tuple(clashes), tuple(sorted(orphaned | set(clashes))))


def refusal(root: Path, found: Conflicts) -> str:
    causes = []
    if found.campaigns:
        causes.append(f"campaigns {list(found.campaigns)} claim other generation settings or seed")
    if found.clashes:
        shown = list(found.clashes[:SHOWN])
        causes.append(
            f"{len(found.clashes)} clip ids were extracted for another candidate "
            f"configuration (e.g. {shown})"
        )
    return (
        f"the stage cache {root} holds stale entries: {'; '.join(causes)}. Stage keys do not hash "
        f"the video, so those entries would be read for other videos: point paths.cache_dir to a "
        f"new directory, or pass {PURGE_FLAG} to drop the {len(found.stale)} clips' entries"
    )


def claimed(
    claims: CacheClaims,
    campaign: str,
    generation_hash: str,
    records: Sequence[CandidateRecord],
    dropped: Collection[str],
) -> CacheClaims:
    clips = {clip: claim for clip, claim in claims.clips.items() if clip not in dropped}
    clips.update(
        {r.clip_id: ClipClaim(campaign=campaign, config_hash=r.config_hash) for r in records}
    )
    generation = {c: h for c, h in claims.generation.items() if h == generation_hash}
    return CacheClaims(generation={**generation, campaign: generation_hash}, clips=clips)


@dataclass(frozen=True)
class CacheGuard:
    """Admits the candidates a campaign is about to extract into ``cache``."""

    cache: StageCache
    generation_hash: str
    purge: bool = False

    @property
    def path(self) -> Path:
        return self.cache.root / CLAIMS_FILE

    def admit(self, campaign: str, records: Sequence[CandidateRecord]) -> None:
        claims = read_record(self.path, CacheClaims) if self.path.is_file() else CacheClaims()
        found = conflicts(claims, self.generation_hash, records)
        if found.stale or found.campaigns:
            if not self.purge:
                raise UsageError(refusal(self.cache.root, found))
            entries = sum(self.cache.purge_clip(LOCAL_VIDEO, clip) for clip in found.stale)
            log.warning("purged %d cache entries of %d stale clips", entries, len(found.stale))
        new = claimed(claims, campaign, self.generation_hash, records, found.stale)
        write_record(self.path, new)
