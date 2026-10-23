"""Hand review of the VideoPhy test subset (captions only, exclusions only).

The review sees captions only. The review can exclude a caption the lexical rule kept, with a
reason from a fixed list, never add one. ``record`` stores n and the sha256 of the clip ids, and
the dataset refuses a reviewed subset whose recomputed hash differs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from physverify.core.atomic import write_text_atomic
from physverify.core.errors import ConfigError, MissingInputError
from physverify.core.jsonio import read_model
from physverify.core.models import FrozenModel
from physverify.datasets._videophy_tables import VideophyRow
from physverify.schema.io import stable_hash

KEEP = "keep"
PENDING = "pending_review"
REVIEWED = "reviewed"
REVIEW_REASONS = ("human_agent", "nonrigid", "fracture", "fluid", "no_interaction", "ambiguous")
CRITERIA = (
    "Keep a caption only if every moving body in it is rigid and the event is an object-object "
    "interaction (fall, collision, rolling, sliding, toppling, resting contact) visible in the "
    "frame. Exclude with one reason: human_agent (a person, animal or body part acts or is "
    "involved), nonrigid (cloth, paper, rope, soft or elastic bodies), fracture (breaking, "
    "shattering, crushing, denting), fluid (liquids, granular matter, smoke), no_interaction "
    "(no contact event between objects), ambiguous (the caption does not fix what happens). "
    "Decide from the caption alone: labels, model outputs and verifier outputs are not shown. "
    "Tie-breaks: (1) agent: human_agent when a body part or a wielded tool (bat, hammer, cue, "
    "bar, a hand that strikes) takes part in the interaction; a passive with no instrument "
    "('is struck', 'is pushed', 'is rolled', 'is knocked over') is kept, the agent being out of "
    "frame. (2) spinning: spinning in place on a support with no other event is no_interaction "
    "(spinning coin, fidget spinner, globe, plate); it is kept when it ends in a fall or coming "
    "to rest (a coin landing on a face, a top falling over, a roulette wheel). (3) material: paper "
    "and cardboard are nonrigid; damaging the surface (a dart sticking in, a mark left in the "
    "wood) is fracture."
)
Status = Literal["pending_review", "reviewed"]


class SubsetRecord(FrozenModel):
    n: int | None = None
    sha256: str | None = None


class SubsetReview(FrozenModel):
    """``decisions[part][caption]`` is ``keep``, an exclusion reason, or None while pending."""

    status: Status
    date: str | None
    filter_version: int
    criteria: str
    reasons: tuple[str, ...]
    queries_status: Status
    decisions: dict[str, dict[str, str | None]]
    subsets: dict[str, SubsetRecord]


def subset_hash(clip_ids: Sequence[str]) -> str:
    return stable_hash(sorted(clip_ids))


def load_review(path: Path) -> SubsetReview | None:
    if not path.is_file():
        return None
    return read_model(SubsetReview, path)


def write_review(path: Path, review: SubsetReview) -> None:
    write_text_atomic(path, review.model_dump_json(indent=2) + "\n")


def template(
    candidates: Mapping[str, Sequence[str]], filter_version: int, previous: SubsetReview | None
) -> SubsetReview:
    """A pending review over the captions the lexical rule kept, keeping earlier decisions."""
    old = {} if previous is None else previous.decisions
    decisions = {
        part: {c: old.get(part, {}).get(c) for c in sorted(set(captions))}
        for part, captions in sorted(candidates.items())
    }
    return SubsetReview(
        status=PENDING,
        date=None,
        filter_version=filter_version,
        criteria=CRITERIA,
        reasons=REVIEW_REASONS,
        queries_status=PENDING if previous is None else previous.queries_status,
        decisions=decisions,
        subsets={part: SubsetRecord() for part in sorted(candidates)},
    )


def _decided(review: SubsetReview, part: str, rows: Sequence[VideophyRow]) -> list[VideophyRow]:
    """Rows whose caption the review kept; every caption needs a valid decision."""
    table = review.decisions.get(part)
    if table is None:
        raise MissingInputError(f"the review has no decisions for {part}")
    captions = {row.caption for row in rows}
    if set(table) != captions:
        raise ConfigError(f"{part}: the reviewed captions differ from the lexical rule's; re-audit")
    for caption, decision in table.items():
        if decision is None:
            raise MissingInputError(f"{part}: caption without a decision: {caption!r}")
        if decision != KEEP and decision not in review.reasons:
            raise ConfigError(f"{part}: unknown exclusion reason {decision!r}")
    return [row for row in rows if table[row.caption] == KEEP]


def require_recorded(review: SubsetReview | None, filter_version: int) -> SubsetReview:
    """The review, refused while pending or when it was made under another filter version; needs
    no label table, so a run can check it before downloading anything."""
    if review is None or review.status != REVIEWED:
        raise MissingInputError(
            f"the subset needs the hand review (status {PENDING}); run `physverify agreement "
            "subset --audit`, fill the decisions, then `--record`"
        )
    if review.filter_version != filter_version:
        raise ConfigError(
            f"the review used filter_version {review.filter_version}, not {filter_version}"
        )
    return review


def reviewed_rows(
    review: SubsetReview | None, part: str, rows: Sequence[VideophyRow], filter_version: int
) -> list[VideophyRow]:
    """The reviewed subset of ``part``, refused while pending or when its hash changed."""
    recorded = require_recorded(review, filter_version)
    kept = _decided(recorded, part, rows)
    stored = recorded.subsets.get(part, SubsetRecord())
    if stored.n != len(kept) or stored.sha256 != subset_hash([r.clip_id for r in kept]):
        raise ConfigError(f"{part}: the subset no longer reproduces its recorded n and sha256")
    return kept


def record(review: SubsetReview, rows: Mapping[str, Sequence[VideophyRow]]) -> SubsetReview:
    """Freeze n and sha256 per part once every caption is decided and the review is dated; the
    record names no reviewer."""
    if not review.date:
        raise MissingInputError("set the date in the review file before --record")
    subsets = {}
    for part, part_rows in sorted(rows.items()):
        kept = _decided(review, part, part_rows)
        subsets[part] = SubsetRecord(n=len(kept), sha256=subset_hash([r.clip_id for r in kept]))
    return review.model_copy(update={"status": REVIEWED, "subsets": subsets})
