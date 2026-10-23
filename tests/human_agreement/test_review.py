"""The recorded hand review of the test captions."""

from __future__ import annotations

import pytest

from physverify.config.models import Settings
from physverify.core.errors import ConfigError, MissingInputError
from physverify.datasets._videophy_review import (
    KEEP,
    PENDING,
    REVIEWED,
    SubsetReview,
    load_review,
    record,
    subset_hash,
    template,
    write_review,
)
from physverify.datasets._videophy_subset import (
    audit,
    reviewed_candidates,
    sized_labels_or_none,
)


def test_candidates_apply_scope_and_the_lexical_rule(world) -> None:
    candidates = world.dataset.candidates("h2_test")
    v1 = candidates["v1_test"]
    assert {r.caption for r in v1} == {world.BALL, world.BOOK}
    assert len(v1) == 5
    assert {r.caption for r in candidates["v2_test"]} == {world.BOX}


def test_the_test_rows_are_refused_while_the_review_is_pending(world) -> None:
    write_review(world.dataset.subset_path, audit(world.dataset, "h2_test"))
    review = load_review(world.dataset.subset_path)
    assert review is not None and review.status == PENDING
    assert set(review.decisions["v1_test"].values()) == {None}
    with pytest.raises(MissingInputError, match="hand review"):
        world.dataset.rows("h2_test")


def test_dev_rows_need_no_review(world) -> None:
    assert {r.caption for r in world.dataset.rows("h2_dev")} == {world.CUBE, world.CAN}


def test_a_recorded_review_freezes_n_and_sha256(world) -> None:
    world.keep_all()
    review = load_review(world.dataset.subset_path)
    assert review is not None and review.status == REVIEWED
    rows = world.dataset.rows("h2_test")
    v1 = [r.clip_id for r in rows if r.part == "v1_test"]
    assert review.subsets["v1_test"].n == len(v1) == 5
    assert review.subsets["v1_test"].sha256 == subset_hash(v1)


def test_an_excluded_caption_leaves_the_subset(world) -> None:
    pending = audit(world.dataset, "h2_test")
    decisions = {p: dict.fromkeys(t, KEEP) for p, t in pending.decisions.items()}
    decisions["v1_test"][world.BOOK] = "ambiguous"
    filled = pending.model_copy(update={"decisions": decisions, "date": "d"})
    write_review(
        world.dataset.subset_path, record(filled, reviewed_candidates(world.dataset, "h2_test"))
    )
    assert {r.caption for r in world.dataset.rows("v1_test")} == {world.BALL}


def test_record_needs_a_date_and_every_decision(world) -> None:
    pending = audit(world.dataset, "h2_test")
    candidates = reviewed_candidates(world.dataset, "h2_test")
    with pytest.raises(MissingInputError, match="date"):
        record(pending, candidates)
    dated = pending.model_copy(update={"date": "d"})
    with pytest.raises(MissingInputError, match="without a decision"):
        record(dated, candidates)


def test_the_review_record_names_no_reviewer(world) -> None:
    assert "reviewed_by" not in SubsetReview.model_fields
    assert "reviewed_by" not in audit(world.dataset, "h2_test").model_dump()


def test_an_unknown_reason_is_refused(world) -> None:
    pending = audit(world.dataset, "h2_test")
    decisions = {p: dict.fromkeys(t, "looks_wrong") for p, t in pending.decisions.items()}
    filled = pending.model_copy(update={"decisions": decisions, "date": "d"})
    with pytest.raises(ConfigError, match="unknown exclusion reason"):
        record(filled, reviewed_candidates(world.dataset, "h2_test"))


def test_a_changed_subset_no_longer_reproduces_its_hash(world) -> None:
    world.keep_all()
    review = load_review(world.dataset.subset_path)
    assert review is not None
    subsets = dict(review.subsets)
    subsets["v1_test"] = subsets["v1_test"].model_copy(update={"n": 4})
    write_review(world.dataset.subset_path, review.model_copy(update={"subsets": subsets}))
    with pytest.raises(ConfigError, match="no longer reproduces"):
        world.dataset.rows("v1_test")


def test_a_review_of_other_captions_must_be_redone(world) -> None:
    pending = audit(world.dataset, "h2_test")
    decisions = {p: dict.fromkeys(t, KEEP) for p, t in pending.decisions.items()}
    decisions["v1_test"]["A caption that is no longer a candidate."] = KEEP
    filled = pending.model_copy(update={"decisions": decisions, "date": "d"})
    with pytest.raises(ConfigError, match="re-audit"):
        record(filled, reviewed_candidates(world.dataset, "h2_test"))


def test_a_new_template_keeps_earlier_decisions() -> None:
    first = template({"v1_test": ["a", "b"]}, 1, None)
    decided = first.model_copy(update={"decisions": {"v1_test": {"a": KEEP, "b": None}}})
    second = template({"v1_test": ["a", "b", "c"]}, 1, decided)
    assert second.decisions["v1_test"] == {"a": KEEP, "b": None, "c": None}
    assert second.status == PENDING


def test_a_review_with_another_filter_version_is_refused(world) -> None:
    world.keep_all()
    review = load_review(world.dataset.subset_path)
    assert review is not None
    write_review(world.dataset.subset_path, review.model_copy(update={"filter_version": 99}))
    with pytest.raises(ConfigError, match="filter_version"):
        world.dataset.rows("h2_test")


def test_the_sized_labels_come_from_the_injected_settings(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import physverify.sizing.model as sizing_model

    seen: list[Settings] = []

    def fake(given: Settings) -> frozenset[str]:
        seen.append(given)
        return frozenset({"ball"})

    monkeypatch.setattr(sizing_model, "sized_labels", fake)
    assert sized_labels_or_none(settings) == frozenset({"ball"})
    assert seen == [settings]
