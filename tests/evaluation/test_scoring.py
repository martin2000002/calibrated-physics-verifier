from __future__ import annotations

from typing import Any

from physverify.finetune.records import BASE_TAG, write_record
from physverify.generator_comparison.campaign import eval_records, eval_scores
from physverify.generator_comparison.prereg import H3Preregistration
from physverify.generator_comparison.scoring import frozen_stamp_problem, manifest, rescore, stale

SCORER = "provisional"


def _age(campaign: Any, tag: str, n: int) -> None:
    """Gives ``n`` stored eval scores of ``tag`` an older stamp."""
    for score in eval_scores(campaign.store, tag, SCORER)[:n]:
        old = score.model_copy(update={"thresholds": "older-rule", "score": 0.0})
        write_record(campaign.store.score_path(tag, SCORER, score.candidate_id), old)


def test_stale_lists_only_scores_of_another_stamp(campaign: Any) -> None:
    campaign.build(eval_scored=True)
    _age(campaign, BASE_TAG, 2)
    found = stale(eval_scores(campaign.store, BASE_TAG, SCORER), campaign.scorer)
    assert len(found) == 2
    assert {s.thresholds for s in found} == {"older-rule"}


def test_rescore_restamps_stale_scores_from_their_stored_check_results(campaign: Any) -> None:
    campaign.build(eval_scored=True)
    before = {s.candidate_id: s.score for s in eval_scores(campaign.store, BASE_TAG, SCORER)}
    _age(campaign, BASE_TAG, 3)
    assert rescore(campaign.store, BASE_TAG, campaign.scorer) == 3
    after = eval_scores(campaign.store, BASE_TAG, SCORER)
    assert not stale(after, campaign.scorer)
    assert {s.candidate_id: s.score for s in after} == before
    assert rescore(campaign.store, BASE_TAG, campaign.scorer) == 0


def test_the_manifest_binds_the_scores_to_a_freeze(campaign: Any) -> None:
    campaign.build(eval_scored=True)
    record = manifest(campaign.store, BASE_TAG, campaign.scorer, "checks", "a" * 64)
    assert record.n == len(eval_records(campaign.store, BASE_TAG))
    assert (record.stamp, record.prereg_sha256) == (campaign.scorer.thresholds(), "a" * 64)
    campaign.write_candidate(BASE_TAG, campaign.store.prompts()[0], 99)
    moved = manifest(campaign.store, BASE_TAG, campaign.scorer, "checks", "a" * 64)
    assert moved.candidates_hash != record.candidates_hash


def test_the_frozen_stamp_is_the_scorer_version_and_label(campaign: Any) -> None:
    scorer = campaign.scorer

    def frozen(version: str, label: str) -> H3Preregistration:
        return H3Preregistration.model_construct(scorer_version=version, rule_label=label)

    assert frozen_stamp_problem(scorer, frozen(scorer.version, scorer.thresholds())) is None
    for other in (frozen("0", scorer.thresholds()), frozen(scorer.version, "older-rule")):
        problem = frozen_stamp_problem(scorer, other)
        assert problem is not None and "refreeze" in problem
