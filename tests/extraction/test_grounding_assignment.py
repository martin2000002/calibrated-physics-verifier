"""Per-query Grounding DINO: candidate merge and slot assignment (ADR 0014 §2.2).

The f4/f5/f2 cases mirror what the GDINO-tiny probe saw on local ``eval`` frames; they are
development fixtures, not held-out evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from support import make_context, run_provider
from synthetic_clip import SyntheticClip

from physverify.extraction.providers import _grounding_per_query, detect_grounding_dino
from physverify.extraction.providers._grounding_per_query import (
    Candidates,
    PromptedParams,
    assign,
    merge_candidates,
)
from physverify.extraction.providers.detect_grounding_dino import (
    GroundingDinoDetector,
    Params,
    RawDetections,
    select,
)
from physverify.extraction.providers.detect_grounding_dino_base import (
    BASE_WEIGHTS,
    GroundingDinoBase,
)
from physverify.extraction.providers.detect_grounding_dino_tiny_per_query import (
    GroundingDinoTinyPerQuery,
)
from physverify.schema.clip import ObjectQuery
from physverify.schema.perception import Detections

BOX_0 = (10.0, 10.0, 30.0, 30.0)
BOX_1 = (40.0, 10.0, 60.0, 30.0)
BOX_2 = (70.0, 10.0, 90.0, 30.0)


def raw(
    boxes: Sequence[tuple[float, ...]], scores: Sequence[float], text: str = ""
) -> RawDetections:
    return RawDetections(
        boxes=np.array(boxes, dtype=np.float64).reshape(-1, 4),
        scores=np.array(scores, dtype=np.float64),
        texts=tuple(text for _ in scores),
    )


def candidates(scores: Sequence[Sequence[float]]) -> Candidates:
    boxes = np.array([BOX_0, BOX_1, BOX_2][: len(scores)], dtype=np.float64)
    return Candidates(boxes=boxes, scores=np.array(scores, dtype=np.float64))


def test_f4_assignment_keeps_all_three_where_the_joint_prompt_keeps_one() -> None:
    queries = (ObjectQuery("cube", 1), ObjectQuery("cylinder", 2))
    joint = raw([BOX_0, BOX_1, BOX_2], [0.6, 0.5, 0.45], "cube cylinder")
    assert len(select(joint, queries, Params()).query) == 1
    picked = assign(candidates([[0.6, 0.65], [0.4, 0.5], [0.0, 0.45]]), queries, PromptedParams())
    assert picked.query == (0, 1, 1)
    np.testing.assert_array_equal(picked.boxes, np.array([BOX_0, BOX_1, BOX_2]))


def test_f5_the_ball_box_goes_to_the_ball_not_the_unseen_cube() -> None:
    queries = (ObjectQuery("cube", 1), ObjectQuery("ball", 1))
    picked = assign(candidates([[0.3, 0.7]]), queries, PromptedParams())
    assert picked.query == (1,)
    np.testing.assert_allclose(picked.scores, [0.7])


def test_f2_distinct_objects_match_the_joint_selection() -> None:
    queries = (ObjectQuery("ball", 1), ObjectQuery("box", 1))
    joint = raw([BOX_0, BOX_1], [0.8, 0.7], "")
    joint = RawDetections(joint.boxes, joint.scores, ("ball", "box"))
    legacy = select(joint, queries, Params())
    picked = assign(candidates([[0.8, 0.0], [0.0, 0.7]]), queries, PromptedParams())
    assert picked.query == legacy.query
    np.testing.assert_array_equal(picked.boxes, legacy.boxes)


def test_unknown_count_takes_every_box_by_its_best_query() -> None:
    queries = (ObjectQuery("ball", None), ObjectQuery("box", 1))
    picked = assign(candidates([[0.9, 0.1], [0.8, 0.2], [0.3, 0.6]]), queries, PromptedParams())
    assert picked.query == (0, 0, 1)


def test_max_objects_keeps_the_best_scores_ordered_by_query() -> None:
    queries = (ObjectQuery("ball", None),)
    params = PromptedParams(max_objects=2)
    picked = assign(candidates([[0.5], [0.9], [0.7]]), queries, params)
    np.testing.assert_allclose(picked.scores, [0.9, 0.7])


def test_no_candidate_gives_an_empty_selection() -> None:
    empty = Candidates(boxes=np.zeros((0, 4)), scores=np.zeros((0, 2)))
    picked = assign(empty, (ObjectQuery("ball", 1), ObjectQuery("box", 1)), PromptedParams())
    assert picked.query == ()
    assert picked.boxes.shape == (0, 4)


def test_overlapping_boxes_from_two_prompts_merge_with_both_scores() -> None:
    near = (11.0, 10.0, 31.0, 30.0)
    merged = merge_candidates([raw([BOX_0], [0.6]), raw([near, BOX_1], [0.4, 0.5])], 0.7)
    np.testing.assert_array_equal(merged.boxes, np.array([BOX_0, BOX_1]))
    np.testing.assert_allclose(merged.scores, [[0.6, 0.4], [0.0, 0.5]])


def fake_detector(monkeypatch: pytest.MonkeyPatch, prompts: list[tuple[str, ...]]) -> list[Any]:
    loaded: list[Any] = []

    def load(ctx: Any, spec: Any = detect_grounding_dino.WEIGHTS) -> tuple[Any, Any]:
        loaded.append(spec)
        return object(), object()

    def detect(
        model: Any,
        processor: Any,
        frame: NDArray[np.uint8],
        queries: Sequence[ObjectQuery],
        params: Params,
        device: str,
    ) -> RawDetections:
        phrases = tuple(q.phrase for q in queries)
        prompts.append(phrases)
        if phrases == ("ball", "box"):
            return RawDetections(
                np.array([BOX_0, BOX_1], dtype=np.float64), np.array([0.8, 0.7]), ("ball", "box")
            )
        box = BOX_0 if phrases == ("ball",) else BOX_1
        return raw([box], [0.8 if phrases == ("ball",) else 0.7])

    monkeypatch.setattr(detect_grounding_dino, "load_detector", load)
    monkeypatch.setattr(detect_grounding_dino, "detect", detect)
    monkeypatch.setattr(detect_grounding_dino, "set_torch_threads", lambda threads: None)
    return loaded


def same(a: Detections, b: Detections) -> None:
    np.testing.assert_array_equal(a.boxes_xyxy, b.boxes_xyxy)
    np.testing.assert_array_equal(a.object_ids, b.object_ids)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.scores, b.scores)


def test_joint_mode_reproduces_the_original_detector(
    fixture_clip: SyntheticClip, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[tuple[str, ...]] = []
    fake_detector(monkeypatch, prompts)
    ctx = make_context(fixture_clip)
    legacy = run_provider(GroundingDinoDetector, ctx)
    joint = run_provider(GroundingDinoTinyPerQuery, ctx, prompt_mode="joint")
    assert isinstance(legacy, Detections) and isinstance(joint, Detections)
    same(legacy, joint)
    assert prompts == [("ball", "box"), ("ball", "box")]


def test_per_query_mode_prompts_each_query_once_per_frame(
    fixture_clip: SyntheticClip, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[tuple[str, ...]] = []
    fake_detector(monkeypatch, prompts)
    out = run_provider(GroundingDinoTinyPerQuery, make_context(fixture_clip))
    assert isinstance(out, Detections)
    assert prompts == [("ball",), ("box",)]
    assert out.labels.tolist() == ["ball", "box"]


def test_base_model_loads_its_own_weights(
    fixture_clip: SyntheticClip, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = fake_detector(monkeypatch, [])
    run_provider(GroundingDinoBase, make_context(fixture_clip))
    run_provider(GroundingDinoTinyPerQuery, make_context(fixture_clip))
    assert loaded == [BASE_WEIGHTS, detect_grounding_dino.WEIGHTS]
    assert GroundingDinoBase.weights == (BASE_WEIGHTS,)
    assert _grounding_per_query.PromptedDetector.Params is PromptedParams
