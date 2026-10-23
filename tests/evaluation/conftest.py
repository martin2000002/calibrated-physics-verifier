"""Fixtures of H3: settings over ``configs/`` and a synthetic campaign on disk.

The campaign has 8 analysed eval prompts (two families, four templates, half unseen), one demo
prompt and two dev prompts; base has K = 8 eval candidates, the selected adapter K = 4, both K = 2
on dev. Scores come from the provisional scorer over stored check results, so ``rescorer`` works.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from wiring import build_scorer

from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.finetune.candidates import adapter_tag
from physverify.finetune.layout import SELECTED_FILE, CampaignStore
from physverify.finetune.records import (
    BASE_TAG,
    DEMO_TAG,
    UNSEEN_TAG,
    CandidateRecord,
    CandidateScore,
    PromptRecord,
    write_record,
    write_records,
)
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.seeds import candidate_id
from physverify.finetune.selection import CheckpointResult, Selection
from physverify.finetune.settings import load_finetune
from physverify.generator_comparison.layout import H3Layout
from physverify.generator_comparison.settings import H3Settings, load_h3
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.schema.io import write_jsonl

SCORER = "provisional"
CHECKS = ("R1", "R2")
REFERENCE_FILE = "videophy2_autoeval_scores.csv"
REFERENCE_HEADER = (
    "video_url",
    "caption",
    "model_name",
    "sa",
    "pc",
    "joint",
    "sa_pred",
    "pc_pred",
    "joint_pred",
)
# Generator means: PC 3.6 and 3.86 (margin 0.13), SA 3.0 and 3.1 (margin 0.05).
REFERENCE_ROWS = (
    ("A", "3.6", "3.0"),
    ("A", "3.6", "3.0"),
    ("B", "3.86", "3.1"),
    ("B", "3.86", "3.1"),
)


@pytest.fixture
def h3_environ(environ: dict[str, str], tmp_path: Path) -> dict[str, str]:
    return {
        **environ,
        "PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data"),
        "PHYSVERIFY__FINETUNE__SCORER": SCORER,
        "PHYSVERIFY__H3__MARGIN_MIN_GENERATOR_N": "2",
        "PHYSVERIFY__H3__BOOTSTRAP_N": "200",
    }


@pytest.fixture
def h3_settings(config_dir: Path, h3_environ: dict[str, str]) -> Settings:
    return load_settings(config_dir, environ=h3_environ)


@pytest.fixture
def h3_cfg(h3_settings: Settings) -> H3Settings:
    return load_h3(h3_settings)


@pytest.fixture
def make_score() -> Callable[..., CandidateScore]:
    def build(candidate_id: str, score: float | None, **update: Any) -> CandidateScore:
        fields: dict[str, Any] = {
            "scorer": SCORER,
            "scorer_version": "2",
            "calibrated": False,
            "thresholds": "provisional",
            "candidate_id": candidate_id,
            "prompt_id": candidate_id.split("__")[0],
            "model_tag": BASE_TAG,
            "score": score,
            "sigma": None,
            "abstained": score is None,
            "reason": None,
            "n_decided": 2,
            "n_failed": 0,
            "coverage": 1.0,
            "decided": list(CHECKS),
            "per_check": {},
            "motion": 2.0,
            "objects_ok": True,
            "results_path": "",
        }
        return CandidateScore(**{**fields, **update})

    return build


def prompt(prompt_id: str, split: str, index: int, tags: Sequence[str] = ()) -> PromptRecord:
    return PromptRecord.model_validate(
        {
            "prompt_id": prompt_id,
            "text": f"a ball falls onto a table ({prompt_id})",
            "queries": ["ball", "table"],
            "family": f"f{index % 2}",
            "template_id": f"t{index % 4}",
            "target_checks": list(CHECKS),
            "split": split,
            "tags": list(tags),
        }
    )


@pytest.fixture
def make_prompt() -> Callable[..., PromptRecord]:
    """``make_prompt(id, split, index, tags)``: family f(index % 2), template t(index % 4)."""
    return prompt


def campaign_prompts() -> list[PromptRecord]:
    analysed = [prompt(f"e{i}", "eval", i, [UNSEEN_TAG] if i % 2 else []) for i in range(8)]
    demo = prompt("demo0", "eval", 0, [DEMO_TAG])
    dev = [prompt(f"d{i}", "dev", i) for i in range(2)]
    return [*analysed, demo, *dev]


def check_results(cid: str, magnitudes: Sequence[float]) -> list[CheckResult]:
    return [
        CheckResult(
            check_id=check,
            clip_id=cid,
            subject=("o0",),
            frame_span=None,
            magnitude=magnitude,
            sigma=0.01,
            unit="",
            verdict=Verdict.PASS if magnitude <= 1.0 else Verdict.FAIL,
            input_level=InputLevel.PERCEIVED,
            threshold=1.0,
            abstain_reason=None,
        )
        for check, magnitude in zip(CHECKS, magnitudes, strict=True)
    ]


Magnitudes = Callable[[str, str, int], tuple[float, float]]


def default_magnitudes(tag: str, prompt_id: str, k: int) -> tuple[float, float]:
    """The fine-tuned candidates violate less than the base ones, with some spread."""
    spread = 0.05 * (sum(map(ord, prompt_id)) % 5 + k)
    base = 0.2 if tag != BASE_TAG else 0.6
    return base + spread, base + 2 * spread


@dataclass
class Campaign:
    settings: Settings
    store: CampaignStore
    layout: H3Layout
    adapter: Path
    ft_tag: str
    scorer: AggregateScorer

    def write_prompts(self) -> list[PromptRecord]:
        prompts = campaign_prompts()
        write_records(self.store.prompts_path, prompts)
        return prompts

    def write_candidate(self, tag: str, p: PromptRecord, k: int) -> CandidateRecord:
        cid = candidate_id(p.prompt_id, k)
        video = self.store.video_path(tag, p.prompt_id, k)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
        record = CandidateRecord(
            candidate_id=cid,
            prompt_id=p.prompt_id,
            split=p.split,
            k=k,
            seed=1000 + k,
            model_tag=tag,
            video_path=str(video),
            latents_path=None,
            embeds_path="",
            config_hash=f"cfg-{tag}-{cid}",
        )
        write_record(self.store.meta_path(tag, p.prompt_id, k), record)
        return record

    def write_score(
        self, record: CandidateRecord, magnitudes: tuple[float, float]
    ) -> CandidateScore:
        tag, cid = record.model_tag, record.candidate_id
        results_path = self.store.checks_path(tag, SCORER, cid)
        results = check_results(cid, magnitudes)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(results_path, results)
        aggregate = self.scorer.score(results)
        score = CandidateScore(
            scorer=self.scorer.name,
            scorer_version=self.scorer.version,
            calibrated=self.scorer.calibrated,
            thresholds=self.scorer.thresholds(),
            candidate_id=cid,
            prompt_id=record.prompt_id,
            model_tag=tag,
            score=aggregate.score,
            sigma=aggregate.sigma,
            abstained=aggregate.abstained,
            reason=aggregate.reason,
            n_decided=aggregate.n_decided,
            n_failed=aggregate.n_failed,
            coverage=aggregate.coverage,
            decided=sorted(aggregate.decided),
            per_check={},
            motion=2.0,
            objects_ok=True,
            results_path=str(results_path),
        )
        write_record(self.store.score_path(tag, SCORER, cid), score)
        return score

    def write_selection(self) -> None:
        chosen = CheckpointResult(
            step=100,
            adapter=str(self.adapter),
            model_tag=self.ft_tag,
            n=4,
            n_paired=4,
            mean_score=0.7,
            paired_diff=0.1,
            ci=(0.05, 0.15),
            abstention=0.0,
            base_abstention=0.0,
            motion=2.0,
            base_motion=2.0,
            suspected_hacking=False,
        )
        selection = Selection(
            run_id="main",
            scorer=SCORER,
            thresholds=self.scorer.thresholds(),
            selected=chosen,
            checkpoints=[chosen],
        )
        write_record(self.store.select_dir("main") / SELECTED_FILE, selection)

    def write_reference(self) -> Path:
        path = self.settings.paths.raw_dir / "videophy" / "labels" / REFERENCE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(REFERENCE_HEADER)
            for i, (name, pc, sa) in enumerate(REFERENCE_ROWS):
                writer.writerow([f"https://x/{name}/{i}.mp4", "c", name, sa, pc, "1", sa, pc, "1"])
        return path

    def k_of(self, tag: str, split: str) -> int:
        cand = load_finetune(self.settings).candidates
        if split == "dev":
            return cand.k_dev
        return cand.k_eval if tag == BASE_TAG else cand.k_eval_finetuned

    def write_split(
        self, split: str, scored: bool, magnitudes: Magnitudes = default_magnitudes
    ) -> None:
        """Candidates of both arms on ``split`` (demo prompts with k = 0 only)."""
        for tag in (BASE_TAG, self.ft_tag):
            for p in self.store.prompts():
                if p.split != split:
                    continue
                ks = range(1) if DEMO_TAG in p.tags else range(self.k_of(tag, split))
                for k in ks:
                    record = self.write_candidate(tag, p, k)
                    if scored:
                        self.write_score(record, magnitudes(tag, p.prompt_id, k))

    def build(self, eval_scored: bool = False) -> None:
        """Prompts, selection, reference, scored dev candidates and the eval candidates."""
        self.write_prompts()
        self.write_selection()
        self.write_reference()
        self.write_split("dev", scored=True)
        self.write_split("eval", scored=eval_scored)


@pytest.fixture
def campaign(h3_settings: Settings, tmp_path: Path) -> Campaign:
    adapter = tmp_path / "adapters" / "step-100"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"lora weights")
    store = CampaignStore.from_settings(h3_settings)
    return Campaign(
        settings=h3_settings,
        store=store,
        layout=H3Layout.of(store.root),
        adapter=adapter,
        ft_tag=adapter_tag(adapter),
        scorer=build_scorer(SCORER, h3_settings),
    )
