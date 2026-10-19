"""On-disk layout of one preference campaign (ADR 0012, 3.3); paths only, no policy.

```
<data_dir>/finetune/<campaign>/
  prompts.jsonl
  embeds/<prompt_id>.safetensors
  candidates/<model_tag>/<prompt_id>/kNN.latents.safetensors, kNN.meta.json
  scores/<model_tag>/<scorer>/<candidate_id>.json, checks/<candidate_id>.jsonl, scores.jsonl,
    calibration/status.json
  pairs/<scorer>/pairs.jsonl, report.json, val_pairs.jsonl, val_report.json, pilot_review.json,
    stop_rule.json
  train/<run_id>/step-<n>/, metrics.csv, val.csv, run.json
  train/beta_grid.json
  select/<run_id>/selected.json
<raw_dir>/local_video/<campaign>__<model_tag>/<candidate_id>.mp4, manifest.json
```

The videos live in a ``local_video`` subset, so extraction and verification read them as any
other dataset. The mp4 is named by the candidate id, but the manifest gives each video the
record's ``clip_id`` (candidate id plus a ``config_hash`` prefix), so the extraction cache of a
candidate is never reused by another model tag, generation configuration or imported video. Two
campaigns with identical settings share it, since ``config_hash`` does not include the campaign. A
candidate is complete only once its meta exists: the mp4 and the latents are written first, the
meta last, each atomically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from physverify.config.models import Settings
from physverify.datasets.local_video import MANIFEST_FILE, ClipEntry, LocalManifest
from physverify.finetune.records import (
    CandidateRecord,
    CandidateScore,
    PromptRecord,
    read_record,
    read_records,
)
from physverify.finetune.seeds import candidate_id
from physverify.finetune.settings import load_finetune
from physverify.synth.store import atomic_write_bytes

FINETUNE_DIR: Final = "finetune"
LOCAL_VIDEO: Final = "local_video"
PROMPTS_FILE: Final = "prompts.jsonl"
SCORES_FILE: Final = "scores.jsonl"
PAIRS_FILE: Final = "pairs.jsonl"
REPORT_FILE: Final = "report.json"
VAL_PAIRS_FILE: Final = "val_pairs.jsonl"
VAL_REPORT_FILE: Final = "val_report.json"
PILOT_REVIEW_FILE: Final = "pilot_review.json"
STOP_RULE_FILE: Final = "stop_rule.json"
CALIBRATION_STATUS: Final = ("calibration", "status.json")
RUN_FILE: Final = "run.json"
BETA_FILE: Final = "beta_grid.json"
SELECTED_FILE: Final = "selected.json"
META_SUFFIX: Final = ".meta.json"
LATENTS_SUFFIX: Final = ".latents.safetensors"
EMBEDS_SUFFIX: Final = ".safetensors"
VIDEO_SUFFIX: Final = ".mp4"


@dataclass(frozen=True)
class CampaignStore:
    root: Path
    raw_root: Path
    campaign: str

    @classmethod
    def from_settings(cls, settings: Settings) -> CampaignStore:
        campaign = load_finetune(settings).campaign
        return cls(
            root=settings.paths.data_dir / FINETUNE_DIR / campaign,
            raw_root=settings.paths.raw_dir / LOCAL_VIDEO,
            campaign=campaign,
        )

    @property
    def prompts_path(self) -> Path:
        return self.root / PROMPTS_FILE

    def prompts(self) -> list[PromptRecord]:
        return read_records(self.prompts_path, PromptRecord)

    def embeds_path(self, prompt_id: str) -> Path:
        return self.root / "embeds" / f"{prompt_id}{EMBEDS_SUFFIX}"

    def subset(self, model_tag: str) -> str:
        return f"{self.campaign}__{model_tag}"

    def video_dir(self, model_tag: str) -> Path:
        return self.raw_root / self.subset(model_tag)

    def video_path(self, model_tag: str, prompt_id: str, k: int) -> Path:
        return self.video_dir(model_tag) / f"{candidate_id(prompt_id, k)}{VIDEO_SUFFIX}"

    def candidate_dir(self, model_tag: str, prompt_id: str) -> Path:
        return self.root / "candidates" / model_tag / prompt_id

    def meta_path(self, model_tag: str, prompt_id: str, k: int) -> Path:
        return self.candidate_dir(model_tag, prompt_id) / f"k{k:02d}{META_SUFFIX}"

    def latents_path(self, model_tag: str, prompt_id: str, k: int) -> Path:
        return self.candidate_dir(model_tag, prompt_id) / f"k{k:02d}{LATENTS_SUFFIX}"

    def candidates(self, model_tag: str) -> list[CandidateRecord]:
        directory = self.root / "candidates" / model_tag
        if not directory.is_dir():
            return []
        paths = sorted(directory.glob(f"*/*{META_SUFFIX}"))
        return [read_record(path, CandidateRecord) for path in paths]

    def model_tags(self) -> list[str]:
        directory = self.root / "candidates"
        return (
            sorted(p.name for p in directory.iterdir() if p.is_dir()) if directory.is_dir() else []
        )

    def is_complete(self, record: CandidateRecord) -> bool:
        meta = self.meta_path(record.model_tag, record.prompt_id, record.k)
        if not meta.is_file() or not Path(record.video_path).is_file():
            return False
        return read_record(meta, CandidateRecord).config_hash == record.config_hash

    def write_manifest(self, model_tag: str) -> Path:
        """The ``local_video`` manifest of every complete candidate of ``model_tag``."""
        records = sorted(self.candidates(model_tag), key=lambda r: r.candidate_id)
        prompts = {p.prompt_id: p for p in self.prompts()} if self.prompts_path.is_file() else {}
        entries = [
            ClipEntry(
                video=Path(record.video_path).name,
                clip_id=record.clip_id,
                queries=prompts[record.prompt_id].queries if record.prompt_id in prompts else None,
            )
            for record in records
            if Path(record.video_path).is_file()
        ]
        path = self.video_dir(model_tag) / MANIFEST_FILE
        payload = LocalManifest(clips=entries).model_dump(mode="json", exclude_none=True)
        atomic_write_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))
        return path

    def score_dir(self, model_tag: str, scorer: str) -> Path:
        return self.root / "scores" / model_tag / scorer

    def score_path(self, model_tag: str, scorer: str, cid: str) -> Path:
        return self.score_dir(model_tag, scorer) / f"{cid}.json"

    def checks_path(self, model_tag: str, scorer: str, cid: str) -> Path:
        return self.score_dir(model_tag, scorer) / "checks" / f"{cid}.jsonl"

    def calibration_status_path(self, model_tag: str, scorer: str) -> Path:
        """In a subdirectory, so ``scores`` never reads it as a candidate's score."""
        return self.score_dir(model_tag, scorer).joinpath(*CALIBRATION_STATUS)

    def scores_path(self, model_tag: str, scorer: str) -> Path:
        return self.score_dir(model_tag, scorer) / SCORES_FILE

    def scores(self, model_tag: str, scorer: str) -> list[CandidateScore]:
        directory = self.score_dir(model_tag, scorer)
        if not directory.is_dir():
            return []
        return [read_record(p, CandidateScore) for p in sorted(directory.glob("*.json"))]

    def pairs_dir(self, scorer: str) -> Path:
        return self.root / "pairs" / scorer

    def train_dir(self, run_id: str) -> Path:
        return self.root / "train" / run_id

    def beta_path(self) -> Path:
        return self.root / "train" / BETA_FILE

    def train_runs(self) -> list[str]:
        directory = self.root / "train"
        return (
            sorted(p.name for p in directory.iterdir() if p.is_dir()) if directory.is_dir() else []
        )

    def select_dir(self, run_id: str) -> Path:
        return self.root / "select" / run_id
