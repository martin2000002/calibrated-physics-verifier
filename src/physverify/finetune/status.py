"""The ``finetune status`` lines: candidates, scores, pairs and training steps of a campaign."""

from __future__ import annotations

import json

from physverify.finetune.layout import REPORT_FILE, SELECTED_FILE, CampaignStore
from physverify.finetune.pairs_report import yields
from physverify.finetune.records import PairsReport, read_record
from physverify.finetune.schedule import checkpoints
from physverify.finetune.settings import SPLITS


def status_lines(store: CampaignStore) -> list[str]:
    if not store.prompts_path.is_file():
        return [f"campaign {store.campaign}: no prompts.jsonl yet"]
    prompts = store.prompts()
    lines = [f"campaign {store.campaign}: {len(prompts)} prompts at {store.root}"]
    for tag in store.model_tags():
        records = store.candidates(tag)
        by_split = {s: sum(r.split == s for r in records) for s in SPLITS}
        lines.append(f"  {tag}: candidates {json.dumps(by_split)}")
        scores_root = store.root / "scores" / tag
        for directory in sorted(scores_root.iterdir()) if scores_root.is_dir() else []:
            scores = store.scores(tag, directory.name)
            abstained = sum(s.score is None for s in scores)
            lines.append(f"    {directory.name}: {len(scores)} scored, {abstained} abstained")
    pairs_root = store.root / "pairs"
    for directory in sorted(pairs_root.iterdir()) if pairs_root.is_dir() else []:
        report = directory / REPORT_FILE
        if report.is_file():
            parsed = read_record(report, PairsReport)
            lines.append(f"  pairs/{directory.name}: {parsed.pairs} ({yields(parsed)})")
    for run in store.train_runs():
        found = checkpoints(store.train_dir(run))
        last = found[-1][0] if found else 0
        selected = store.select_dir(run) / SELECTED_FILE
        mark = ", selected" if selected.is_file() else ""
        lines.append(f"  train/{run}: last checkpoint step {last}{mark}")
    return lines
