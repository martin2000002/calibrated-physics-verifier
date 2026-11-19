"""Ad-hoc demo prompts and the fixed demo scenes: their own campaign store, the prompt records
built from a command line, and the inputs the scene reports need."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.datasets.local_video import NAME as LOCAL_VIDEO
from physverify.demo.settings import load_demo
from physverify.finetune.layout import FINETUNE_DIR, SELECTED_FILE, CampaignStore
from physverify.finetune.prompts import fixed_record
from physverify.finetune.records import DEMO_TAG, PromptRecord, read_record
from physverify.finetune.selection import Selection
from physverify.finetune.settings import FixedPrompt, QueryTemplate
from physverify.schema.io import stable_hash

ADHOC_PREFIX: Final = "adhoc-"
ADHOC_TAG: Final = "adhoc"
ADHOC_FAMILY: Final = "adhoc"
ID_HASH_CHARS: Final = 8
COUNT_SEPARATOR: Final = ":"


def adhoc_store(settings: Settings) -> CampaignStore:
    campaign = load_demo(settings).adhoc_campaign
    return CampaignStore(
        root=settings.paths.data_dir / FINETUNE_DIR / campaign,
        raw_root=settings.paths.raw_dir / LOCAL_VIDEO,
        campaign=campaign,
    )


def parse_query(text: str) -> QueryTemplate:
    """``phrase`` or ``phrase:N`` (N objects expected)."""
    phrase, separator, count = text.rpartition(COUNT_SEPARATOR)
    if not separator or not count.isdigit():
        return QueryTemplate(phrase=text.strip())
    return QueryTemplate(phrase=phrase.strip(), expected_count=int(count))


def adhoc_prompt(
    text: str, queries: Sequence[QueryTemplate], family: str, target_checks: Sequence[str]
) -> PromptRecord:
    fields = {
        "text": text,
        "queries": [query.model_dump() for query in queries],
        "family": family,
        "target_checks": list(target_checks),
    }
    fixed = FixedPrompt(
        id=f"{ADHOC_PREFIX}{stable_hash(fields)[:ID_HASH_CHARS]}",
        family=family,
        text=text,
        queries=list(queries),
        target_checks=list(target_checks),
    )
    return fixed_record(fixed).model_copy(update={"tags": [ADHOC_TAG]})


def tuned_tag(store: CampaignStore, run_id: str) -> str:
    path = store.select_dir(run_id) / SELECTED_FILE
    if not path.is_file():
        raise UsageError(f"{path} does not exist: run 'finetune select' or pass --adapter")
    selection = read_record(path, Selection)
    if selection.selected is None:
        raise UsageError(f"{path} selected no checkpoint; pass --adapter")
    return selection.selected.model_tag


def demo_prompts(store: CampaignStore) -> list[PromptRecord]:
    if not store.prompts_path.is_file():
        raise UsageError(f"{store.prompts_path} does not exist: run 'finetune prompts build'")
    prompts = [p for p in store.prompts() if DEMO_TAG in p.tags]
    if not prompts:
        raise UsageError(f"{store.prompts_path} has no prompts tagged {DEMO_TAG!r}")
    return prompts


def require_videos(
    store: CampaignStore, prompts: Sequence[PromptRecord], tags: Sequence[str]
) -> None:
    have = {
        (r.model_tag, r.prompt_id)
        for tag in tags
        for r in store.candidates(tag)
        if r.k == 0 and store.is_complete(r)
    }
    missing = [f"{p.prompt_id}/{t}" for p in prompts for t in tags if (t, p.prompt_id) not in have]
    if missing:
        raise UsageError(
            f"k = 0 videos missing: {missing}; run 'compare-generators generate' first"
        )
