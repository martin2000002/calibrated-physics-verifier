"""The store of a fine-tuning campaign under the data dir of the test settings."""

from __future__ import annotations

from physverify.config.models import Settings
from physverify.finetune.layout import CampaignStore


def campaign_store(settings: Settings, campaign: str) -> CampaignStore:
    return CampaignStore(
        settings.paths.data_dir / "finetune" / campaign,
        settings.paths.raw_dir / "local_video",
        campaign,
    )
