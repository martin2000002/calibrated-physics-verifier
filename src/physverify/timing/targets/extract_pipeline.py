"""Per-clip cost of the extraction pipeline, read from its own stage events (no re-timing)."""

from physverify.timing.targets._extract_pipeline import ExtractPipeline

PLUGINS = (ExtractPipeline,)
