"""Injected dependencies of a diagnosis run (ADR 0007 §4.1)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.cases import SampledMetric
from physverify.diagnosis.chain import ChainRunner
from physverify.diagnosis.checks_port import CheckSuite
from physverify.diagnosis.settings import DiagnosisSettings
from physverify.extraction._base import StageProvider
from physverify.extraction.pipeline import PipelineDeps
from physverify.runtime.runs import RunContext


@dataclass(frozen=True)
class DiagnosisContext:
    settings: Settings
    diag: DiagnosisSettings
    resolve_dataset: Callable[[str], DatasetSource]
    providers: Mapping[str, type[StageProvider]]
    checks: Mapping[str, type[Check]]
    metrics: Mapping[str, SampledMetric]
    deps: PipelineDeps
    suite: CheckSuite | None
    clock: Callable[[], dt.datetime]
    run: RunContext

    def dataset(self, name: str | None = None) -> DatasetSource:
        """Instance of ``name`` (default: ``diagnosis.dataset``)."""
        return self.resolve_dataset(name or self.diag.dataset)

    def chain(self, dataset: DatasetSource) -> ChainRunner:
        return ChainRunner(self.settings, self.providers, self.checks, self.deps, dataset)
