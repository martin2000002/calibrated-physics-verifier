"""Per-stage metric contract; registered by ``<stage>/<name>``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Protocol, runtime_checkable

from physverify.config.plugin_context import PluginKind
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage


@runtime_checkable
class StageMetric(Protocol):
    stage: ClassVar[Stage]
    name: ClassVar[str]

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]: ...


def metric_key(metric: type[StageMetric]) -> str:
    return f"{metric.stage.value}/{metric.name}"


METRICS = PluginKind[type[StageMetric]]("metrics")
