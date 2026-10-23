"""Option types of the subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app.finetune import STOP_RULE_DEVIATION

StopRuleDeviationOption = Annotated[
    str | None,
    typer.Option(
        STOP_RULE_DEVIATION,
        help="Go on although the pilot stop rule fired "
        "(recorded; the generator comparison then has no verdict).",
    ),
]
ShardOption = Annotated[str | None, typer.Option(help="Shard i/n of the candidates.")]

SplitOption = Annotated[
    str, typer.Option(help="train or dev (eval belongs to `compare-generators`).")
]
ScorerOption = Annotated[
    str | None, typer.Option(help="calibrated or provisional (default: finetune.scorer).")
]
RunIdOption = Annotated[str, typer.Option(help="Training run id.")]
ProvisionalOption = Annotated[bool, typer.Option(help="Train on pairs of an uncalibrated scorer.")]
AdapterOption = Annotated[Path | None, typer.Option(help="LoRA adapter directory.")]
LimitOption = Annotated[int | None, typer.Option(min=1, help="At most N candidates.")]
PilotOption = Annotated[
    int | None,
    typer.Option(min=1, help="Only N fit prompts of train, spread across the families."),
]
ResumeOption = Annotated[bool, typer.Option(help="Resume from the last checkpoint.")]
ForceOption = Annotated[
    bool, typer.Option(help="With --no-resume, delete the run's existing checkpoints.")
]
MaxStepsOption = Annotated[int | None, typer.Option(min=1, help="Stop after N steps.")]
BetaOption = Annotated[float | None, typer.Option(help="DPO beta (default: beta-grid or config).")]
BetterOption = Annotated[
    int | None, typer.Option(min=0, help="How many of them show a better winner.")
]
PairYieldOption = Annotated[
    float | None, typer.Option(help="Pairs per prompt (default: measured in report.json).")
]
ExtractBatchOption = Annotated[int | None, typer.Option(help="Clips per extraction process (B_x).")]
GeneratorOnlyOption = Annotated[
    bool, typer.Option(help="Cost = generation only (extraction runs elsewhere).")
]
