"""Option types and parsing of the subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.core.choices import choice
from physverify.core.errors import UsageError
from physverify.finetune.settings import SPLITS, PromptSplit
from physverify.finetune.stop_rule import STOP_RULE_DEVIATION
from physverify.generator_comparison.arms import EVAL_SPLIT

StopRuleDeviationOption = Annotated[
    str | None,
    typer.Option(
        STOP_RULE_DEVIATION,
        help="Go on although the pilot stop rule fired (recorded; H3 then has no verdict).",
    ),
]
ShardOption = Annotated[str | None, typer.Option(help="Shard i/n of the candidates.")]

SplitOption = Annotated[str, typer.Option(help="train or dev (eval belongs to `h3`).")]
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


def campaign_split(split: str, comparison_step: str) -> PromptSplit:
    """A train or dev split; the eval candidates and scores belong to the generator comparison,
    whose generation stamp and pre-registration a finetune run would break."""
    parsed = choice(split, SPLITS, "--split")
    if parsed == EVAL_SPLIT:
        raise UsageError(
            "the eval split belongs to H3: run "
            f"`physverify compare-generators {comparison_step} --arm base|ft`"
        )
    return parsed
