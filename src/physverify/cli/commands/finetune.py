"""``physverify finetune``: preference pairs from the verifier and Flow-DPO LoRA (ADR 0012, 3.10).

Every step resumes: ``generate`` skips complete candidates, ``score`` the ones with a current
score (same scorer stamp, same candidate ``config_hash``), ``train`` starts from the last
checkpoint. The GPU subcommands exit 2 without ``--device cuda`` and a CUDA device,
before they write anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final

import typer

from physverify.app import finetune as campaign
from physverify.app import finetune_training as training
from physverify.cli import options as opt
from physverify.cli.context import app_context, echo_err, heavy_command
from physverify.core.errors import EXIT_ERROR, UsageError
from physverify.finetune.layout import CampaignStore
from physverify.finetune.records import BASE_TAG
from physverify.finetune.status import status_lines
from physverify.timing.finetune_budget import BudgetRequest, budget_report

NAME = "finetune"
DEFAULT_RUN: Final = "main"
PILOT_PROMPTS: Final = 20

app = typer.Typer(no_args_is_help=True, help="Preference pairs and Flow-DPO LoRA of Wan2.1.")
prompts_app = typer.Typer(no_args_is_help=True, help="The prompt set (prompts.jsonl).")
app.add_typer(prompts_app, name="prompts")


@prompts_app.command("build", help="Write prompts.jsonl (or with --check, verify it).")
def prompts_build(
    check: Annotated[bool, typer.Option(help="Only verify the existing file.")] = False,
) -> None:
    if not campaign.build_prompts(app_context(), check, echo=typer.echo, err=echo_err):
        raise typer.Exit(code=EXIT_ERROR)


@app.command("generate", help="Generate the candidates of a split (resumes).")
@heavy_command
def generate(
    split: opt.SplitOption,
    adapter: opt.AdapterOption = None,
    shard: opt.ShardOption = None,
    limit: opt.LimitOption = None,
    pilot: opt.PilotOption = None,
) -> None:
    which = opt.campaign_split(split, "generate")
    if pilot is not None and which != "train":
        raise UsageError("--pilot draws train prompts: use it with --split train")
    campaign.generate_split(app_context(), which, adapter, shard, limit, pilot, echo=typer.echo)


@app.command("score", help="Extract, verify and score the candidates of a model tag (resumes).")
@heavy_command
def score(
    split: opt.SplitOption,
    model_tag: Annotated[str, typer.Option(help="base or lora-<hash>.")] = BASE_TAG,
    scorer: opt.ScorerOption = None,
    shard: opt.ShardOption = None,
) -> None:
    which = opt.campaign_split(split, "score")
    campaign.score_split(app_context(), which, model_tag, scorer, shard, echo=typer.echo)


@app.command("pairs", help="Winner-loser pairs of the train prompts (val prompts kept apart).")
def pairs(scorer: opt.ScorerOption = None) -> None:
    campaign.write_pairs(app_context(), scorer, echo=typer.echo)


@app.command("pilot-review", help="Record the hand review of the pilot pairs (stop rule).")
def pilot_review(
    sample: Annotated[bool, typer.Option(help="Print the pairs to inspect.")] = False,
    better: opt.BetterOption = None,
    note: Annotated[str | None, typer.Option(help="Free text kept with the review.")] = None,
    scorer: opt.ScorerOption = None,
) -> None:
    if sample == (better is not None):
        raise UsageError("pass either --sample or --better N")
    for line in campaign.pilot_review_lines(app_context(), scorer, better, note):
        typer.echo(line)


@app.command("train", help="Flow-DPO LoRA on the stored pairs (resumes from the last checkpoint).")
@heavy_command
def train(
    scorer: opt.ScorerOption = None,
    run_id: opt.RunIdOption = DEFAULT_RUN,
    resume: opt.ResumeOption = True,
    force: opt.ForceOption = False,
    max_steps: opt.MaxStepsOption = None,
    beta: opt.BetaOption = None,
    allow_provisional: opt.ProvisionalOption = False,
    stop_rule_deviation: opt.StopRuleDeviationOption = None,
) -> None:
    request = training.TrainRequest(
        scorer, run_id, resume, force, max_steps, beta, allow_provisional, stop_rule_deviation
    )
    training.train_lora(app_context(), request, echo=typer.echo, err=echo_err)


@app.command("beta-grid", help="Short runs over finetune.train.beta_grid, chosen on val pairs.")
@heavy_command
def beta_grid(
    scorer: opt.ScorerOption = None,
    allow_provisional: opt.ProvisionalOption = False,
    stop_rule_deviation: opt.StopRuleDeviationOption = None,
) -> None:
    deviation = stop_rule_deviation
    training.beta_grid_runs(
        app_context(), scorer, allow_provisional, deviation, echo=typer.echo, err=echo_err
    )


@app.command("select", help="Score dev for spaced checkpoints against the base; pick one.")
@heavy_command
def select(scorer: opt.ScorerOption = None, run_id: opt.RunIdOption = DEFAULT_RUN) -> None:
    training.select_checkpoint(app_context(), scorer, run_id, echo=typer.echo, err=echo_err)


@app.command("budget", help="Training prompts that fit in a GPU budget (valid timing runs only).")
def budget(
    timing_run: Annotated[Path, typer.Option(help="Timing run directory with dpo_step.")],
    gpu_hours: Annotated[float, typer.Option(help="GPU hours available (H).")],
    pair_yield: opt.PairYieldOption = None,
    extract_batch: opt.ExtractBatchOption = None,
    generator_only: opt.GeneratorOnlyOption = False,
    pilot_prompts: Annotated[int, typer.Option(min=0, help="Pilot prompts still to run.")] = (
        PILOT_PROMPTS
    ),
    scorer: opt.ScorerOption = None,
) -> None:
    request = BudgetRequest(
        timing_run, gpu_hours, pair_yield, extract_batch, generator_only, pilot_prompts, scorer
    )
    for line in budget_report(app_context().settings, request):
        typer.echo(line)


@app.command("status", help="Candidates, scores, pairs and training steps of the campaign.")
def status() -> None:
    for line in status_lines(CampaignStore.from_settings(app_context().settings)):
        typer.echo(line)
