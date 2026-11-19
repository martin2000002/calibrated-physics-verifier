"""``physverify baselines``: an external judge on the final run's sets, beside B1.

``score`` is heavy (the judge runs on the GPU) and looks at no label; ``compare`` reads cached
stage outputs and the scores, and is guarded by the sealed ledger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app import baselines as service
from physverify.cli import options as opt
from physverify.cli.context import app_context, echo_err, heavy_command

NAME = "baselines"

app = typer.Typer(
    help="External judges beside the verifier's B1 on the final run's sets.",
    no_args_is_help=True,
)
JudgeOption = Annotated[
    str, typer.Option(help="Key of [vlm_comparison.judges] (configs/studies/vlm_comparison.toml).")
]
ScoresOption = Annotated[Path, typer.Option(help="Directory of the judges' score caches.")]


@app.command("score", help="Score a dataset's clips with an external judge (cached, resumable).")
@heavy_command
def score(
    ctx: typer.Context,
    dataset: Annotated[str, typer.Option(help="Dataset to score.")],
    scores: ScoresOption,
    judge: JudgeOption = "autoeval",
    subset: Annotated[str | None, typer.Option(help="Default: [diagnosis].subset.")] = None,
    autoeval_root: Annotated[
        Path | None, typer.Option(help="AutoEval install root (default: [h3.autoeval] root).")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Rehearsal only: the N clips of smallest sha256(id).")
    ] = None,
) -> None:
    request = service.ScoreRequest(judge, dataset, subset, scores, autoeval_root, limit)
    service.score_clips(app_context(ctx), request, echo=typer.echo)


@app.command("compare", help="u_B1, then every looked-at clip's label, B1 and judge score.")
def compare(
    ctx: typer.Context,
    part: Annotated[str, typer.Option(help="kubric or real.")],
    artifact: Annotated[Path, typer.Option(help="The frozen calibration artifact.")],
    scores: ScoresOption,
    out: Annotated[Path, typer.Option(help="Directory of cuts.json and the part's rows.")],
    judge: JudgeOption = "autoeval",
    dataset: Annotated[str | None, typer.Option(help="The Kubric set (--part kubric).")] = None,
    subset: Annotated[str | None, typer.Option(help="Its subset (--part kubric).")] = None,
    real_dataset: opt.RealDatasetOption = None,
    clips: opt.ClipsOption = None,
) -> None:
    request = service.CompareRequest(
        part,
        artifact,
        dataset,
        subset,
        tuple(real_dataset or ()),
        scores,
        out,
        judge,
        tuple(clips or ()),
    )
    service.compare_part(
        app_context(ctx), f"{NAME} compare", request, echo=typer.echo, err=echo_err
    )
