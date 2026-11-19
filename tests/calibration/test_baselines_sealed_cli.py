"""``baselines compare`` under the sealed ledger: it looks under the freeze of
``calibrate evaluate``, counts its own lines, and writes u_B1 before it reads a looked-at clip."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from calibration_cases import CHECKS, config, fit_meta, synthetic_level, synthetic_report
from cli_runner import CliResult, CliRunner
from repo_configs import CALIBRATION_FILE, declare
from steps_sh import steps_read
from test_sealed_cli import DATASET, FAST, REPORT, UNSEAL, _ReportRunner
from test_sealed_cli import artifact_text as artifact_text
from test_sealed_cli import checkout as checkout
from test_sealed_cli import evaluate_cli as evaluate_cli
from test_sealed_cli import filed as filed
from test_sealed_cli import fixture_dataset as fixture_dataset
from test_sealed_cli import ledger as ledger
from test_sealed_cli import reached as reached
from test_sealed_cli import sealed_in_files as sealed_in_files
from test_sealed_cli import version as version

from physverify.app import baselines as app_baselines
from physverify.app import calibration as app_calibration
from physverify.baselines.records import CUTS_FILE, PartRows
from physverify.baselines.score_cache import CachedScore, score_file
from physverify.calibration.artifact_io import default_artifact_path
from physverify.calibration.fit import FitInputs, fit_calibration
from physverify.calibration.sealed_ledger import read_entries
from physverify.config.models import Settings
from physverify.core.jsonio import read_model
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split

VLM_FILE = "studies/vlm_comparison.toml"
Compare = Callable[..., CliResult]


@pytest.fixture(autouse=True)
def compared_set(config_dir: Path) -> None:
    """The fixture dataset as the calibration set and a Kubric set of the comparison, declared
    in the files ``checkout`` commits."""
    declare(config_dir, CALIBRATION_FILE, "dataset", f'"{DATASET}"')
    path = config_dir / VLM_FILE
    table = f'\n[vlm_comparison.sets.{DATASET}]\npart = "kubric"\ncaption = "Objects move."\n'
    path.write_text(path.read_text(encoding="utf-8") + table, encoding="utf-8")


@pytest.fixture(scope="module")
def two_strata() -> str:
    levels = tuple(
        synthetic_level(synthetic_report(30, level=level), level)
        for level in (InputLevel.ORACLE, InputLevel.PERCEIVED)
    )
    return fit_calibration(FitInputs(levels, CHECKS, fit_meta()), config(bootstrap=50)).to_json()


@pytest.fixture
def frozen(settings: Settings, two_strata: str) -> Path:
    path = default_artifact_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(two_strata, encoding="utf-8")
    return path


@pytest.fixture
def scores(tmp_path: Path) -> Path:
    root = tmp_path / "scores"
    path = score_file(root, "autoeval", DATASET)
    path.parent.mkdir(parents=True)
    lines = (
        CachedScore(
            clip_id=c.clip_id, video_sha256="v", value=3.0, scale=(1, 5), stamp="s"
        ).model_dump_json()
        for c in REPORT.clips
    )
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return root


@pytest.fixture
def compare(cli: CliRunner, frozen: Path, scores: Path, tmp_path: Path) -> Compare:
    def invoke(*extra: str, artifact: Path = frozen, env: dict[str, str] = UNSEAL) -> CliResult:
        command = ("baselines", "compare", "--part", "kubric", "--dataset", DATASET)
        paths = (
            "--artifact",
            str(artifact),
            "--scores",
            str(scores),
            "--out",
            str(tmp_path / "vlm"),
        )
        return cli(*FAST, *command, *paths, *extra, env=env)

    return invoke


def test_compare_looks_under_the_freeze_of_evaluate(
    evaluate_cli: Callable[..., CliResult],
    compare: Compare,
    frozen: Path,
    ledger: Path,
    tmp_path: Path,
) -> None:
    evaluated = evaluate_cli(artifact=frozen, env=UNSEAL)
    assert evaluated.code == 0, evaluated.err
    for looks in (1, 2):
        result = compare()
        assert result.code == 0, result.err
        part = read_model(PartRows, tmp_path / "vlm" / "kubric.json")
        assert part.looks == looks
    first, *compared = read_entries(ledger)
    assert first.command == "calibrate evaluate"
    assert [e.command for e in compared] == ["baselines compare"] * 2
    assert all(e.freeze == first.freeze and not e.repeat for e in compared)
    assert all(e.artifacts == first.artifacts for e in compared)
    assert part.freeze == first.freeze
    assert {r.clip_id for s in part.sets for r in s.rows} == {c.clip_id for c in REPORT.clips}
    assert "cuts: " in result.out and "comparison: " in result.out


@pytest.mark.usefixtures("evaluate_cli")
def test_the_runners_read_the_paths_compare_and_its_table_print(
    cli: CliRunner, compare: Compare, tmp_path: Path
) -> None:
    compared = compare()
    assert compared.code == 0, compared.err
    rows = _printed(tmp_path, "comparison", compared.out)
    assert rows == tmp_path / "vlm" / "kubric.json"
    out = tmp_path / "table"
    table = cli(*FAST, "final-table", "comparison", "--out", str(out), "--kubric", str(rows))
    assert table.code == 0, table.err
    assert _printed(tmp_path, "comparison table", table.out) == out / "comparison_table.md"


@pytest.mark.usefixtures("evaluate_cli")
def test_u_b1_is_written_before_a_looked_at_clip_is_read(
    compare: Compare, ledger: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuts = tmp_path / "vlm" / CUTS_FILE
    runs: list[tuple[int, bool, bool]] = []
    reads: list[bool] = []

    class Spy(_ReportRunner):
        def run(self, source: object, clips: Sequence[ClipManifest]) -> Any:
            sealed = any(c.split is Split.EVAL for c in clips)
            runs.append((len(read_entries(ledger)), cuts.is_file(), sealed))
            return super().run(source, clips)

    real_read = app_baselines.read_scores

    def read_scores(path: Path) -> Any:
        reads.append(cuts.is_file())
        return real_read(path)

    monkeypatch.setattr(app_calibration, "_raw_runner", lambda context: Spy())
    monkeypatch.setattr(app_baselines, "read_scores", read_scores)
    result = compare()
    assert result.code == 0, result.err
    assert runs == [(1, False, False), (1, True, True)] and reads == [True]


@pytest.mark.usefixtures("evaluate_cli")
def test_missing_scores_stop_before_the_look(compare: Compare, scores: Path, ledger: Path) -> None:
    score_file(scores, "autoeval", DATASET).unlink()
    result = compare()
    assert result.code == 1 and "baselines score" in result.err
    assert read_entries(ledger) == ()


@pytest.mark.usefixtures("evaluate_cli")
def test_only_the_frozen_artifact_is_compared(
    compare: Compare, frozen: Path, ledger: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other.json"
    other.write_text(frozen.read_text(encoding="utf-8"), encoding="utf-8")
    result = compare(artifact=other)
    assert result.code == 2 and "frozen artifact" in result.err
    assert read_entries(ledger) == ()


@pytest.mark.usefixtures("evaluate_cli")
def test_the_sealed_split_needs_the_variable(compare: Compare, ledger: Path) -> None:
    result = compare(env={})
    assert result.code == 2 and "PHYSVERIFY_UNSEAL" in result.err
    assert read_entries(ledger) == ()


@pytest.mark.usefixtures("evaluate_cli")
@pytest.mark.parametrize("split", [Split.EVAL, Split.CALIB])
def test_clips_are_refused_on_a_sealed_set_before_any_read(
    compare: Compare, ledger: Path, monkeypatch: pytest.MonkeyPatch, split: Split
) -> None:
    """Whatever split the clips are in: no ledger line, no session, no clip listed."""
    reached: list[str] = []
    for name in ("calibration_session", "dataset_for"):
        real = getattr(app_baselines, name)

        def spy(*args: Any, name: str = name, real: Any = real) -> Any:
            reached.append(name)
            return real(*args)

        monkeypatch.setattr(app_baselines, name, spy)
    clip = next(c.clip_id for c in REPORT.clips if c.split is split)
    result = compare("--clips", clip)
    assert result.code == 2 and "--clips narrows development sets only" in result.err
    assert read_entries(ledger) == () and reached == []


@pytest.mark.usefixtures("filed", "evaluate_cli")
def test_clips_narrow_a_development_set_without_a_look(
    compare: Compare, version: Callable[[str], None], ledger: Path, tmp_path: Path
) -> None:
    version("v2")
    clip = next(c.clip_id for c in REPORT.clips if c.split is Split.EVAL)
    result = compare("--clips", clip, env={})
    assert result.code == 0, result.err
    part = read_model(PartRows, tmp_path / "vlm" / "kubric.json")
    assert [r.clip_id for s in part.sets for r in s.rows] == [clip]
    assert read_entries(ledger) == ()


def _printed(tmp_path: Path, label: str, out: str) -> Path:
    """The path ``printed LABEL`` of scripts/h200/_steps.sh reads from ``out``."""
    log = tmp_path / f"{label}.log"
    log.write_text(out, encoding="utf-8")
    return Path(steps_read(log, "printed", label, str(log)))
