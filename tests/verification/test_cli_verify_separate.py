"""``physverify verify-separate`` over the fixture clip: R1_dyn rows only, never scored."""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from cli_runner import CliResult, CliRunner
from synthetic_clip import SyntheticFixtureDataset

from physverify.app import plugins, separate_rows
from physverify.app.context import AppContext
from physverify.calibration.collect import Subjects
from physverify.calibration.row_sigma import ROW_SIGMA_FILE, ClipResiduals, RowNoise, fit_row_sigma
from physverify.datasets._base import ArrayGroundTruthStore, NativeGroundTruth
from physverify.runtime.heavy import holder_path
from physverify.schema.checks import InputLevel, Verdict
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.scene import MagnitudeKind, PerturbationRecord
from physverify.schema.stages import Stage
from physverify.schema.states import ground_truth_source_key
from physverify.verification.inputs import InputSource, OracleSource
from physverify.verification.runner import TABLE_FILE, RowStatus

COMMAND = "verify-separate"
DATASET = ("--dataset", SyntheticFixtureDataset.name)
EXIT_USAGE = 2


CALIB_DATASET = ("--dataset", "synthetic_fixture_calib")
FIT = ("--level", "perceived", "--gt", "depth", "--fit-row-sigma")


def generated_record(name: str, level: int) -> PerturbationRecord:
    """Shaped like the generator's records: a clean clip carries ``name='none'``, not None."""
    return PerturbationRecord(
        name=name,
        target_checks=() if level == 0 else ("R1_dyn",),
        magnitude=0.1 * level,
        magnitude_unit="none" if level == 0 else "pen_over_size",
        magnitude_kind=MagnitudeKind.CHECK_UNIT if level == 0 else MagnitudeKind.INJECTED_PARAMETER,
        level=level,
        target_objects=() if level == 0 else ("ball",),
        frame_span=None if level == 0 else (0, 11),
        seed=level,
    )


def relabelled(native: NativeGroundTruth, clip_id: str) -> NativeGroundTruth:
    key = ground_truth_source_key(clip_id)
    return dataclasses.replace(
        native,
        states={
            f: dataclasses.replace(t, clip_id=clip_id, source_key=key)
            for f, t in native.states.items()
        },
        contacts={
            f: dataclasses.replace(t, clip_id=clip_id, source_key=key)
            for f, t in native.contacts.items()
        },
    )


class CalibFixtureDataset(SyntheticFixtureDataset):
    """The analytic clip as five generated calib scenes (one per fold), each clean and perturbed."""

    name = "synthetic_fixture_calib"

    def ground_truth(self, clip: ClipManifest) -> ArrayGroundTruthStore:
        return ArrayGroundTruthStore(relabelled(self._clip.native, clip.clip_id))

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        (clip,) = super().clips(subset)
        return [
            dataclasses.replace(
                clip,
                clip_id=f"f5_contact-s{scene:03d}__{suffix}",
                split=Split.CALIB,
                scene_id=f"f5_contact-s{scene:03d}",
                perturbation=generated_record(name, level),
            )
            for scene in range(5)
            for suffix, name, level in (
                ("none", "none", 0),
                ("r1_pass_through_L2", "r1_pass_through", 2),
            )
        ]


@pytest.fixture(autouse=True)
def fixture_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {
        **kinds,
        "datasets": lambda: {
            **datasets(),
            SyntheticFixtureDataset.name: SyntheticFixtureDataset,
            CalibFixtureDataset.name: CalibFixtureDataset,
        },
    }
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))


def run_dir(result: CliResult) -> Path:
    (line,) = [line for line in result.out.splitlines() if line.startswith("run: ")]
    return Path(line.removeprefix("run: "))


def table(result: CliResult) -> list[dict[str, str]]:
    with (run_dir(result) / TABLE_FILE).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verdicts(result: CliResult) -> set[str]:
    return {row["verdict"] for row in table(result)}


def test_the_oracle_run_writes_only_the_separate_rows(cli: CliRunner, lock_path: Path) -> None:
    result = cli(COMMAND, *DATASET, "--level", "oracle")
    assert result.code == 0, result.err
    assert [row["check_id"] for row in table(result)] == ["R1_dyn"]
    assert verdicts(result) <= {Verdict.UNSCORED.value, Verdict.ABSTAIN.value}
    assert not holder_path(lock_path).exists()


def test_a_run_with_ground_truth_stages_takes_the_substitution_path(cli: CliRunner) -> None:
    result = cli(COMMAND, *DATASET, "--level", "perceived", "--gt", "depth")
    assert result.code == 0, result.err
    assert {row["status"] for row in table(result)} <= {
        RowStatus.ABSTAINED.value,
        RowStatus.NO_EVENT.value,
        RowStatus.SCORED.value,
    }
    assert verdicts(result) <= {Verdict.UNSCORED.value, Verdict.ABSTAIN.value}


@pytest.mark.parametrize(
    "argv", [("--level", "oracle", "--gt", "depth"), ()], ids=["gt_on_oracle", "no_level"]
)
def test_usage_errors(cli: CliRunner, argv: tuple[str, ...]) -> None:
    assert cli(COMMAND, *DATASET, *argv).code == EXIT_USAGE


def test_the_row_sigma_is_not_fitted_on_oracle_inputs(cli: CliRunner) -> None:
    assert cli(COMMAND, *DATASET, "--level", "oracle", "--fit-row-sigma").code == EXIT_USAGE


def test_the_row_sigma_is_fitted_on_calib_clips_only(cli: CliRunner) -> None:
    result = cli(COMMAND, *DATASET, "--level", "perceived", "--gt", "depth", "--fit-row-sigma")
    assert result.code == EXIT_USAGE
    assert "calib clips only" in result.err


UNMAPPED = "f5_contact-s004__none"


@pytest.fixture
def light_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Perceived inputs read from ground truth (no weights), so every residual is 0; one clean
    clip has no subject map."""

    def inputs(
        context: AppContext, source: Any, level: InputLevel, stages: Sequence[Stage]
    ) -> InputSource:
        return OracleSource(source)

    def lookup(context: AppContext, source: Any) -> Any:
        return lambda clip: None if clip.clip_id == UNMAPPED else Subjects(None)

    monkeypatch.setattr(separate_rows, "input_source", inputs)
    monkeypatch.setattr(separate_rows, "subject_lookup", lookup)


def fitted_artifact(result: CliResult) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((run_dir(result) / ROW_SIGMA_FILE).read_text("utf-8"))
    return loaded


@pytest.mark.usefixtures("light_chain")
def test_the_row_sigma_is_fitted_on_the_generated_clean_clips(cli: CliRunner) -> None:
    result = cli(COMMAND, *CALIB_DATASET, *FIT)
    assert result.code == 0, result.err
    fitted = fitted_artifact(result)
    assert fitted["scenes_per_fold"] == [[f"f5_contact-s{j:03d}"] for j in range(4)] + [[]]
    assert [n > 0 for n in fitted["residuals_per_fold"]] == [True] * 4 + [False]
    assert fitted["by_fold"] == fitted["size_by_fold"] == [0.0] * 5
    assert fitted["autocorrelation_by_fold"] == [1.0] * 5
    assert fitted["law"] == "quantile"
    assert fitted["clips_without_residuals"] == [UNMAPPED]
    assert "clips without residuals: 1" in result.out


@pytest.mark.usefixtures("light_chain")
def test_the_perturbed_clips_are_left_out_of_the_row_sigma(cli: CliRunner) -> None:
    perturbed = [f"f5_contact-s{j:03d}__r1_pass_through_L2" for j in range(5)]
    result = cli(COMMAND, *CALIB_DATASET, *FIT, *(a for c in perturbed for a in ("--clips", c)))
    assert result.code != 0
    assert "no residuals outside it" in result.err


def test_fitting_and_using_a_row_sigma_together_is_refused(cli: CliRunner, tmp_path: Path) -> None:
    artifact = row_sigma_file(tmp_path, InputLevel.MIXED, ("depth",))
    result = cli(COMMAND, *CALIB_DATASET, *FIT, "--row-sigma", artifact)
    assert result.code == EXIT_USAGE
    assert "--row-sigma" in result.err


def row_sigma_file(
    tmp_path: Path, level: InputLevel, gt: tuple[str, ...], scene: str = "f-s"
) -> str:
    fitted = fit_row_sigma(
        {
            f"c{j}": ClipResiduals((np.array([0.5, 1.0 + j, -2.0]),), np.array([0.01, -0.02 * j]))
            for j in range(5)
        },
        {f"c{j}": f"{scene}{j:03d}" for j in range(5)},
        check_id="R1_dyn",
        level=level,
        gt_stages=gt,
    )
    path = tmp_path / ROW_SIGMA_FILE
    path.write_text(fitted.model_dump_json(by_alias=True), encoding="utf-8")
    return str(path)


def test_a_row_sigma_artifact_runs_at_its_own_level(cli: CliRunner, tmp_path: Path) -> None:
    artifact = row_sigma_file(tmp_path, InputLevel.MIXED, ("depth",))
    used = cli(COMMAND, *DATASET, "--level", "perceived", "--gt", "depth", "--row-sigma", artifact)
    assert used.code == 0, used.err
    assert {row["check_id"] for row in table(used)} == {"R1_dyn"}
    other = cli(COMMAND, *DATASET, "--level", "perceived", "--row-sigma", artifact)
    assert other.code == EXIT_USAGE


@pytest.mark.usefixtures("light_chain")
def test_each_fold_runs_with_its_own_row_sigma_in_clip_order(
    cli: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = row_sigma_file(tmp_path, InputLevel.PERCEIVED, (), scene="f5_contact-s")
    by_fold = [0.5, 1.0, 1.5, 2.0, 2.5]
    fitted = json.loads(Path(artifact).read_text("utf-8"))
    Path(artifact).write_text(json.dumps({**fitted, "by_fold": by_fold}), encoding="utf-8")
    ran: dict[str, float | None] = {}
    build = separate_rows._runner

    def spied(context: AppContext, noise: RowNoise | None) -> Any:
        sigma = None if noise is None else noise.row_sigma
        runner = build(context, noise)
        run = runner.run

        def recorded(source: InputSource, clips: Sequence[ClipManifest]) -> Any:
            ran.update({clip.clip_id: sigma for clip in clips})
            return run(source, clips)

        monkeypatch.setattr(runner, "run", recorded)
        return runner

    monkeypatch.setattr(separate_rows, "_runner", spied)
    result = cli(COMMAND, *CALIB_DATASET, "--level", "perceived", "--row-sigma", artifact)
    assert result.code == 0, result.err
    order = [clip.clip_id for clip in CalibFixtureDataset().clips()]
    assert ran == {c: by_fold[int(c.split("__")[0][-3:]) % len(by_fold)] for c in order}
    assert [row["clip_id"] for row in table(result)] == order


def test_an_unknown_row_law_is_refused(cli: CliRunner) -> None:
    assert cli(COMMAND, *CALIB_DATASET, *FIT, "--row-law", "mad").code == EXIT_USAGE


def test_a_true_lift_runs_with_the_true_camera(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Stage, ...]] = []

    def inputs(
        context: AppContext, source: Any, level: InputLevel, stages: Sequence[Stage]
    ) -> InputSource:
        seen.append(tuple(stages))
        return OracleSource(source)

    monkeypatch.setattr(separate_rows, "input_source", inputs)
    result = cli(COMMAND, *DATASET, "--level", "perceived", "--gt", "lift")
    assert result.code == 0, result.err
    assert seen == [(Stage.CAMERA, Stage.LIFT)]
