"""The sealed-ledger guard through the commands (WP-O AC12-AC15, C5 and the freeze rule)."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, ClassVar

import pytest
import typer.main
from calibration_cases import DATASET, fitted_artifact, synthetic_report
from cli_runner import CliResult, CliRunner

from physverify.calibration.sealed import LedgerEntry, read_entries
from physverify.cli import context
from physverify.cli.commands import _sealed, abstain, calibrate, diagnose, verify
from physverify.cli.context import COMMANDS_PACKAGE, check_command, command_name
from physverify.cli.main import build_app
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.core.errors import PhysverifyError
from physverify.core.registry import discover_modules
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.diagnosis import scene_stats
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split
from physverify.verification.runner import VerificationReport, write_report

EXIT_USAGE = 2
REACHED = 9
CALIB = synthetic_report(30)
HELD = synthetic_report(20, seed=7, split=Split.EVAL, prefix="e")
REPORT = VerificationReport(
    CALIB.results + HELD.results, CALIB.table + HELD.table, CALIB.clips + HELD.clips
)
SEAL = ("--set", f'sealed.sets=["{DATASET}:eval"]', "--set", "calibration.bootstrap=50")
UNSEAL = {"PHYSVERIFY_UNSEAL": f"{DATASET}:eval"}
CALIB_CLIP = CALIB.clips[0].clip_id
THREE = ("--set", "runtime.threads=3")
"""Moves the full config_hash but not the look's (paths and threads are left out)."""
SECTIONS = ("--set", "calibration.bootstrap=40")
"""Moves the look's config hash, hence its frozen state."""


class ReachedError(PhysverifyError):
    """Raised by a sentinel standing for the first computation of a guarded command."""

    exit_code = REACHED


class SealFixtureDataset:
    name: ClassVar[str] = DATASET

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return REPORT.clips

    def video(self, clip: ClipManifest) -> VideoSource:
        raise NotImplementedError

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return None

    def fetch(self, subset: str | None, dest: Path) -> None:
        return None


@pytest.fixture(autouse=True)
def fixture_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    kinds: Mapping[str, Any] = context.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {**kinds, "datasets": lambda: {**datasets(), DATASET: SealFixtureDataset}}
    monkeypatch.setattr(context, "PLUGIN_KINDS", MappingProxyType(patched))


@pytest.fixture
def checkout(config_dir: Path) -> Path:
    """The copied repository committed: a sealed look needs a clean git checkout."""
    root = config_dir.parent
    git = ("git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
           "-c", "commit.gpgsign=false")  # fmt: skip
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-qm", "fixture")):
        subprocess.run([*git, *args], check=True, capture_output=True)
    return root


@pytest.fixture
def ledger(environ: dict[str, str], checkout: Path) -> Path:
    """Created once by hand, as on the H200."""
    path = Path(environ["HOME"]) / "sealed-ledger.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@pytest.fixture
def settings(config_dir: Path, environ: dict[str, str]) -> Settings:
    return load_settings(config_dir, environ=environ)


@pytest.fixture
def reached(ledger: Path) -> list[int]:
    """Each sentinel call records how many ledger lines existed when it ran."""
    return []


def sentinel(ledger: Path, calls: list[int]) -> Callable[..., Any]:
    def stop(*args: object, **kwargs: object) -> Any:
        calls.append(len(read_entries(ledger)))
        raise ReachedError("computation reached")

    return stop


class _SentinelRunner:
    def __init__(self, stop: Callable[..., Any], *args: object) -> None:
        self.run = stop


class _ReportRunner:
    """``calibrate evaluate``'s checks: the fixture report restricted to the asked clips."""

    def run(self, source: object, clips: Sequence[ClipManifest]) -> VerificationReport:
        ids = {c.clip_id for c in clips}
        return VerificationReport(
            tuple(r for r in REPORT.results if r.clip_id in ids),
            tuple(r for r in REPORT.table if r.clip_id in ids),
            tuple(c for c in REPORT.clips if c.clip_id in ids),
        )


@pytest.fixture(scope="module")
def artifact_text() -> str:
    return fitted_artifact(30, bootstrap=50).to_json()


@pytest.fixture
def evaluate_cli(
    cli: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ledger: Path,
    reached: list[int],
    artifact_text: str,
) -> Callable[..., CliResult]:
    """``calibrate evaluate`` on the fixture; the first check run records the ledger size."""
    monkeypatch.setattr(calibrate, "_checks_hash", lambda session: "checks")
    monkeypatch.setattr(calibrate, "_raw_runner", lambda session: _ReportRunner())
    monkeypatch.setattr(calibrate, "_source", lambda session, dataset, level: None)
    real = calibrate._eval_data

    def counted(*args: Any) -> Any:
        reached.append(len(read_entries(ledger)))
        return real(*args)

    monkeypatch.setattr(calibrate, "_eval_data", counted)
    path = tmp_path / "artifact" / "calibration.json"
    path.parent.mkdir(parents=True)
    path.write_text(artifact_text, encoding="utf-8")

    def invoke(
        *extra: str,
        options: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        artifact: Path = path,
    ) -> CliResult:
        command = ("calibrate", "evaluate", "--dataset", DATASET, "--no-real", "--artifact")
        return cli(*SEAL, *options, *command, str(artifact), *extra, env=env)

    return invoke


def test_evaluate_on_the_sealed_split_needs_the_variable(
    evaluate_cli: Callable[..., CliResult], ledger: Path, reached: list[int]
) -> None:
    """AC12: exit 2, no line, nothing computed."""
    result = evaluate_cli()
    assert result.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL" in result.err
    assert read_entries(ledger) == () and reached == []


def test_evaluate_leaves_one_line_before_computing(
    evaluate_cli: Callable[..., CliResult], ledger: Path, reached: list[int]
) -> None:
    """AC13: one line, written before the first check runs, with the artifact's sha256."""
    result = evaluate_cli(env=UNSEAL)
    assert result.code == 0, result.err
    (line,) = read_entries(ledger)
    assert reached and set(reached) == {1}
    evaluation = Path(result.out.split("evaluation: ")[1].split()[0])
    sha = json.loads(evaluation.read_text(encoding="utf-8"))["artifact_sha256"]
    assert line.artifacts["calibration"] == sha
    assert line.sets == (f"{DATASET}:eval",) and line.command == "calibrate evaluate"
    assert len(line.commit) == 40
    assert line.datasets[0].n_clips == len(REPORT.clips) and not line.repeat
    assert f"ledger: {ledger}" in result.err


def test_evaluate_on_calib_clips_leaves_no_line(
    evaluate_cli: Callable[..., CliResult], ledger: Path, reached: list[int]
) -> None:
    """AC15: a calib-only selection does not touch the sealed split."""
    result = evaluate_cli("--clips", CALIB_CLIP)
    assert result.code == 0, result.err
    assert read_entries(ledger) == () and reached


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a read-only file")
def test_an_unwritable_ledger_stops_before_computing(
    evaluate_cli: Callable[..., CliResult], ledger: Path, reached: list[int]
) -> None:
    """AC14."""
    ledger.chmod(0o444)
    result = evaluate_cli(env=UNSEAL)
    assert result.code == 1 and "cannot write the sealed ledger" in result.err
    assert reached == []


def test_a_missing_ledger_fails_closed_and_is_not_created(
    evaluate_cli: Callable[..., CliResult], tmp_path: Path, reached: list[int]
) -> None:
    """Audit M2: a wrong path (or HOME) never starts an empty ledger."""
    absent = tmp_path / "elsewhere" / "sealed-ledger.txt"
    result = evaluate_cli(options=("--set", f"sealed.ledger={absent}"), env=UNSEAL)
    assert result.code == 1 and f"touch {absent}" in result.err
    assert f"ledger: {absent}" in result.err
    assert not absent.parent.exists() and reached == []


def test_another_artifact_at_the_same_commit_is_another_freeze(
    evaluate_cli: Callable[..., CliResult], ledger: Path, tmp_path: Path, reached: list[int]
) -> None:
    """Audit M1: a refit (another --artifact) needs the repeat variable."""
    assert evaluate_cli(env=UNSEAL).code == 0
    refit = tmp_path / "refit" / "calibration.json"
    refit.parent.mkdir()
    refit.write_text(fitted_artifact(30, bootstrap=40).to_json(), encoding="utf-8")
    again = evaluate_cli(env=UNSEAL, artifact=refit)
    assert again.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL_REPEAT" in again.err
    assert len(read_entries(ledger)) == 1
    repeat = evaluate_cli(env={**UNSEAL, "PHYSVERIFY_UNSEAL_REPEAT": "1"}, artifact=refit)
    assert repeat.code == 0, repeat.err
    first, second = read_entries(ledger)
    assert second.repeat and second.artifacts["calibration"] != first.artifacts["calibration"]


def _guarded_verify(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int]
) -> Callable[..., CliResult]:
    stop = sentinel(ledger, reached)
    monkeypatch.setattr(verify, "VerificationRunner", lambda *a: _SentinelRunner(stop))

    def invoke(
        *extra: str, options: Sequence[str] = (), env: Mapping[str, str] | None = None
    ) -> CliResult:
        command = ("verify", "--dataset", DATASET, "--level", "oracle", *extra)
        return cli(*SEAL, *options, *command, env=env)

    return invoke


@pytest.fixture
def guarded(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int]
) -> Callable[[str], Callable[..., CliResult]]:
    """A guarded command whose first computation is a sentinel, by name."""
    stop = sentinel(ledger, reached)

    def abstain_curve(*extra: str, env: Mapping[str, str] | None = None) -> CliResult:
        monkeypatch.setattr(abstain, "_session", stop)
        sets = ("--set", f"abstention.dataset={DATASET}", "--set", "abstention.real_datasets=[]")
        return cli(*SEAL, *sets, *extra, "abstain", "curve", env=env)

    def diagnose_stages(*extra: str, env: Mapping[str, str] | None = None) -> CliResult:
        monkeypatch.setattr(diagnose, "run_stage_errors", stop)
        return cli(*SEAL, *extra, "diagnose", "stages", "--dataset", DATASET, env=env)

    def scene_stats_per_clip(*extra: str, env: Mapping[str, str] | None = None) -> CliResult:
        monkeypatch.setattr(scene_stats, "collect_rows", stop)
        return cli(*SEAL, *extra, "scene-stats", "per-clip", "--dataset", DATASET,
                   "--label", "x", "--out", str(ledger.parent / "x.json"), env=env)  # fmt: skip

    commands = {
        "verify": _guarded_verify(cli, monkeypatch, ledger, reached),
        "abstain curve": abstain_curve,
        "diagnose stages": diagnose_stages,
        "scene-stats per-clip": scene_stats_per_clip,
    }
    return commands.__getitem__


COMMANDS = ("verify", "abstain curve", "diagnose stages", "scene-stats per-clip")


@pytest.mark.parametrize("name", COMMANDS)
def test_each_guarded_command_refuses_without_the_variable(
    guarded: Callable[[str], Callable[..., CliResult]],
    ledger: Path,
    reached: list[int],
    name: str,
) -> None:
    """AC12."""
    result = guarded(name)()
    assert result.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL" in result.err
    assert read_entries(ledger) == () and reached == []


@pytest.mark.parametrize("name", COMMANDS)
def test_each_guarded_command_writes_its_line_first(
    guarded: Callable[[str], Callable[..., CliResult]],
    ledger: Path,
    reached: list[int],
    name: str,
) -> None:
    """AC13: exactly one line per invocation, there when the computation starts."""
    result = guarded(name)(env=UNSEAL)
    assert result.code == REACHED, result.err
    (line,) = read_entries(ledger)
    assert reached == [1] and line.command == name


def test_verify_on_calib_clips_leaves_no_line(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int]
) -> None:
    """AC15."""
    run = _guarded_verify(cli, monkeypatch, ledger, reached)
    assert run("--clips", CALIB_CLIP).code == REACHED
    assert reached == [0] and read_entries(ledger) == ()


def test_without_the_sealed_section_every_guarded_command_fails_closed(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int],
    config_dir: Path,
) -> None:  # fmt: skip
    """Audit M3: even a calib-only selection stops when the sealed sets are unknown."""
    (config_dir / "sealed.toml").unlink()
    stop = sentinel(ledger, reached)
    monkeypatch.setattr(verify, "VerificationRunner", lambda *a: _SentinelRunner(stop))
    result = cli("verify", "--dataset", DATASET, "--level", "oracle", "--clips", CALIB_CLIP)
    assert result.code == EXIT_USAGE and "no [sealed] section" in result.err
    assert reached == []


def test_a_dirty_tree_cannot_look(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int],
    checkout: Path,
) -> None:  # fmt: skip
    """Audit m2: the commit of the freeze must be the tree that runs."""
    (checkout / "src").mkdir()
    (checkout / "src" / "patch.py").write_text("x = 1\n", encoding="utf-8")
    result = _guarded_verify(cli, monkeypatch, ledger, reached)(env=UNSEAL)
    assert result.code == EXIT_USAGE and "src/patch.py" in result.err
    assert read_entries(ledger) == () and reached == []


def test_a_refit_of_the_default_artifact_is_another_freeze(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int],
    settings: Settings,
) -> None:  # fmt: skip
    """Audit M1: calibration.json and exclusions.json enter the freeze."""
    run = _guarded_verify(cli, monkeypatch, ledger, reached)
    calibration_dir = settings.paths.calibration_dir
    calibration_dir.mkdir(parents=True)
    for name in ("calibration.json", "exclusions.json"):
        (calibration_dir / name).write_text("{}\n", encoding="utf-8")
    assert run(env=UNSEAL).code == REACHED
    for name in ("calibration.json", "exclusions.json"):
        (calibration_dir / name).write_text('{"refit": 1}\n', encoding="utf-8")
        moved = run(env=UNSEAL)
        assert moved.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL_REPEAT" in moved.err
        assert run(env={**UNSEAL, "PHYSVERIFY_UNSEAL_REPEAT": "1"}).code == REACHED
    lines = read_entries(ledger)
    assert "exclusions" in lines[-1].artifacts and "calibration" in lines[-1].artifacts


def test_another_frozen_state_needs_the_repeat_variable(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int]
) -> None:
    """Juicio answer 4: the same freeze looks again, another one needs the repeat variable."""
    run = _guarded_verify(cli, monkeypatch, ledger, reached)
    assert run(env=UNSEAL).code == REACHED
    assert run(env=UNSEAL).code == REACHED
    assert run(options=THREE, env=UNSEAL).code == REACHED
    first, second, third = read_entries(ledger)
    assert first.freeze == second.freeze == third.freeze and not third.repeat
    moved = run(options=SECTIONS, env=UNSEAL)
    assert moved.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL_REPEAT" in moved.err
    assert len(read_entries(ledger)) == 3
    repeat = run(options=SECTIONS, env={**UNSEAL, "PHYSVERIFY_UNSEAL_REPEAT": "1"})
    assert repeat.code == REACHED and "earlier look" in repeat.err
    last: LedgerEntry = read_entries(ledger)[-1]
    assert last.repeat and last.freeze != first.freeze


def _leaves() -> dict[str, ModuleType]:
    commands = discover_modules(COMMANDS_PACKAGE, check_command, command_name)
    root = typer.main.get_command(build_app(commands))
    found: dict[str, ModuleType] = {}
    for name, module in commands.items():
        group = root.commands[name]  # type: ignore[attr-defined]
        children = sorted(getattr(group, "commands", {}))
        for leaf in [f"{name} {child}" for child in children] or [name]:
            found[leaf] = module
    return found


def test_every_subcommand_is_guarded_or_exempt_with_a_reason() -> None:
    """C5: a new subcommand fails here until ``_sealed`` classifies it."""
    leaves = _leaves()
    assert not _sealed.GUARDED & set(_sealed.EXEMPT)
    assert set(leaves) == _sealed.GUARDED | set(_sealed.EXEMPT)
    assert all(reason.strip() for reason in _sealed.EXEMPT.values())
    for leaf in _sealed.GUARDED:
        assert "guard_sealed" in inspect.getsource(leaves[leaf]), leaf


def test_a_guard_for_an_unclassified_command_is_a_bug() -> None:
    """The classification is checked before the context is read."""
    with pytest.raises(ValueError, match="not a guarded"):
        _sealed.guard_sealed(None, "calibrate show", [])  # type: ignore[arg-type]


def test_fit_from_a_perceived_run_guards_before_the_oracle_runs(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int],
    settings: Settings,
) -> None:  # fmt: skip
    """Audit m3: the oracle a --from-run fit completes is computed after the guard."""
    perceived = synthetic_report(
        20, seed=7, split=Split.EVAL, prefix="e", level=InputLevel.PERCEIVED
    )
    folder = settings.paths.runs_dir / "perceived"
    folder.mkdir(parents=True)
    write_report(folder, perceived)
    stop = sentinel(ledger, reached)
    monkeypatch.setattr(calibrate, "_raw_runner", lambda session: _SentinelRunner(stop))
    refused = cli(*SEAL, "calibrate", "fit", "--dataset", DATASET, "--from-run", "perceived")
    assert refused.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL" in refused.err
    assert reached == []
    looked = cli(*SEAL, "calibrate", "fit", "--dataset", DATASET, "--from-run", "perceived",
                 env=UNSEAL)  # fmt: skip
    assert looked.code == REACHED and reached == [1]
