"""The sealed-ledger guard through the commands and the freeze rule."""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, ClassVar

import pytest
from calibration_cases import DATASET, fitted_artifact, synthetic_report
from cli_runner import CliResult, CliRunner
from cli_tree import cli_leaves, command_modules

from physverify.app import calibration as app_calibration
from physverify.app import diagnosis as app_diagnosis
from physverify.app import plugins, sealing
from physverify.app import selective_risk as app_selective_risk
from physverify.app import verification as app_verification
from physverify.calibration import session as calibration_session
from physverify.calibration.sealed_ledger import LedgerEntry, read_entries
from physverify.config.models import Settings
from physverify.core.errors import PhysverifyError
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


def _real_fixture(dataset: str) -> type[SealFixtureDataset]:
    """A real control standing in under ``dataset``: two fixture clips relabelled."""
    clips = tuple(replace(c, dataset=dataset) for c in CALIB.clips[:2])

    class RealFixture(SealFixtureDataset):
        name: ClassVar[str] = dataset

        def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
            return clips

    return RealFixture


NEW_REAL = ("physics_iq_views", "objectron_static")
DEV_REAL = ("physics_iq", "pisa_real")
"""``calibration.real_datasets`` as shipped."""


@pytest.fixture(autouse=True)
def fixture_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    datasets = kinds["datasets"]
    reals = {name: _real_fixture(name) for name in (*NEW_REAL, *DEV_REAL)}
    fixtures = {DATASET: SealFixtureDataset, **reals}
    patched = {**kinds, "datasets": lambda: {**datasets(), **fixtures}}
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))


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
    monkeypatch.setattr(app_calibration, "checks_hash_for", lambda context: "checks")
    monkeypatch.setattr(app_calibration, "_raw_runner", lambda session: _ReportRunner())
    monkeypatch.setattr(app_calibration, "_input_source", lambda session, dataset, level: None)
    real = calibration_session._eval_data

    def counted(*args: Any) -> Any:
        reached.append(len(read_entries(ledger)))
        return real(*args)

    monkeypatch.setattr(calibration_session, "_eval_data", counted)
    path = tmp_path / "artifact" / "calibration.json"
    path.parent.mkdir(parents=True)
    path.write_text(artifact_text, encoding="utf-8")

    def invoke(
        *extra: str,
        options: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        artifact: Path = path,
        seal: Sequence[str] = SEAL,
        real: Sequence[str] = ("--no-real",),
    ) -> CliResult:
        command = ("calibrate", "evaluate", "--dataset", DATASET, *real, "--artifact")
        return cli(*seal, *options, *command, str(artifact), *extra, env=env)

    return invoke


def test_evaluate_on_the_sealed_split_needs_the_variable(
    evaluate_cli: Callable[..., CliResult], ledger: Path, reached: list[int]
) -> None:
    """Exit 2, no line, nothing computed."""
    result = evaluate_cli()
    assert result.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL" in result.err
    assert read_entries(ledger) == () and reached == []


def test_evaluate_leaves_one_line_before_computing(
    evaluate_cli: Callable[..., CliResult], ledger: Path, reached: list[int]
) -> None:
    """One line, written before the first check runs, with the artifact's sha256."""
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
    """A calib-only selection does not touch the sealed split."""
    result = evaluate_cli("--clips", CALIB_CLIP)
    assert result.code == 0, result.err
    assert read_entries(ledger) == () and reached


REAL_SEAL = (
    "--set", f'sealed.sets=["{DATASET}:eval", "{NEW_REAL[0]}", "{NEW_REAL[1]}"]',
    "--set", "calibration.bootstrap=50",
)  # fmt: skip
REAL_UNSEAL = {"PHYSVERIFY_UNSEAL": ",".join((f"{DATASET}:eval", *NEW_REAL))}
NAMED = tuple(arg for name in NEW_REAL for arg in ("--real-dataset", name))


def test_the_named_real_datasets_enter_the_ledger_line(
    evaluate_cli: Callable[..., CliResult], ledger: Path, settings: Settings
) -> None:
    """``--real-dataset`` replaces ``calibration.real_datasets`` in the
    selections, so the line names the new controls only and its config hash stays the same."""
    named = evaluate_cli(seal=REAL_SEAL, real=NAMED, env=REAL_UNSEAL)
    assert named.code == 0, named.err
    unnamed = evaluate_cli(seal=REAL_SEAL, real=(), env=REAL_UNSEAL)
    assert unnamed.code == 0, unnamed.err
    first, second = read_entries(ledger)
    assert [d.name for d in first.datasets] == [DATASET, *sorted(NEW_REAL)]
    assert set(first.sets) == {f"{DATASET}:eval", *NEW_REAL}
    assert tuple(settings.sections["calibration"]["real_datasets"]) == DEV_REAL
    assert [first.argv.count(n) for n in NEW_REAL] == [1, 1]
    assert first.config_hash == second.config_hash and first.freeze == second.freeze
    assert [d.name for d in second.datasets] == [DATASET, *sorted(DEV_REAL)]


def test_a_named_real_dataset_must_resolve(evaluate_cli: Callable[..., CliResult]) -> None:
    result = evaluate_cli(real=("--real-dataset", "no_such_real"), env=UNSEAL)
    assert result.code != 0 and "no_such_real" in result.err


def test_named_real_datasets_and_no_real_exclude_each_other(
    evaluate_cli: Callable[..., CliResult], ledger: Path
) -> None:
    result = evaluate_cli(real=("--no-real", *NAMED), env=REAL_UNSEAL)
    assert result.code == EXIT_USAGE and "--no-real" in result.err
    assert read_entries(ledger) == ()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a read-only file")
def test_an_unwritable_ledger_stops_before_computing(
    evaluate_cli: Callable[..., CliResult], ledger: Path, reached: list[int]
) -> None:
    ledger.chmod(0o444)
    result = evaluate_cli(env=UNSEAL)
    assert result.code == 1 and "cannot write the sealed ledger" in result.err
    assert reached == []


def test_a_missing_ledger_fails_closed_and_is_not_created(
    evaluate_cli: Callable[..., CliResult], tmp_path: Path, reached: list[int]
) -> None:
    """A wrong path (or HOME) never starts an empty ledger."""
    absent = tmp_path / "elsewhere" / "sealed-ledger.txt"
    result = evaluate_cli(options=("--set", f"sealed.ledger={absent}"), env=UNSEAL)
    assert result.code == 1 and f"touch {absent}" in result.err
    assert f"ledger: {absent}" in result.err
    assert not absent.parent.exists() and reached == []


def test_another_artifact_at_the_same_commit_is_another_freeze(
    evaluate_cli: Callable[..., CliResult], ledger: Path, tmp_path: Path, reached: list[int]
) -> None:
    """A refit (another --artifact) needs the repeat variable."""
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
    monkeypatch.setattr(app_verification, "VerificationRunner", lambda *a: _SentinelRunner(stop))

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

    def selective_risk(leaf: str, *own: str) -> Callable[..., CliResult]:
        def run(*extra: str, env: Mapping[str, str] | None = None) -> CliResult:
            monkeypatch.setattr(app_selective_risk, "_session", stop)
            sets = (
                "--set",
                f"abstention.dataset={DATASET}",
                "--set",
                "abstention.real_datasets=[]",
            )
            return cli(*SEAL, *sets, *own, *extra, "selective-risk", leaf, env=env)

        return run

    def diagnose_stages(*extra: str, env: Mapping[str, str] | None = None) -> CliResult:
        monkeypatch.setattr(app_diagnosis, "run_stage_errors", stop)
        return cli(*SEAL, *extra, "diagnose", "stages", "--dataset", DATASET, env=env)

    def scene_stats_per_clip(*extra: str, env: Mapping[str, str] | None = None) -> CliResult:
        monkeypatch.setattr(scene_stats, "collect_rows", stop)
        return cli(*SEAL, *extra, "scene-stats", "per-clip", "--dataset", DATASET,
                   "--label", "x", "--out", str(ledger.parent / "x.json"), env=env)  # fmt: skip

    commands = {
        "verify": _guarded_verify(cli, monkeypatch, ledger, reached),
        "abstain curve": selective_risk("curve"),
        "abstain fit": selective_risk("fit", "--set", "abstention.calib_split=eval"),
        "diagnose stages": diagnose_stages,
        "scene-stats per-clip": scene_stats_per_clip,
    }
    return commands.__getitem__


COMMANDS = ("verify", "abstain curve", "abstain fit", "diagnose stages", "scene-stats per-clip")
"""Ledger labels; ``abstain *`` run as ``selective-risk *`` and keep their persisted label."""


@pytest.mark.parametrize("name", COMMANDS)
def test_each_guarded_command_refuses_without_the_variable(
    guarded: Callable[[str], Callable[..., CliResult]],
    ledger: Path,
    reached: list[int],
    name: str,
) -> None:
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
    """Exactly one line per invocation, there when the computation starts."""
    result = guarded(name)(env=UNSEAL)
    assert result.code == REACHED, result.err
    (line,) = read_entries(ledger)
    assert reached == [1] and line.command == name


def test_verify_on_calib_clips_leaves_no_line(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int]
) -> None:
    run = _guarded_verify(cli, monkeypatch, ledger, reached)
    assert run("--clips", CALIB_CLIP).code == REACHED
    assert reached == [0] and read_entries(ledger) == ()


def test_without_the_sealed_section_every_guarded_command_fails_closed(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int],
    config_dir: Path,
) -> None:  # fmt: skip
    """Even a calib-only selection stops when the sealed sets are unknown."""
    (config_dir / "verifier" / "sealed.toml").unlink()
    stop = sentinel(ledger, reached)
    monkeypatch.setattr(app_verification, "VerificationRunner", lambda *a: _SentinelRunner(stop))
    result = cli("verify", "--dataset", DATASET, "--level", "oracle", "--clips", CALIB_CLIP)
    assert result.code == EXIT_USAGE and "no [sealed] section" in result.err
    assert reached == []


def test_a_dirty_tree_cannot_look(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int],
    checkout: Path,
) -> None:  # fmt: skip
    """The commit of the freeze must be the tree that runs."""
    (checkout / "src").mkdir()
    (checkout / "src" / "patch.py").write_text("x = 1\n", encoding="utf-8")
    result = _guarded_verify(cli, monkeypatch, ledger, reached)(env=UNSEAL)
    assert result.code == EXIT_USAGE and "src/patch.py" in result.err
    assert read_entries(ledger) == () and reached == []


def test_a_refit_of_the_default_artifact_is_another_freeze(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int],
    settings: Settings,
) -> None:  # fmt: skip
    """The calibration.json and exclusions.json artifacts enter the freeze."""
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
    """The same freeze looks again, another one needs the repeat variable."""
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


def _guard_labels(module: ModuleType) -> set[str]:
    """The ledger labels a command module's guard calls form: ``guard_sealed(context, label)``
    directly, or through a helper of the module that passes ``f"{NAME} {leaf}"`` on."""
    tree = ast.parse(inspect.getsource(module))
    labels: set[str] = set()
    helpers: dict[str, int] = {}
    for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        params = [arg.arg for arg in function.args.args]
        for call in _calls(function, "guard_sealed"):
            label = _label(call.args[1], module.NAME)
            if label is None:
                helpers[function.name] = params.index(_leaf_param(call.args[1]))
            else:
                labels.add(label)
    for name, index in helpers.items():
        for call in _calls(tree, name):
            leaf = call.args[index]
            assert isinstance(leaf, ast.Constant) and isinstance(leaf.value, str), ast.dump(leaf)
            labels.add(f"{module.NAME} {leaf.value}")
    return labels | _service_labels(module, tree)


def _service_labels(module: ModuleType, tree: ast.Module) -> set[str]:
    """Labels ``NAME`` or ``f"{NAME} {leaf}"`` a command passes to a service of another module
    that hands that argument on to ``guard_sealed``."""
    labels: set[str] = set()
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        service = _callee(module, call.func)
        if getattr(service, "__name__", None) == "guard_sealed":
            continue
        if not inspect.isfunction(service) or service.__module__ == module.__name__:
            continue
        for index, arg in enumerate(call.args):
            if not (isinstance(arg, ast.JoinedStr) or _is_name(arg)):
                continue
            label = _label(arg, module.NAME)
            if label is not None and _reaches_guard(service, index):
                labels.add(label)
    return labels


def _callee(module: ModuleType, func: ast.expr) -> object:
    """What a call ``f(...)`` or ``service.f(...)`` in the command module calls."""
    if isinstance(func, ast.Name):
        return getattr(module, func.id, None)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return getattr(getattr(module, func.value.id, None), func.attr, None)
    return None


def _is_name(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "NAME"


def _reaches_guard(function: Callable[..., Any], index: int) -> bool:
    home = inspect.getmodule(function)
    assert home is not None
    tree = ast.parse(inspect.getsource(home))
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    return _forwards(defs, defs[function.__name__], index, set())


def _forwards(
    defs: Mapping[str, ast.FunctionDef], function: ast.FunctionDef, index: int, seen: set[str]
) -> bool:
    """Whether positional parameter ``index`` of ``function`` reaches ``guard_sealed``'s label."""
    if function.name in seen:
        return False
    seen.add(function.name)
    param = function.args.args[index].arg
    for call in (n for n in ast.walk(function) if isinstance(n, ast.Call)):
        if not isinstance(call.func, ast.Name):
            continue
        for position, arg in enumerate(call.args):
            if not (isinstance(arg, ast.Name) and arg.id == param):
                continue
            if call.func.id == "guard_sealed" and position == 1:
                return True
            callee = defs.get(call.func.id)
            if callee is not None and _forwards(defs, callee, position, seen):
                return True
    return False


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def _label(node: ast.expr, command: str) -> str | None:
    """The literal label, NAME resolved; None when a parameter supplies the leaf."""
    if _is_name(node):
        return command
    assert isinstance(node, ast.JoinedStr), ast.dump(node)
    parts: list[str] = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            parts.append(str(part.value))
        elif isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name):
            if part.value.id != "NAME":
                return None
            parts.append(command)
    return "".join(parts)


def _leaf_param(node: ast.expr) -> str:
    assert isinstance(node, ast.JoinedStr)
    last = node.values[-1]
    assert isinstance(last, ast.FormattedValue) and isinstance(last.value, ast.Name)
    return last.value.id


def test_every_subcommand_is_guarded_or_exempt_with_a_reason() -> None:
    """A new subcommand fails here until ``sealing`` classifies it."""
    leaves = cli_leaves()
    assert not sealing.GUARDED_LEAVES & set(sealing.EXEMPT)
    assert set(leaves) == sealing.GUARDED_LEAVES | set(sealing.EXEMPT)
    assert all(reason.strip() for reason in sealing.EXEMPT.values())


def test_guarded_leaves_and_ledger_labels_are_a_bijection() -> None:
    """Every guarded leaf writes its own persisted label, and a renamed leaf's label is never
    another leaf's."""
    labels = [sealing.ledger_label(leaf) for leaf in sealing.GUARDED_LEAVES]
    assert len(set(labels)) == len(labels) == len(sealing.GUARDED)
    assert set(labels) == sealing.GUARDED
    assert set(sealing.LEDGER_LABELS_BY_LEAF.values()) <= sealing.GUARDED
    assert set(sealing.LEDGER_LABELS_BY_LEAF) <= sealing.GUARDED_LEAVES
    assert not set(sealing.LEDGER_LABELS_BY_LEAF.values()) & set(cli_leaves())


@pytest.mark.parametrize("leaf", sorted(sealing.GUARDED_LEAVES))
def test_the_guarded_leaf_guards_with_its_own_label(leaf: str) -> None:
    """The leaf a guard is called with is ``f"{NAME} {leaf}"`` of the leaf it guards; the ledger
    records ``ledger_label`` of it."""
    assert leaf in _guard_labels(cli_leaves()[leaf])
    assert sealing.ledger_label(leaf) in sealing.GUARDED


def test_every_label_a_guard_forms_is_a_guarded_leaf() -> None:
    formed = {label for module in command_modules().values() for label in _guard_labels(module)}
    assert formed == sealing.GUARDED_LEAVES
    assert {sealing.ledger_label(leaf) for leaf in formed} == sealing.GUARDED


def test_a_guard_for_an_unclassified_command_is_a_bug() -> None:
    """The classification is checked before the context is read."""
    with pytest.raises(ValueError, match="not a guarded"):
        sealing.guard_sealed(None, "calibrate show", [], err=print)  # type: ignore[arg-type]


def test_fit_from_a_perceived_run_guards_before_the_oracle_runs(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch, ledger: Path, reached: list[int],
    settings: Settings,
) -> None:  # fmt: skip
    """The oracle a --from-run fit completes is computed after the guard."""
    perceived = synthetic_report(
        20, seed=7, split=Split.EVAL, prefix="e", level=InputLevel.PERCEIVED
    )
    folder = settings.paths.runs_dir / "perceived"
    folder.mkdir(parents=True)
    write_report(folder, perceived)
    stop = sentinel(ledger, reached)
    monkeypatch.setattr(app_calibration, "_raw_runner", lambda session: _SentinelRunner(stop))
    refused = cli(*SEAL, "calibrate", "fit", "--dataset", DATASET, "--from-run", "perceived")
    assert refused.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL" in refused.err
    assert reached == []
    looked = cli(*SEAL, "calibrate", "fit", "--dataset", DATASET, "--from-run", "perceived",
                 env=UNSEAL)  # fmt: skip
    assert looked.code == REACHED and reached == [1]
