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
from repo_configs import CALIBRATION_FILE, SEALED_FILE, SEALED_LEGACY_FILE, declare
from verify_runs import write_verify_run

from physverify.app import calibration as app_calibration
from physverify.app import diagnosis as app_diagnosis
from physverify.app import plugins, sealing
from physverify.app import selective_risk as app_selective_risk
from physverify.calibration import session as calibration_session
from physverify.calibration.sealed_ledger import LedgerEntry, append_entry, read_entries
from physverify.config.models import Settings
from physverify.core.errors import PhysverifyError
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.datasets._kubric_synthetic import content_identity
from physverify.diagnosis import scene_stats
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split
from physverify.verification.runner import ConfiguredSuite, VerificationReport

EXIT_USAGE = 2
REACHED = 9
CALIB = synthetic_report(30)
HELD = synthetic_report(20, seed=7, split=Split.EVAL, prefix="e")
REPORT = VerificationReport(
    CALIB.results + HELD.results, CALIB.table + HELD.table, CALIB.clips + HELD.clips
)
FAST = ("--set", "calibration.bootstrap=50")
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
def sealed_in_files(config_dir: Path) -> None:
    """The fixture's eval split sealed by the config files, before ``checkout`` commits them."""
    declare(config_dir, SEALED_FILE, "sets", f'["{DATASET}:eval"]')


@pytest.fixture(autouse=True)
def fixture_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    datasets = kinds["datasets"]
    reals = {name: _real_fixture(name) for name in (*NEW_REAL, *DEV_REAL)}
    fixtures = {DATASET: SealFixtureDataset, **reals}
    patched = {**kinds, "datasets": lambda: {**datasets(), **fixtures}}
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))


def _git(root: Path, *args: str) -> None:
    git = ("git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
           "-c", "commit.gpgsign=false")  # fmt: skip
    subprocess.run([*git, *args], check=True, capture_output=True)


@pytest.fixture
def checkout(config_dir: Path) -> Path:
    """The copied repository committed: a sealed look needs a clean git checkout."""
    root = config_dir.parent
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-qm", "fixture")):
        _git(root, *args)
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
        real: Sequence[str] = ("--no-real",),
    ) -> CliResult:
        command = ("calibrate", "evaluate", "--dataset", DATASET, *real, "--artifact")
        return cli(*FAST, *options, *command, str(artifact), *extra, env=env)

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


REAL_UNSEAL = {"PHYSVERIFY_UNSEAL": ",".join((f"{DATASET}:eval", *NEW_REAL))}
NAMED = tuple(arg for name in NEW_REAL for arg in ("--real-dataset", name))


def test_the_named_real_datasets_enter_the_ledger_line(
    evaluate_cli: Callable[..., CliResult],
    ledger: Path,
    settings: Settings,
    config_dir: Path,
    checkout: Path,
) -> None:
    """``--real-dataset`` replaces ``calibration.real_datasets`` in the
    selections, so the line names the new controls only and its config hash stays the same."""
    sealed = (f"{DATASET}:eval", *NEW_REAL)
    declare(config_dir, SEALED_FILE, "sets", json.dumps(sealed))
    _git(checkout, "commit", "-qam", "new real controls sealed")
    named = evaluate_cli(real=NAMED, env=REAL_UNSEAL)
    assert named.code == 0, named.err
    unnamed = evaluate_cli(real=(), env=REAL_UNSEAL)
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
    monkeypatch.setattr(ConfiguredSuite, "runner", lambda self: _SentinelRunner(stop))

    def invoke(
        *extra: str, options: Sequence[str] = (), env: Mapping[str, str] | None = None
    ) -> CliResult:
        command = ("verify", "--dataset", DATASET, "--level", "oracle", *extra)
        return cli(*FAST, *options, *command, env=env)

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
            return cli(*FAST, *sets, *own, *extra, "selective-risk", leaf, env=env)

        return run

    def diagnose_stages(*extra: str, env: Mapping[str, str] | None = None) -> CliResult:
        monkeypatch.setattr(app_diagnosis, "run_stage_errors", stop)
        return cli(*FAST, *extra, "diagnose", "stages", "--dataset", DATASET, env=env)

    def scene_stats_per_clip(*extra: str, env: Mapping[str, str] | None = None) -> CliResult:
        monkeypatch.setattr(scene_stats, "collect_rows", stop)
        return cli(*FAST, *extra, "scene-stats", "per-clip", "--dataset", DATASET,
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
    monkeypatch.setattr(ConfiguredSuite, "runner", lambda self: _SentinelRunner(stop))
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
    """The calibration.json, exclusions.json and abstention.json artifacts enter the freeze."""
    run = _guarded_verify(cli, monkeypatch, ledger, reached)
    calibration_dir = settings.paths.calibration_dir
    calibration_dir.mkdir(parents=True)
    for name in ("calibration.json", "exclusions.json", "abstention.json"):
        (calibration_dir / name).write_text("{}\n", encoding="utf-8")
    assert run(env=UNSEAL).code == REACHED
    for name in ("calibration.json", "exclusions.json", "abstention.json"):
        (calibration_dir / name).write_text('{"refit": 1}\n', encoding="utf-8")
        moved = run(env=UNSEAL)
        assert moved.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL_REPEAT" in moved.err
        assert run(env={**UNSEAL, "PHYSVERIFY_UNSEAL_REPEAT": "1"}).code == REACHED
    lines = read_entries(ledger)
    assert {"calibration", "exclusions", "abstention"} <= set(lines[-1].artifacts)


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
    write_verify_run(settings, "perceived", perceived)
    stop = sentinel(ledger, reached)
    monkeypatch.setattr(app_calibration, "_raw_runner", lambda session: _SentinelRunner(stop))
    refused = cli(*FAST, "calibrate", "fit", "--dataset", DATASET, "--from-run", "perceived")
    assert refused.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL" in refused.err
    assert reached == []
    looked = cli(*FAST, "calibrate", "fit", "--dataset", DATASET, "--from-run", "perceived",
                 env=UNSEAL)  # fmt: skip
    assert looked.code == REACHED and reached == [1]


SCORE_ALL = "--score-all-clips"


def _versioned(version: str) -> type[SealFixtureDataset]:
    """The fixture as one version of a dataset whose content has versions."""

    class VersionedSeal(SealFixtureDataset):
        def cache_identity(self) -> str:
            return content_identity(version, "fixture")

        def content_version(self) -> str:
            return version

    return VersionedSeal


def _sealed_render(version: str, sealed: str) -> type[SealFixtureDataset]:
    """The fixture as ``version`` of a sealed render whose configured version is ``sealed``."""

    class SealedVersioned(_versioned(version)):
        def sealed_version(self, sections: Mapping[str, Any]) -> str:
            return sealed

    return SealedVersioned


def _register(monkeypatch: pytest.MonkeyPatch, dataset: type[SealFixtureDataset]) -> None:
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {**kinds, "datasets": lambda: {**datasets(), DATASET: dataset}}
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))


@pytest.fixture
def version(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Registers the fixture dataset as ``version`` of a versioned content."""
    return lambda label: _register(monkeypatch, _versioned(label))


def _fitted_on(identity: str | None, tmp_path: Path, artifact_text: str) -> Path:
    data = json.loads(artifact_text)
    if identity is not None:
        data["dataset"]["identity"] = identity
    path = tmp_path / "fitted" / "calibration.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _evaluation(result: CliResult) -> dict[str, Any]:
    path = Path(result.out.split("evaluation: ")[1].split()[0])
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _clean_n(evaluation: Mapping[str, Any]) -> int:
    """Clean clips scored by the checks that score any."""
    return max(check["fpr_clean"]["n"] for check in evaluation["strata"][0]["checks"])


@pytest.fixture
def filed(config_dir: Path, checkout: Path) -> None:
    """``DATASET@v2`` declared a development set by the committed config files."""
    declare(config_dir, CALIBRATION_FILE, "development_sets", f'["{DATASET}@v2"]')
    _git(checkout, "commit", "-qam", "development set")


@pytest.mark.usefixtures("filed")
def test_a_development_set_is_not_a_sealed_look(
    evaluate_cli: Callable[..., CliResult],
    version: Callable[[str], None],
    ledger: Path,
) -> None:
    """Its declared version runs without the variable and writes no line."""
    version("v2")
    result = evaluate_cli()
    assert result.code == 0, result.err
    assert f"development set {DATASET}@v2: not a sealed look" in result.err
    assert read_entries(ledger) == ()


@pytest.mark.usefixtures("filed")
def test_another_version_of_a_development_set_stays_sealed(
    evaluate_cli: Callable[..., CliResult],
    version: Callable[[str], None],
    ledger: Path,
    reached: list[int],
) -> None:
    version("v3")
    result = evaluate_cli()
    assert result.code == EXIT_USAGE and "PHYSVERIFY_UNSEAL" in result.err
    assert read_entries(ledger) == () and reached == []


@pytest.mark.usefixtures("filed")
def test_a_sealed_render_declared_a_development_set_fails_closed(
    evaluate_cli: Callable[..., CliResult],
    monkeypatch: pytest.MonkeyPatch,
    ledger: Path,
    reached: list[int],
) -> None:
    """The configured sealed version collides with a development set: no exemption, no compute."""
    _register(monkeypatch, _sealed_render("v2", sealed="v2"))
    result = evaluate_cli()
    assert result.code == EXIT_USAGE, result.err
    assert f"{DATASET}@v2 is a sealed render and a declared development set" in result.err
    assert read_entries(ledger) == () and reached == []


@pytest.mark.usefixtures("filed")
def test_a_development_version_of_a_sealed_render_is_still_development(
    evaluate_cli: Callable[..., CliResult], monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    """A sealed render of its own version leaves the development reads of the name as they are."""
    _register(monkeypatch, _sealed_render("v2", sealed="v3"))
    result = evaluate_cli()
    assert result.code == 0, result.err
    assert f"development set {DATASET}@v2: not a sealed look" in result.err
    assert read_entries(ledger) == ()


@pytest.mark.parametrize(
    ("options", "env"),
    [
        (("--set", f'calibration.development_sets=["{DATASET}@v2"]'), {}),
        ((), {"PHYSVERIFY__CALIBRATION__DEVELOPMENT_SETS": f'["{DATASET}@v2"]'}),
    ],
    ids=["set", "environment"],
)
def test_a_development_set_declared_outside_the_files_is_refused(
    evaluate_cli: Callable[..., CliResult],
    version: Callable[[str], None],
    ledger: Path,
    reached: list[int],
    options: tuple[str, ...],
    env: dict[str, str],
) -> None:
    """``--set`` and the environment never lift a seal: the loader refuses the key."""
    version("v2")
    result = evaluate_cli(options=options, env=env)
    assert result.code == EXIT_USAGE and "only the committed configs" in result.err
    assert read_entries(ledger) == () and reached == []


@pytest.mark.parametrize(
    ("identity", "development", "message"),
    [
        (content_identity("v2", "fixture"), True, "own fit set"),
        (None, True, "refit it"),
        ("v1:other", False, "not in calibration.development_sets"),
    ],
)
def test_score_all_clips_needs_a_development_set_and_another_fit_set(
    evaluate_cli: Callable[..., CliResult],
    version: Callable[[str], None],
    tmp_path: Path,
    artifact_text: str,
    reached: list[int],
    request: pytest.FixtureRequest,
    identity: str | None,
    development: bool,
    message: str,
) -> None:
    if development:
        request.getfixturevalue("filed")
    version("v2")
    artifact = _fitted_on(identity, tmp_path, artifact_text)
    result = evaluate_cli(SCORE_ALL, artifact=artifact, env=UNSEAL)
    assert result.code == EXIT_USAGE and message in result.err
    assert reached == []


@pytest.mark.usefixtures("filed")
def test_score_all_clips_scores_every_row(
    evaluate_cli: Callable[..., CliResult],
    version: Callable[[str], None],
    tmp_path: Path,
    artifact_text: str,
) -> None:
    """One clean clip per scene: every clean clip with the flag, the eval ones without."""
    version("v2")
    artifact = _fitted_on("v1:other", tmp_path, artifact_text)
    every = _evaluation(evaluate_cli(SCORE_ALL, artifact=artifact))
    assert _clean_n(every) == len(REPORT.clips) // 7
    assert every["scored_splits"] == ["calib", "eval"]
    held = _evaluation(evaluate_cli(artifact=artifact))
    assert _clean_n(held) == len(HELD.clips) // 7 and "scored_splits" not in held


LEGACY_COMMIT = "0" * 40


def _bare_line(ledger: Path, commit: str) -> None:
    """A line written before lines recorded versions: the bare spec, another frozen state."""
    sets = (f"{DATASET}:eval",)
    append_entry(
        ledger,
        LedgerEntry(utc="2026-10-30T16:56:41+00:00", commit=commit, command="calibrate evaluate",
                    argv=(), sets=sets, sealed=sets, datasets=(), artifacts={}, config_hash="old",
                    freeze="old", repeat=False, run_dir=None, user="u", host="h"),
    )  # fmt: skip


@pytest.fixture
def legacy(config_dir: Path, checkout: Path) -> None:
    """The committed mapping reads the fixture's bare lines at ``LEGACY_COMMIT`` as ``v2``."""
    path = config_dir / SEALED_LEGACY_FILE
    table = f'[sealed_legacy.versions.{DATASET}]\nversion = "v2"\ncommit = "{LEGACY_COMMIT}"\n'
    path.write_text(path.read_text(encoding="utf-8") + f'\n{table}reason = "fixture"\n')
    _git(checkout, "commit", "-qam", "legacy mapping")


def test_a_look_records_the_version_it_looked_at(
    evaluate_cli: Callable[..., CliResult],
    version: Callable[[str], None],
    ledger: Path,
    tmp_path: Path,
    artifact_text: str,
) -> None:
    """A new version is unseen until its own line; then a look under another freeze repeats."""
    version("v2")
    assert evaluate_cli(env=UNSEAL).code == 0
    (line,) = read_entries(ledger)
    assert line.sets == (f"{DATASET}:eval@v2",)
    refit = _fitted_on(None, tmp_path, artifact_text)
    again = evaluate_cli(env=UNSEAL, artifact=refit)
    assert again.code == EXIT_USAGE and f"['{DATASET}:eval@v2'] were already" in again.err
    version("v3")
    fresh = evaluate_cli(env=UNSEAL, artifact=refit)
    assert fresh.code == 0, fresh.err
    assert "earlier look" not in fresh.err and not read_entries(ledger)[-1].repeat


@pytest.mark.usefixtures("legacy")
def test_a_legacy_line_blocks_only_its_mapped_version(
    evaluate_cli: Callable[..., CliResult],
    version: Callable[[str], None],
    ledger: Path,
    reached: list[int],
) -> None:
    _bare_line(ledger, LEGACY_COMMIT)
    version("v2")
    refused = evaluate_cli(env=UNSEAL)
    assert refused.code == EXIT_USAGE and f"['{DATASET}:eval@v2'] were already" in refused.err
    assert len(read_entries(ledger)) == 1 and reached == []
    version("v3")
    looked = evaluate_cli(env=UNSEAL)
    assert looked.code == 0, looked.err
    assert read_entries(ledger)[-1].sets == (f"{DATASET}:eval@v3",)


@pytest.mark.usefixtures("legacy")
def test_an_unmapped_legacy_line_fails_closed(
    evaluate_cli: Callable[..., CliResult],
    version: Callable[[str], None],
    ledger: Path,
    reached: list[int],
) -> None:
    """A bare line of a versioned dataset at a commit the mapping does not name: no look."""
    _bare_line(ledger, "1" * 40)
    version("v3")
    result = evaluate_cli(env=UNSEAL)
    assert result.code == 1 and f"names {DATASET}:eval without a version" in result.err
    assert len(read_entries(ledger)) == 1 and reached == []
