"""Behavior-equivalence driver: runs one CLI scenario of one checkout, prepares the shared inputs,
or normalizes a finished run directory for diffing.

    driver.py prepare <run_dir>
    driver.py scenario <name> <run_dir>
    driver.py list
    driver.py normalize <run_dir> <out_dir> <recorded_run_dir> <worktree> [commit ...]

`prepare` and `scenario` import physverify from the checkout named by $EQ_WORKTREE (asserted).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import sys
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

CAPTURE = "_capture"
RUN_IDS = "run_ids.txt"
CHAIN_FILE = "chain.json"
RAW_RUN = "raw-oracle"
FIXTURE_DATASET = "synthetic_fixture"

COMMAND_RENAMES = {
    "h3": "compare-generators",
    "abstain": "selective-risk",
    "data": "download",
    "dataset": "synth",
    "devscore": "dev-score",
}
OPTION_RENAMES = {
    "--a8v2-dir": "--chain-dir",
    "--h1-evaluation": "--calibration-evaluation",
    "--h2-run": "--agreement-run",
    "--abstention-run": "--selective-risk-run",
}

RUN_ID = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9a-f]+")
ISO_TS = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:[.,][0-9]+)?(?:Z|[+-][0-9]{2}:?[0-9]{2})?"
)


@dataclass(frozen=True)
class Scenario:
    argv: Callable[[Path], list[str]]
    code: int
    writes: bool


def _chain(run: Path) -> dict[str, str]:
    data: dict[str, str] = json.loads((run / CHAIN_FILE).read_text(encoding="utf-8"))
    return data


def _config(run: Path) -> list[str]:
    return ["--config-dir", str(run / "repo" / "configs")]


def _calib_dataset() -> str:
    return str(importlib.import_module("calibration_cases").DATASET)


def _scenes_dataset() -> str:
    return str(importlib.import_module("check_scenarios").DATASET)


def _verify_scenes(*extra: str) -> Callable[[Path], list[str]]:
    return lambda run: [
        *_config(run),
        "verify",
        "--dataset",
        _scenes_dataset(),
        "--level",
        "oracle",
        *extra,
    ]


SCENARIOS: Mapping[str, Scenario] = {
    "doctor": Scenario(lambda run: [*_config(run), "doctor"], 0, False),
    "doctor_set": Scenario(
        lambda run: [*_config(run), "--set", "runtime.threads=2", "--profile", "precise", "doctor"],
        0,
        False,
    ),
    "verify_oracle": Scenario(
        lambda run: [*_config(run), "verify", "--dataset", FIXTURE_DATASET, "--level", "oracle"],
        0,
        True,
    ),
    "verify_oracle_sweep": Scenario(
        lambda run: [
            *_config(run),
            "verify",
            "--dataset",
            FIXTURE_DATASET,
            "--level",
            "oracle",
            "--sweep",
        ],
        0,
        True,
    ),
    "verify_perceived_raw": Scenario(
        lambda run: [
            *_config(run),
            "verify",
            "--dataset",
            FIXTURE_DATASET,
            "--level",
            "perceived",
            "--policy",
            "raw",
        ],
        0,
        True,
    ),
    "verify_scenes_oracle": Scenario(_verify_scenes(), 0, True),
    "verify_scenes_raw": Scenario(_verify_scenes("--policy", "raw"), 0, True),
    "verify_scenes_sweep": Scenario(_verify_scenes("--sweep"), 0, True),
    "verify_usage_error": Scenario(
        lambda run: [*_config(run), "verify", "--dataset", FIXTURE_DATASET], 2, False
    ),
    "verify_unknown_dataset": Scenario(
        lambda run: [*_config(run), "verify", "--dataset", "nope", "--level", "oracle"], 2, False
    ),
    "dataset_generate_smoke": Scenario(
        lambda run: [*_config(run), "dataset", "generate", "--subset", "smoke"], 0, True
    ),
    "dataset_generate_again": Scenario(
        lambda run: [*_config(run), "dataset", "generate", "--subset", "smoke"], 0, False
    ),
    "dataset_list_smoke": Scenario(
        lambda run: [*_config(run), "dataset", "list", "--subset", "smoke"], 0, False
    ),
    "dataset_list_local": Scenario(
        lambda run: [*_config(run), "dataset", "list", "--subset", "local"], 2, False
    ),
    "results_chain": Scenario(
        lambda run: [
            *_config(run),
            "--profile",
            "fast",
            "results",
            "--a8v2-dir",
            _chain(run)["dir"],
            "--h2-run",
            _chain(run)["h2_run"],
        ],
        0,
        True,
    ),
    "results_chain_without_agreement": Scenario(
        lambda run: [
            *_config(run),
            "--profile",
            "fast",
            "results",
            "--a8v2-dir",
            _chain(run)["dir"],
        ],
        0,
        True,
    ),
    "calibrate_fit": Scenario(
        lambda run: [
            *_config(run),
            "--set",
            "calibration.bootstrap=100",
            "calibrate",
            "fit",
            "--dataset",
            _calib_dataset(),
            "--from-run",
            RAW_RUN,
            "--out",
            str(run / "cal" / "calibration.json"),
        ],
        0,
        True,
    ),
    "calibrate_show": Scenario(
        lambda run: [
            *_config(run),
            "calibrate",
            "show",
            "--artifact",
            str(run / "cal" / "calibration.json"),
        ],
        0,
        False,
    ),
    "calibrate_publish_refused": Scenario(
        lambda run: [
            *_config(run),
            "--set",
            "calibration.require_calibrated_artifact=true",
            "calibrate",
            "publish",
            "--level",
            "oracle",
            "--artifact",
            str(run / "cal" / "calibration.json"),
        ],
        2,
        False,
    ),
    "root_usage_error": Scenario(lambda run: [*_config(run), "no-such-command"], 2, False),
}
HELP = "help"


def _worktree() -> Path:
    return Path(os.environ["EQ_WORKTREE"]).resolve()


def _assert_checkout() -> None:
    import physverify

    location = Path(physverify.__file__).resolve()
    expected = _worktree() / "src"
    if not location.is_relative_to(expected):
        raise SystemExit(f"physverify imported from {location}, not under {expected}")


def _kinds_modules() -> list[Any]:
    found = []
    for name in ("physverify.app.plugins", "physverify.cli.context"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "PLUGIN_KINDS"):
            found.append(module)
    return found


class CalibFixtureDataset:
    """The manifests of the synthetic calibration report; no video and no ground truth."""

    name = ""

    def __init__(self, settings: Any) -> None:
        self._settings = settings

    def clips(self, subset: str | None = None) -> Any:
        return importlib.import_module("calibration_cases").synthetic_report(40).clips

    def video(self, clip: Any) -> Any:
        raise NotImplementedError

    def ground_truth(self, clip: Any) -> None:
        return None

    def fetch(self, subset: str | None, dest: Path) -> None:
        return None


SCENES: Mapping[str, Callable[[Any], Any]] = {
    "box_sphere": lambda sc: sc.box_sphere(),
    "sinking_box": lambda sc: sc.sinking_box(),
    "floating_box": lambda sc: sc.floating_box(),
    "overhang": lambda sc: sc.overhang(0.3),
    "bounce": lambda sc: sc.bounce(e=0.6),
    "bounce_noisy": lambda sc: sc.with_sigma(sc.bounce(), position=0.01),
    "collision_gain": lambda sc: sc.collision(mass_ratio=3.0, gain=0.2),
    "disappearance": lambda sc: sc.disappearance(),
    "resized_pair": lambda sc: sc.resized_pair(),
    "size_noisy": lambda sc: sc.with_sigma(sc.size_change(factor=1.2), size=0.005),
    "teleport": lambda sc: sc.teleport(),
    "teleport_noisy": lambda sc: sc.with_sigma(sc.teleport(), position=0.01),
    "gravity_change": lambda sc: sc.parabola(g_after=1.3 * sc.STANDARD_GRAVITY, change_frame=18),
    "gravity_strong": lambda sc: sc.with_sigma(
        sc.parabola(g=1.2 * sc.STANDARD_GRAVITY), position=0.002
    ),
}
"""Analytic scenes of tests/checks/check_scenarios.py, chosen so that every check R1..R5b gives
a non-zero magnitude on at least one clip (and some a non-zero sigma)."""


def _renamed(case: Any, clip_id: str) -> Any:
    from dataclasses import replace

    from physverify.schema.states import ground_truth_source_key

    key = ground_truth_source_key(clip_id)
    contacts = case.contacts
    if contacts is not None:
        contacts = replace(contacts, clip_id=clip_id, source_key=key)
    clip = replace(case.clip, clip_id=clip_id, video_path=f"{case.clip.dataset}/{clip_id}")
    states = replace(case.states, clip_id=clip_id, source_key=key)
    return replace(case, clip=clip, states=states, contacts=contacts)


class _SceneStore:
    def __init__(self, case: Any, camera: Any) -> None:
        self._case, self._camera = case, camera

    def states(self, frame: Any) -> Any:
        return self._case.states

    def contacts(self, frame: Any) -> Any:
        return self._case.contacts

    def scene(self) -> Any:
        return self._case.scene

    def camera(self, sampling: Any) -> Any:
        return self._camera


class ScenesDataset:
    """The analytic check scenes as a dataset with oracle ground truth (camera frame) only."""

    name = ""

    def __init__(self, settings: Any) -> None:
        sc = importlib.import_module("check_scenarios")
        self._cases = {name: _renamed(build(sc), name) for name, build in SCENES.items()}
        self._pinhole = sc.pinhole

    def clips(self, subset: str | None = None) -> Any:
        return [case.clip for case in self._cases.values()]

    def video(self, clip: Any) -> Any:
        raise NotImplementedError

    def ground_truth(self, clip: Any) -> Any:
        case = self._cases[clip.clip_id]
        return _SceneStore(case, self._pinhole(case.clip))

    def fetch(self, subset: str | None, dest: Path) -> None:
        return None


def _register_fixture_datasets() -> None:
    synthetic = importlib.import_module("synthetic_clip").SyntheticFixtureDataset
    CalibFixtureDataset.name = _calib_dataset()
    ScenesDataset.name = _scenes_dataset()
    extra = {
        synthetic.name: synthetic,
        CalibFixtureDataset.name: CalibFixtureDataset,
        ScenesDataset.name: ScenesDataset,
    }
    for module in _kinds_modules():
        kinds = module.PLUGIN_KINDS
        datasets = kinds["datasets"]
        patched = {**kinds, "datasets": lambda datasets=datasets: {**datasets(), **extra}}
        module.PLUGIN_KINDS = MappingProxyType(patched)


def _commands() -> Mapping[str, Any]:
    for module in _kinds_modules():
        if "commands" in module.PLUGIN_KINDS:
            registry: Mapping[str, Any] = module.PLUGIN_KINDS["commands"]()
            return registry
    from physverify.cli.context import COMMANDS_PACKAGE, check_command, command_name
    from physverify.core.registry import discover_modules

    found: Mapping[str, Any] = discover_modules(COMMANDS_PACKAGE, check_command, command_name)
    return found


def _click_root() -> Any:
    import typer.main

    from physverify.cli.main import build_app

    return typer.main.get_command(build_app(_commands()))


def _map_argv(argv: list[str]) -> list[str]:
    """Old command and option names translated when this checkout only has the new ones."""
    node = _click_root()
    mapped: list[str] = []
    for token in argv:
        commands = getattr(node, "commands", {})
        if (
            token in COMMAND_RENAMES
            and token not in commands
            and COMMAND_RENAMES[token] in commands
        ):
            token = COMMAND_RENAMES[token]
        if token in commands:
            node = commands[token]
        if token in OPTION_RENAMES:
            options = {opt for param in node.params for opt in getattr(param, "opts", ())}
            if token not in options and OPTION_RENAMES[token] in options:
                token = OPTION_RENAMES[token]
        mapped.append(token)
    return mapped


def _help_paths(node: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, ...]]:
    yield path
    for name in sorted(getattr(node, "commands", {})):
        yield from _help_paths(node.commands[name], (*path, name))


def _files(run: Path) -> dict[str, str]:
    return {
        str(path.relative_to(run)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(run.rglob("*"))
        if path.is_file() and CAPTURE not in path.relative_to(run).parts
    }


def _record_run_ids(run: Path, before: set[str], texts: Iterable[str]) -> None:
    ledger = run / CAPTURE / RUN_IDS
    known = ledger.read_text(encoding="utf-8").split() if ledger.exists() else []
    found: list[str] = []
    for text in [*sorted(set(_files(run)) - before), *texts]:
        found += RUN_ID.findall(text)
    new = [run_id for run_id in dict.fromkeys(found) if run_id not in known]
    ledger.write_text("".join(f"{run_id}\n" for run_id in [*known, *new]), encoding="utf-8")


def _invoke(argv: list[str], environ: Mapping[str, str]) -> int:
    from physverify.cli.main import run

    return run(argv, environ)


def run_scenario(name: str, run: Path) -> int:
    _assert_checkout()
    _register_fixture_datasets()
    capture = run / CAPTURE
    capture.mkdir(parents=True, exist_ok=True)
    environ = dict(os.environ)
    before = _files(run)
    if name == HELP:
        failures = 0
        for path in _help_paths(_click_root()):
            argv = [*_config(run), *path, "--help"]
            for stream in (sys.stdout, sys.stderr):
                print(f"=== {' '.join(path) or '<root>'}", file=stream, flush=True)
            code = _invoke(argv, environ)
            sys.stdout.flush()
            sys.stderr.flush()
            failures += code != 0
        (capture / f"{name}.code").write_text("0\n" if not failures else "1\n", encoding="utf-8")
        return 1 if failures else 0
    scenario = SCENARIOS[name]
    argv = _map_argv(scenario.argv(run))
    code = _invoke(argv, environ)
    sys.stdout.flush()
    sys.stderr.flush()
    (capture / f"{name}.code").write_text(f"{code}\n", encoding="utf-8")
    after = _files(run)
    written = sorted(key for key, digest in after.items() if before.get(key) != digest)
    (capture / f"{name}.files").write_text("".join(f"{p}\n" for p in written), encoding="utf-8")
    out_text = _read(capture / f"{name}.out") + _read(capture / f"{name}.err")
    _record_run_ids(run, set(before), [out_text])
    problems = []
    if code != scenario.code:
        problems.append(f"exit code {code}, expected {scenario.code}")
    if scenario.writes and not written:
        problems.append("expected written files, found none")
    if not scenario.writes and not out_text.strip():
        problems.append("expected output, found none")
    if problems:
        print(f"SCENARIO FAILED {name}: {'; '.join(problems)}", file=sys.stderr, flush=True)
        return 1
    return 0


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def prepare(run: Path) -> None:
    """Derived inputs (results chain, raw oracle run), built by each side with its own checkout."""
    _assert_checkout()
    sys.path[:0] = [str(_worktree() / "tests" / d) for d in ("reporting", "calibration")]
    reporting_cases = importlib.import_module("reporting_cases")
    calibration_cases = importlib.import_module("calibration_cases")
    from physverify.verification.runner import write_report

    chain = reporting_cases.build_chain(run / "chain")
    (run / CHAIN_FILE).write_text(
        json.dumps({"dir": str(chain.dir), "h2_run": str(chain.h2_run)}, indent=2) + "\n",
        encoding="utf-8",
    )
    folder = run / "repo" / "data" / "runs" / RAW_RUN
    folder.mkdir(parents=True)
    write_report(folder, calibration_cases.synthetic_report(40))


def _canonical_npz(data: bytes) -> str:
    """An npz is a zip whose entry times change on every write; compare the entries only."""
    import io

    lines = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in sorted(archive.infolist(), key=lambda i: i.filename):
            payload = archive.read(info)
            lines.append(f"{info.filename} {len(payload)} {sha256(payload).hexdigest()}\n")
    return "".join(lines)


class Normalizer:
    def __init__(self, run: Path, recorded: str, worktree: str, commits: Iterable[str]) -> None:
        ledger = run / CAPTURE / RUN_IDS
        ids = ledger.read_text(encoding="utf-8").split() if ledger.exists() else []
        self._ids = {run_id: f"RUNID{index}" for index, run_id in enumerate(ids)}
        self._commits = sorted({c for c in commits if c}, key=len, reverse=True)
        self._paths = {recorded: "RUN", worktree: "WORKTREE"}
        self._digests = self._file_digests(run)

    @staticmethod
    def _file_digests(run: Path) -> dict[str, str]:
        """sha256 of every file the run wrote, so a recorded digest of a file whose bytes hold a
        timestamp reads as SHA(<file>) on both sides."""
        found = {}
        for path in sorted(run.rglob("*")):
            if path.is_file() and CAPTURE not in path.relative_to(run).parts:
                found[sha256(path.read_bytes()).hexdigest()] = str(path.relative_to(run))
        return found

    def text(self, value: str) -> str:
        for digest, name in self._digests.items():
            if digest[:12] in value:
                marker = f"SHA({RUN_ID.sub('RUNID', name)})"
                value = value.replace(digest, marker).replace(digest[:12], marker)
        value = RUN_ID.sub(lambda m: self._ids.get(m.group(0), "RUNID"), value)
        value = ISO_TS.sub("TS", value)
        for commit in self._commits:
            value = value.replace(commit, "COMMIT").replace(commit[:12], "COMMIT")
        for path, marker in self._paths.items():
            value = value.replace(path, marker)
        return value

    def tree(self, source: Path, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target)
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.name == RUN_IDS:
                continue
            relative = self.text(str(path.relative_to(source)))
            data = path.read_bytes()
            if path.suffix == ".npz":
                relative, content = relative + ".entries", _canonical_npz(data).encode()
            else:
                try:
                    content = self.text(data.decode("utf-8")).encode("utf-8")
                except UnicodeDecodeError:
                    content = data
            out = target / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                raise SystemExit(f"normalized name collision: {relative}")
            out.write_bytes(content)


def main(argv: list[str]) -> int:
    match argv:
        case ["prepare", run]:
            prepare(Path(run))
            return 0
        case ["scenario", name, run]:
            return run_scenario(name, Path(run))
        case ["list"]:
            print("\n".join([*SCENARIOS, HELP]))
            return 0
        case ["normalize", run, out, recorded, worktree, *commits]:
            Normalizer(Path(run), recorded, worktree, commits).tree(Path(run), Path(out))
            return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
