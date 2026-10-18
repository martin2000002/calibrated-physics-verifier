"""scripts/h200/run_p1_perception.sh against fake ``uv`` and ``nvidia-smi`` (ADR 0014 §5)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "h200"
SCRIPT_NAMES = ("_common.sh", "run_p1_perception.sh")
TOOLS = (
    "bash", "cat", "cp", "cut", "date", "dirname", "find", "grep", "ln", "mkdir", "nproc",
    "python3", "rm", "sed", "sha256sum", "sort", "tee",
)  # fmt: skip
SUBCOMMANDS = ("extract", "diagnose", "scene-stats", "verify")
THRESHOLDS = '{"checks": {"R5b": {"tau": 0.5}}}\n'
CHECKS_TOML = "[checks]\n"
PRECISE_VIDEO = "synthetic/v1-precise/clips/c1__none/frames"

FAKE_UV = """#!/bin/bash
printf 'uv %s\\n' "$*" >> "$CALLS"
pwd >> "$CWDS"
code=0
case "$*" in
  *" extract run "*) code="${EXTRACT_EXIT:-0}" ;;
  *" diagnose stages "*) code="${STAGES_EXIT:-0}" ;;
  *" verify "*) code="${VERIFY_EXIT:-0}" ;;
esac
case "$*" in
  *" extract run "* | *" diagnose "* | *" verify "*)
    n=$(wc -l < "$CALLS"); mkdir -p "$RUNS/$n"; echo "run: $RUNS/$n" ;;
esac
exit "$code"
"""
FAKE_NVIDIA_SMI = """#!/bin/bash
[[ -z "${NO_GPU:-}" ]] && echo "GPU 0: NVIDIA H200 (UUID: GPU-fake)"
exit 0
"""


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class Sandbox:
    """A copied ``scripts/h200`` whose H200 root, seed data and HF cache are temporary."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.p1 = root / "p1" / "data"
        self.hf = root / "p1" / "hf"
        self.calls = root / "calls.log"
        self.cwds = root / "cwds.log"
        scripts = self.repo / "scripts" / "h200"
        scripts.mkdir(parents=True)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
        (self.repo / "configs").mkdir()
        (self.repo / "configs" / "checks.toml").write_text(CHECKS_TOML, encoding="utf-8")
        for name in SCRIPT_NAMES:
            shutil.copy2(SCRIPTS / name, scripts / name)
        self.seed = root / "Repo" / "data"
        for name in ("synthetic/v1/manifest.jsonl", "weights/w.pt", "vendor/v.py"):
            (self.seed / name).parent.mkdir(parents=True, exist_ok=True)
            (self.seed / name).write_text(name, encoding="utf-8")
        (self.seed / "calibration").mkdir()
        (self.seed / "calibration" / "thresholds.perceived.fast.json").write_text(THRESHOLDS)
        precise = root / "kubric-data" / "synthetic" / "v1-precise"
        (precise / "clips" / "c1__none" / "frames").mkdir(parents=True)
        record = {"data": {"clip_id": "c1__none", "video_path": PRECISE_VIDEO}}
        (precise / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        blob = root / ".cache" / "huggingface" / "hub" / "models--x" / "blob"
        blob.parent.mkdir(parents=True)
        blob.write_text("weights", encoding="utf-8")
        tools = root / "tools"
        tools.mkdir()
        for tool in (*TOOLS, "wc"):
            found = shutil.which(tool)
            assert found is not None, tool
            (tools / tool).symlink_to(found)
        fakes = root / "fakes"
        fakes.mkdir()
        _executable(fakes / "uv", FAKE_UV)
        _executable(fakes / "nvidia-smi", FAKE_NVIDIA_SMI)
        self.bash = tools / "bash"
        self.env = {
            "PATH": f"{fakes}:{tools}",
            "HOME": str(root),
            "CALLS": str(self.calls),
            "CWDS": str(self.cwds),
            "RUNS": str(root / "runs"),
            "PHYSVERIFY_H200_ROOT": str(root),
            "PHYSVERIFY_TIMING_THREADS": "8",
            "CUDA_VISIBLE_DEVICES": "3",
        }

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.bash), str(self.repo / "scripts" / "h200" / "run_p1_perception.sh"), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def lines(self) -> list[str]:
        if not self.calls.exists():
            return []
        return self.calls.read_text(encoding="utf-8").splitlines()

    def global_options(self, label: str, profile: str, *extra: str) -> str:
        sets = [
            "runtime.threads=8",
            f"runtime.heavy_lock={self.p1}/p1.lock",
            f"paths.data_dir={self.p1}",
            f"paths.hf_home={self.hf}",
            f"paths.error_profile={self.p1}/calibration/stage_errors.p1-{label}.json",
            "dataset.version=v1",
            "dataset.resolution=128",
            "dataset.background=plain",
            *extra,
        ]
        return f"--profile {profile} --device cuda " + " ".join(f"--set {s}" for s in sets)


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


SELECTION = "--dataset kubric_synthetic --subset local"


def expected(sandbox: Sandbox, label: str, profile: str, *extra: str) -> list[str]:
    options = sandbox.global_options(label, profile, *extra)
    pv = f"uv run --no-sync physverify {options}"
    out = f"{sandbox.p1}/runs/p1-v1-{label}.json"
    return [
        f"{pv} extract fetch-weights",
        f"{pv} extract run {SELECTION}",
        f"{pv} diagnose stages --write-profile {SELECTION}",
        f"{pv} scene-stats per-clip --label {label} --out {out} {SELECTION}",
        f"{pv} diagnose end-to-end {SELECTION}",
        f"{pv} verify --level perceived {SELECTION}",
    ]


def test_script_parses_and_is_executable() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(SCRIPTS / "run_p1_perception.sh")], check=True)
    assert os.access(SCRIPTS / "run_p1_perception.sh", os.X_OK)


def test_no_gpu_stops_before_uv(sandbox: Sandbox) -> None:
    result = sandbox.run("v1", "A-fast", NO_GPU="1")
    assert result.returncode == 2
    assert sandbox.lines() == []


@pytest.mark.parametrize(
    "args", [(), ("v2",), ("v1", "A-999")], ids=["no-set", "bad-set", "bad-id"]
)
def test_usage_errors_stop_before_uv(sandbox: Sandbox, args: tuple[str, ...]) -> None:
    assert sandbox.run(*args).returncode == 2
    assert sandbox.lines() == []


def test_v1_commands_profiles_and_order(sandbox: Sandbox) -> None:
    result = sandbox.run("v1", "A-fast", "A-280", "B-bj")
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [
        *expected(sandbox, "A-fast", "fast"),
        *expected(
            sandbox, "A-280", "precise", "extraction.providers.depth.vda_small.input_size=280"
        ),
        *expected(
            sandbox,
            "B-bj",
            "fast",
            "extraction.plan.detect=grounding_dino_base",
            "extraction.providers.detect.grounding_dino_base.prompt_mode=joint",
        ),
    ]


def test_every_set_comes_before_the_subcommand(sandbox: Sandbox) -> None:
    assert sandbox.run("v1").returncode == 0
    physverify = [line.split() for line in sandbox.lines() if " physverify " in line]
    assert len(physverify) == 8 * 6
    for words in physverify:
        first = min(words.index(name) for name in SUBCOMMANDS if name in words)
        last_set = max(i for i, word in enumerate(words) if word == "--set")
        assert last_set < first
        assert words[first + 1 :].count("--set") == 0


def test_one_error_profile_per_configuration(sandbox: Sandbox) -> None:
    assert sandbox.run("v1", "A-e2").returncode == 0
    profiles = {
        word
        for line in sandbox.lines()
        for word in line.split()
        if word.startswith("paths.error_profile=")
    }
    assert profiles == {f"paths.error_profile={sandbox.p1}/calibration/stage_errors.p1-A-e2.json"}


def test_failed_clips_are_tolerated(sandbox: Sandbox) -> None:
    result = sandbox.run("v1", "A-fast", EXTRACT_EXIT="1", VERIFY_EXIT="1")
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == expected(sandbox, "A-fast", "fast")
    assert "some clips failed: v1-A-fast extract" in result.stderr


def test_a_usage_error_stops_the_batch(sandbox: Sandbox) -> None:
    result = sandbox.run("v1", "A-fast", "A-518", STAGES_EXIT="2")
    assert result.returncode == 2
    assert sandbox.lines() == expected(sandbox, "A-fast", "fast")[:3]


def test_seeding_pins_the_thresholds_and_copies_the_hub(sandbox: Sandbox) -> None:
    assert sandbox.run("v1", "A-fast").returncode == 0
    assert (sandbox.p1 / "synthetic" / "v1" / "manifest.jsonl").is_file()
    assert (sandbox.p1 / "weights" / "w.pt").is_file()
    assert (sandbox.p1 / "vendor" / "v.py").is_file()
    pinned = sandbox.p1 / "calibration" / "thresholds.json"
    assert pinned.read_text() == THRESHOLDS
    assert (sandbox.hf / "hub" / "models--x" / "blob").read_text() == "weights"
    (sandbox.seed / "calibration" / "thresholds.perceived.fast.json").write_text("{}\n")
    assert sandbox.run("v1", "A-fast").returncode == 0
    assert pinned.read_text() == THRESHOLDS


def test_manifest_records_runs_device_and_hashes(sandbox: Sandbox) -> None:
    assert sandbox.run("v1", "A-fast").returncode == 0
    rows = [
        line.split("\t")
        for line in (sandbox.p1 / "runs" / "p1-manifest.tsv").read_text().splitlines()
    ]
    assert [row[2] for row in rows] == [
        "fetch", "extract", "stages", "collect", "end-to-end", "verify",
    ]  # fmt: skip
    assert {row[1] for row in rows} == {"v1-A-fast"}
    assert {row[4] for row in rows} == {"3"}
    assert {row[6] for row in rows} == {hashlib.sha256(THRESHOLDS.encode()).hexdigest()}
    assert rows[0][3] == "-"
    assert Path(rows[1][3]).is_dir()
    assert (sandbox.p1 / "runs" / "p1-logs" / "v1-A-fast-verify.log").is_file()


def test_roots_can_be_overridden(sandbox: Sandbox, tmp_path: Path) -> None:
    root = tmp_path / "other-repo"
    (root / "src").mkdir(parents=True)
    data, hf = tmp_path / "other-data", tmp_path / "other-hf"
    env = {
        "PHYSVERIFY_ROOT": str(root),
        "PHYSVERIFY_P1_DATA": str(data),
        "PHYSVERIFY_P1_HF": str(hf),
    }
    assert sandbox.run("v1", "A-fast", **env).returncode == 0
    assert set(sandbox.cwds.read_text().splitlines()) == {str(root)}
    assert f"--set paths.data_dir={data} " in sandbox.lines()[0]
    assert f"--set paths.hf_home={hf} " in sandbox.lines()[0]
    assert (data / "calibration" / "thresholds.json").is_file()
    assert not sandbox.p1.exists()


def test_v1_precise_uses_the_precise_dataset_and_listed_clips(sandbox: Sandbox) -> None:
    result = sandbox.run("v1-precise", "A-518", PHYSVERIFY_P1_CLIPS="c1 c2")
    assert result.returncode == 0, result.stderr
    line = sandbox.lines()[1]
    for item in ("dataset.version=v1-precise", "dataset.resolution=256", "dataset.background=hdri"):
        assert f"--set {item} " in line
    assert "paths.synthetic_dir" not in line
    assert line.endswith(
        "extract run --dataset kubric_synthetic --subset full --clips c1 --clips c2"
    )


def test_v1_precise_video_paths_resolve_under_the_p1_data_dir(sandbox: Sandbox) -> None:
    assert sandbox.run("v1-precise", "A-fast").returncode == 0
    manifest = sandbox.p1 / "synthetic" / "v1-precise" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        assert (sandbox.p1 / json.loads(line)["data"]["video_path"]).is_dir()
    assert sandbox.run("v1-precise", "A-fast").returncode == 0


def test_pinned_thresholds_without_checks_stop_the_batch(sandbox: Sandbox) -> None:
    (sandbox.seed / "calibration" / "thresholds.perceived.fast.json").write_text('{"checks": {}}')
    result = sandbox.run("v1", "A-fast")
    assert result.returncode == 2
    assert "PHYSVERIFY_P1_PROVISIONAL=1" in result.stderr
    assert sandbox.lines() == []


def test_provisional_mode_falls_back_and_is_recorded(sandbox: Sandbox) -> None:
    (sandbox.seed / "calibration" / "thresholds.perceived.fast.json").write_text('{"checks": {}}')
    assert sandbox.run("v1", "A-fast", PHYSVERIFY_P1_PROVISIONAL="1").returncode == 0
    fallback = f"--set paths.thresholds={sandbox.p1}/calibration/provisional.absent "
    assert all(fallback in line for line in sandbox.lines() if " physverify " in line)
    rows = (sandbox.p1 / "runs" / "p1-manifest.tsv").read_text().splitlines()
    sha = hashlib.sha256(CHECKS_TOML.encode()).hexdigest()
    assert {row.split("\t")[6] for row in rows} == {f"provisional:{sha}"}
