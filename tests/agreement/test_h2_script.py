"""scripts/h200/run_h2.sh against a fake interpreter and ``git`` in a copied tree (C9, WP-N)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from check_rows import UNCACHED, write_check_rows

REPO = Path(__file__).resolve().parents[2]
H200 = REPO / "scripts" / "h200"
SCRIPT_NAMES = ("_common.sh", "_steps.sh", "run_h2.sh", "a8v2_query.py", "h2_query.py")
TOOLS = (
    "awk", "basename", "bash", "cat", "cmp", "cp", "cut", "dirname", "env", "flock", "grep",
    "head", "ln", "mkdir", "mv", "nproc", "realpath", "sed", "sha256sum", "sort", "tee", "touch",
    "tr",
)  # fmt: skip
PREFIX = re.compile(r"^--profile precise --device cpu --set runtime\.threads=(\d+) ")
RECORD_SHA = "rec0123"
# pv calls go to the case below; the helpers answer from files (h2_query) or run for real on
# run-errors unless the run holds the clean rows and REAL_QUERY is unset (its import costs
# seconds). freeze writes a key=value pre-registration, compare appends to the comparison log.
FAKE_PYTHON = r"""#!/bin/bash
DATA="$PHYSVERIFY__PATHS__DATA_DIR"
CAL="$PHYSVERIFY__PATHS__CALIBRATION_DIR"
PREREG="$DATA/agreement/preregistration.json"
LOG="$DATA/agreement/preregistration.test_comparisons.jsonl"
if [[ "$1" != -c ]]; then
    script="$(basename "$1")"
    printf 'query %s %s\n' "$script" "${*:2}" >> "$CALLS"
    case "$script ${*:2}" in
      "a8v2_query.py run-errors "*)
        [[ -z "${REAL_QUERY:-}" ]] && cmp -s "$CLEAN_CHECKS" "${!#}/checks.jsonl" && exit 0
        PYTHONPATH="$REAL_SRC" exec "$REAL_PYTHON" "$@" ;;
      "h2_query.py tested")
        if [[ -f "$LOG" ]]; then grep -c . "$LOG" || true; else echo 0; fi ;;
      "h2_query.py prereg "*) sed -n "s/^$4=//p" "$3" ;;
      "h2_query.py gaps "*) echo "- no object query: 32 rows (32 with video) of $3" ;;
      "h2_query.py ready "*) echo v2-ready000001 ;;
    esac
    exit 0
fi
shift 2
printf 'pv %s\n' "$*" >> "$CALLS"
printf 'env cuda=%s|%s|%s|%s|%s|%s|%s|%s\n' "${CUDA_VISIBLE_DEVICES-unset}" \
    "${PHYSVERIFY__PATHS__DATA_DIR-unset}" "${PHYSVERIFY__PATHS__CACHE_DIR-unset}" \
    "${PHYSVERIFY__PATHS__RAW_DIR-unset}" "${PHYSVERIFY__PATHS__CALIBRATION_DIR-unset}" \
    "${PHYSVERIFY__PATHS__ERROR_PROFILE-unset}" "${PHYSVERIFY__PATHS__THRESHOLDS-unset}" \
    "${PHYSVERIFY__CHECK_SCOPE__REQUIRE_DECISION_RECORD-unset}" >> "$CALLS"
new_run() {
    local n
    n=$(( $(cat "$DATA/counter" 2>/dev/null || echo 0) + 1 ))
    mkdir -p "$DATA/runs/r$n"
    echo "$n" > "$DATA/counter"
    RUN="$DATA/runs/r$n"
    echo "run: $RUN"
}
case "$*" in
  *" doctor")
    echo "device: ${DOCTOR_DEVICE:-cpu} (--device)"
    echo "config_hash: ${CONFIG_HASH:-cafe12345678}"
    echo "paths.data_dir: ${DOCTOR_DATA_DIR:-$DATA}"
    echo "paths.cache_dir: ${DOCTOR_CACHE_DIR:-$PHYSVERIFY__PATHS__CACHE_DIR} (env)"
    echo "paths.raw_dir: ${DOCTOR_RAW_DIR:-$PHYSVERIFY__PATHS__RAW_DIR}"
    echo "paths.calibration_dir: ${DOCTOR_CALIBRATION_DIR:-$CAL}"
    echo "paths.error_profile: ${PROFILE_FILE:-$CAL/stage_errors.precise.json}"
    echo "paths.thresholds: ${THRESHOLDS_FILE:-$CAL/thresholds.json}" ;;
  *" agreement subset --check") exit "${CHECK_EXIT:-0}" ;;
  *" agreement fetch "*) ;;
  *" --clips "*)
    new_run
    cp "${PREFLIGHT_CHECKS:-$CLEAN_CHECKS}" "$RUN/checks.jsonl"
    exit "${PREFLIGHT_EXIT:-0}" ;;
  *" extract run "*|*" verify "*) new_run; exit "${CLIP_EXIT:-0}" ;;
  *" agreement freeze "*)
    state="${FROZEN_STATE:-a}"
    if [[ -f "$PREREG" ]] && ! grep -qx "state=$state" "$PREREG"; then
        echo "the frozen state changed" >&2; exit 2
    fi
    if [[ ! -f "$PREREG" ]]; then
        mkdir -p "$DATA/agreement"
        calib="$(sha256sum "$CAL/calibration.json" | cut -d ' ' -f 1)"
        printf '%s\n' "state=$state" "rule=${PREREG_RULE:-calibrated}" \
            "calibration_gate=${PREREG_GATE:-}" "device=${PREREG_DEVICE:-cpu}" "profile=precise" \
            "rule_label=calibrated/cal-1" "n_dev=619" "dev_majority=0" "dev_cut=0.42" \
            "dev_tuned_cut=" "calibration_artifact_sha256=${PREREG_CALIB:-$calib}" \
            "exclusion_record_sha256=${PREREG_RECORD:-rec0123}" > "$PREREG"
    fi
    echo "preregistration: $PREREG" ;;
  *" agreement compare "*)
    [[ -z "${COMPARE_EXIT:-}" ]] || exit "$COMPARE_EXIT"
    echo '{"run": "test"}' >> "$LOG"
    echo "H2: supported" ;;
esac
exit 0
"""
FAKE_GIT = """#!/bin/bash
case "$*" in
  *" rev-parse HEAD") echo "${HEAD_SHA:-c0ffee1234567}" ;;
  *" status --porcelain "*) printf '%s' "${DIRTY_FILES:-}" ;;
esac
"""
FORBIDDEN = '#!/bin/bash\necho "forbidden $(basename "$0") $*" >> "$CALLS"\n'


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.calls = self.root / "calls.log"
        self.out = self.root / "h2"
        self.data = self.root / "h2-data"
        self.a8v2 = self.root / "a8v2"
        self.calibration = self.root / "kubric-data" / "calibration"
        scripts = self.root / "repo" / "scripts" / "h200"
        scripts.mkdir(parents=True)
        self.script = scripts / "run_h2.sh"
        for name in SCRIPT_NAMES:
            shutil.copy2(H200 / name, scripts / name)
        _executable(scripts / "setup.sh", FORBIDDEN)
        tools, fakes = self.root / "tools", self.root / "fakes"
        tools.mkdir()
        fakes.mkdir()
        for tool in TOOLS:
            found = shutil.which(tool)
            assert found is not None, tool
            (tools / tool).symlink_to(found)
        _executable(fakes / "python", FAKE_PYTHON)
        _executable(fakes / "git", FAKE_GIT)
        for name in ("uv", "nvidia-smi"):
            _executable(fakes / name, FORBIDDEN)
        self.a8v2.mkdir()
        (self.a8v2 / "state.env").write_text(f"CHAIN_DEVICE=cpu\nRECORD_SHA={RECORD_SHA}\n")
        (self.a8v2 / "steps.tsv").write_text(
            "".join(f"{k}\tstep\t{c}\t1\t-\n" for k in range(12) for c in ("running", "0"))
        )
        self.calibration.mkdir(parents=True)
        for name in ("calibration.json", "exclusions.json", "stage_errors.precise.json"):
            (self.calibration / name).write_text(f"{name}\n")
        self.env = {
            "PATH": f"{fakes}:{tools}",
            "HOME": str(self.root),
            "CALLS": str(self.calls),
            "REAL_PYTHON": sys.executable,
            "REAL_SRC": str(REPO / "src"),
            "CLEAN_CHECKS": str(write_check_rows(self.root / "clean.jsonl")),
            "PHYSVERIFY_H200_ROOT": str(self.root),
            "PHYSVERIFY_H200_ENV": str(self.root / "no-env.sh"),
            "PHYSVERIFY_H2_PYTHON": str(fakes / "python"),
            "PHYSVERIFY_H2_EXPECT_COMMIT": "c0ffee1",
            "PHYSVERIFY_LOCK_HELD": "1",
        }
        self.uncached = write_check_rows(self.root / "uncached.jsonl", error=UNCACHED)

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.root / "tools" / "bash"), str(self.script), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def lines(self) -> list[str]:
        return self.calls.read_text(encoding="utf-8").splitlines() if self.calls.exists() else []

    def commands(self) -> list[str]:
        """The pv calls without their common prefix."""
        return [PREFIX.sub("", line[3:]) for line in self.lines() if line.startswith("pv ")]

    def state(self) -> dict[str, str]:
        text = (self.out / "state.env").read_text(encoding="utf-8")
        return dict(line.split("=", 1) for line in text.splitlines())

    def last_rows(self) -> dict[str, str]:
        text = (self.out / "steps.tsv").read_text(encoding="utf-8")
        return {row[0]: row[2] for row in (line.split("\t") for line in text.splitlines())}

    def prereg(self) -> Path:
        return self.data / "agreement" / "preregistration.json"

    def comparison_log(self) -> Path:
        return self.data / "agreement" / "preregistration.test_comparisons.jsonl"


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def _dev_done(sandbox: Sandbox, **env: str) -> None:
    result = sandbox.run("dev", **env)
    assert result.returncode == 0, result.stderr
    sandbox.calls.unlink()


def _pass(subset: str) -> list[str]:
    return [
        f"agreement fetch --subset {subset}",
        f"extract run --dataset videophy --subset {subset}",
        f"verify --dataset videophy --level perceived --subset {subset}",
    ]


def test_the_scripts_parse_and_run_h2_is_executable() -> None:
    for name in ("run_h2.sh", "_steps.sh"):
        subprocess.run(["bash", "-n", str(H200 / name)], check=True)
    assert os.access(H200 / "run_h2.sh", os.X_OK)


@pytest.mark.parametrize("args", [(), ("all",), ("dev", "test")])
def test_a_missing_or_unknown_phase_stops_before_any_call(sandbox: Sandbox, args) -> None:
    result = sandbox.run(*args)
    assert result.returncode == 2 and "usage: run_h2.sh dev|test" in result.stderr
    assert sandbox.lines() == []


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PHYSVERIFY_H2_EXPECT_COMMIT": ""}, "PHYSVERIFY_H2_EXPECT_COMMIT"),
        ({"HEAD_SHA": "deadbeef0000"}, "expected c0ffee1"),
        ({"DIRTY_FILES": " M src/physverify/agreement/report.py\n"}, "uncommitted or untracked"),
        ({"DIRTY_FILES": "?? annotations/videophy_queries.new.json\n"}, "annotations/videophy"),
        ({"PHYSVERIFY_H2_DEVICE": "cuda"}, "PHYSVERIFY_H2_DEVICE must be cpu"),
        ({"PHYSVERIFY__AGREEMENT__RULE": "provisional_any_fail"}, "PHYSVERIFY__AGREEMENT__RULE"),
        ({"PHYSVERIFY__VERIFICATION__X": "1"}, "PHYSVERIFY__VERIFICATION__X"),
        ({"PHYSVERIFY__PATHS__THRESHOLDS": "/t.json"}, "PHYSVERIFY__PATHS__THRESHOLDS"),
        ({"PHYSVERIFY_H2_THREADS": "0"}, "PHYSVERIFY_H2_THREADS"),
    ],
)
def test_a_startup_guard_stops_before_any_call(
    sandbox: Sandbox, env: dict[str, str], message: str
) -> None:
    for phase in ("dev", "test"):
        result = sandbox.run(phase, **env)
        assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.lines() == []


def test_the_variables_the_script_exports_itself_are_allowed(sandbox: Sandbox) -> None:
    result = sandbox.run(
        "dev", PHYSVERIFY__PATHS__HF_HOME="/elsewhere", PHYSVERIFY__PATHS__DATA_DIR="/x"
    )
    assert result.returncode == 0, result.stderr
    assert all(f"|{sandbox.data}|" in line for line in sandbox.lines() if line.startswith("env "))


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda s: (s.a8v2 / "state.env").write_text("CHAIN_DEVICE=cuda\nRECORD_SHA=r\n"),
            "cpu chain",
        ),
        (lambda s: (s.a8v2 / "state.env").write_text("CHAIN_DEVICE=cpu\n"), "RECORD_SHA"),
        (lambda s: _append(s.a8v2 / "steps.tsv", "4\tstep\t1\t1\t-\n"), "step 4"),
        (lambda s: _append(s.a8v2 / "steps.tsv", "9\tstep\trunning\t1\t-\n"), "step 9"),
        (lambda s: (s.calibration / "calibration.json").unlink(), "calibration.json"),
        (lambda s: (s.calibration / "exclusions.json").unlink(), "exclusions.json"),
    ],
)
def test_an_unfinished_a8v2_cpu_chain_stops_before_any_call(
    sandbox: Sandbox, change, message
) -> None:
    change(sandbox)
    result = sandbox.run("dev")
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == []


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"DOCTOR_DEVICE": "cuda"}, "device 'cuda'"),
        ({"DOCTOR_DATA_DIR": "/other/data"}, "paths.data_dir"),
        ({"DOCTOR_CACHE_DIR": "/other/cache"}, "paths.cache_dir"),
        ({"DOCTOR_RAW_DIR": "/other/raw"}, "paths.raw_dir"),
        ({"DOCTOR_CALIBRATION_DIR": "/other/calibration"}, "paths.calibration_dir"),
        ({"PROFILE_FILE": "/c/stage_errors.recalibrated.json"}, "paths.error_profile"),
        ({"THRESHOLDS_FILE": "/other/thresholds.json"}, "paths.thresholds"),
    ],
)
def test_doctor_must_report_the_exported_cpu_configuration(
    sandbox: Sandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run("dev", **env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == ["doctor"]


def test_every_call_runs_on_the_cpu_through_the_interpreter(sandbox: Sandbox) -> None:
    _dev_done(sandbox)
    assert sandbox.run("test").returncode == 0
    lines = sandbox.lines()
    assert not [line for line in lines if line.startswith("forbidden")]
    pv = [line[3:] for line in lines if line.startswith("pv ")]
    assert pv and all(PREFIX.match(call) and PREFIX.match(call)[1] == "32" for call in pv)
    paths = (sandbox.data, *(sandbox.root / "Repo" / "data" / n for n in ("cache", "raw")))
    expected = f"env cuda=|{'|'.join(map(str, paths))}|{sandbox.calibration}|unset|unset|true"
    envs = [line for line in lines if line.startswith("env ")]
    assert len(envs) == len(pv) and set(envs) == {expected}


def test_the_thread_budget_can_be_set(sandbox: Sandbox) -> None:
    assert sandbox.run("dev", PHYSVERIFY_H2_THREADS="8").returncode == 0
    pv = [line[3:] for line in sandbox.lines() if line.startswith("pv ")]
    assert {PREFIX.match(call)[1] for call in pv} == {"8"}


def test_dev_freezes_without_touching_test(sandbox: Sandbox) -> None:
    result = sandbox.run("dev", CLIP_EXIT="1", REAL_QUERY="1")
    assert result.returncode == 0, result.stderr
    assert "run_h2.sh test" in result.stdout
    runs = sandbox.data / "runs"
    assert sandbox.commands() == [
        "doctor",
        "agreement subset --check",
        "verify --dataset videophy --level perceived --subset h2_dev --clips v1-a8208f4dd41e",
        *_pass("h2_dev"),
        f"agreement freeze --verify-run {runs}/r3",
    ]
    assert not any("h2_test" in line for line in sandbox.lines())
    state = sandbox.state()
    keys = {"DEV_VERIFY_RUN", "PREREG", "PREREG_SHA", "CALIB_SHA", "RECORD_FILE_SHA"}
    keys |= {"PROFILE_SHA", "FROZEN_COMMIT", "DEV_ENV_SHA", "DEV_CONFIG_HASH"}
    assert keys <= set(state)
    assert state["FROZEN_COMMIT"] == "c0ffee1234567" and state["PREREG"] == str(sandbox.prereg())
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    for needle in (
        "- rule: calibrated",
        "- calibration_gate: \n",
        "- dev_cut: 0.42",
        f"- exclusion_record_sha256: {RECORD_SHA}",
        "### Input gaps (h2_dev)",
        "- no object query: 32 rows (32 with video) of h2_dev",
        "some clips failed (exit 1)",
    ):
        assert needle in summary, needle
    assert sandbox.last_rows()["freeze"] == "0"


def test_dev_refuses_once_a_test_comparison_is_logged(sandbox: Sandbox) -> None:
    sandbox.comparison_log().parent.mkdir(parents=True)
    sandbox.comparison_log().write_text('{"run": "earlier"}\n')
    result = sandbox.run("dev")
    assert result.returncode == 2 and "post hoc" in result.stderr
    assert sandbox.commands() == []


def test_an_uncached_preflight_clip_stops_before_any_extraction(sandbox: Sandbox) -> None:
    result = sandbox.run("dev", PREFLIGHT_CHECKS=str(sandbox.uncached))
    assert result.returncode == 2 and "preflight" in result.stderr
    assert "heavy stage is not cached" in result.stderr
    assert not any(" extract run " in call for call in sandbox.commands())
    assert sandbox.last_rows()["pre"] == "2"


def test_an_unrecorded_review_stops_before_any_download(sandbox: Sandbox) -> None:
    result = sandbox.run("dev", CHECK_EXIT="1")
    assert result.returncode == 2 and "caption review" in result.stderr
    assert not any("fetch" in call for call in sandbox.commands())


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PREREG_RULE": "provisional_any_fail"}, "not calibrated"),
        ({"PREREG_RULE": "calibrated", "PREREG_GATE": "weights 0"}, "with a calibration gate"),
        ({"PREREG_DEVICE": "cuda"}, "device 'cuda'"),
        ({"PREREG_CALIB": "0" * 64}, "calibration artifact"),
        ({"PREREG_RECORD": "other"}, "decision record other"),
    ],
)
def test_a_pre_registration_that_is_not_h2s_fails_the_freeze(
    sandbox: Sandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run("dev", **env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.last_rows()["freeze"] == "2"
    sandbox.calls.unlink()
    assert sandbox.run("test").returncode == 2
    assert sandbox.commands() == []


def test_a_rerun_with_another_frozen_state_stops_at_the_freeze(sandbox: Sandbox) -> None:
    _dev_done(sandbox)
    result = sandbox.run("dev", FROZEN_STATE="b")
    assert result.returncode == 2 and "frozen state changed" in result.stdout
    assert sandbox.last_rows()["freeze"] == "2"


def test_test_without_a_completed_dev_phase_stops_before_any_call(sandbox: Sandbox) -> None:
    result = sandbox.run("test")
    assert result.returncode == 2 and "run run_h2.sh dev" in result.stderr
    assert sandbox.commands() == []


def test_test_compares_once_with_the_calibrated_rule(sandbox: Sandbox) -> None:
    _dev_done(sandbox)
    result = sandbox.run("test", CLIP_EXIT="1")
    assert result.returncode == 0, result.stderr
    runs = sandbox.data / "runs"
    assert sandbox.commands() == [
        "doctor",
        "agreement subset --check",
        "verify --dataset videophy --level perceived --subset h2_test --clips v2-ready000001",
        *_pass("h2_test"),
        f"agreement compare --verify-run {runs}/r6 --subset h2_test --rule calibrated",
    ]
    assert sandbox.state()["TEST_COMPARED"] == "1"
    link = runs / "h2-latest"
    assert link.is_symlink() and os.readlink(link) == "r6"
    copied = runs / "r6" / "agreement" / "preregistration.json"
    assert copied.read_bytes() == sandbox.prereg().read_bytes()
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "- H2: supported" in summary and "### Input gaps (h2_test)" in summary


def test_the_test_preflight_clip_can_be_given(sandbox: Sandbox) -> None:
    _dev_done(sandbox)
    assert sandbox.run("test", PHYSVERIFY_H2_PREFLIGHT_TEST_CLIP="v1-chosen").returncode == 0
    assert any(call.endswith("--clips v1-chosen") for call in sandbox.commands())
    assert not any("ready" in line for line in sandbox.lines())


def test_a_second_test_run_or_a_logged_comparison_stops_before_any_call(sandbox: Sandbox) -> None:
    _dev_done(sandbox)
    assert sandbox.run("test").returncode == 0
    sandbox.calls.unlink()
    again = sandbox.run("test")
    assert again.returncode == 2 and "spent" in again.stderr and "TEST_COMPARED" in again.stderr
    assert sandbox.commands() == []
    state = (sandbox.out / "state.env").read_text(encoding="utf-8")
    (sandbox.out / "state.env").write_text(state.replace("TEST_COMPARED=1\n", ""))
    logged = sandbox.run("test")
    assert logged.returncode == 2 and "spent" in logged.stderr
    assert sandbox.commands() == []


def test_a_compare_that_fails_before_logging_can_be_repeated(sandbox: Sandbox) -> None:
    _dev_done(sandbox)
    failed = sandbox.run("test", COMPARE_EXIT="1")
    assert failed.returncode == 1
    assert "TEST_COMPARED" not in sandbox.state() and not sandbox.comparison_log().exists()
    assert not (sandbox.data / "runs" / "h2-latest").exists()
    again = sandbox.run("test")
    assert again.returncode == 0, again.stderr
    assert sandbox.state()["TEST_COMPARED"] == "1"


def test_a_gate_needs_the_explicit_acceptance(sandbox: Sandbox) -> None:
    gate = {"PREREG_RULE": "provisional_any_fail", "PREREG_GATE": "aggregate weight 0"}
    _dev_done(sandbox, **gate)
    for value in ("", "yes", "true"):
        refused = sandbox.run("test", PHYSVERIFY_H2_ACCEPT_NOT_EVALUABLE=value)
        assert refused.returncode == 2 and "not evaluable" in refused.stderr
        assert "post hoc" in refused.stderr
    assert sandbox.commands() == []
    accepted = sandbox.run("test", PHYSVERIFY_H2_ACCEPT_NOT_EVALUABLE="1")
    assert accepted.returncode == 0, accepted.stderr
    assert sandbox.commands()[-1].endswith("--subset h2_test --rule calibrated")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s.prereg().write_text(s.prereg().read_text() + "x=1\n"), "changed since"),
        (lambda s: (s.calibration / "calibration.json").write_text("v3\n"), "A8 v2 files"),
        (lambda s: (s.calibration / "exclusions.json").write_text("v3\n"), "A8 v2 files"),
    ],
)
def test_a_change_after_the_dev_phase_stops_the_test_before_any_call(
    sandbox: Sandbox, change, message: str
) -> None:
    _dev_done(sandbox)
    change(sandbox)
    result = sandbox.run("test")
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == []


def test_another_environment_or_commit_stops_the_test_before_any_call(sandbox: Sandbox) -> None:
    _dev_done(sandbox)
    moved = sandbox.run("test", PHYSVERIFY_H2_RAW_DIR=str(sandbox.root / "raw2"))
    assert moved.returncode == 2 and "environment differs" in moved.stderr
    later = sandbox.run("test", HEAD_SHA="c0ffee1999999")
    assert later.returncode == 2 and "the freeze was at c0ffee1234567" in later.stderr
    assert sandbox.commands() == []


def test_another_config_hash_stops_the_test_after_doctor(sandbox: Sandbox) -> None:
    _dev_done(sandbox)
    result = sandbox.run("test", CONFIG_HASH="beef00000000")
    assert result.returncode == 2 and "config_hash" in result.stderr
    assert sandbox.commands() == ["doctor", "agreement subset --check"]
