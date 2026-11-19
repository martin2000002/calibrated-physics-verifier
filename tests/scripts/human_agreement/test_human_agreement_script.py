"""scripts/h200/run_human_agreement.sh against a fake interpreter and ``git`` in a copied tree."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from check_rows import UNCACHED, write_check_rows
from shell_sandbox import Sandbox, executable

REPO = Path(__file__).resolve().parents[3]
H200 = REPO / "scripts" / "h200"
SCRIPT_NAMES = (
    "_common.sh", "_steps.sh", "run_human_agreement.sh", "run_query.py", "agreement_query.py",
    "freeze_query.py",
)  # fmt: skip
TOOLS = (
    "awk", "basename", "bash", "cat", "cmp", "cp", "cut", "dirname", "env", "flock", "grep",
    "head", "ln", "mkdir", "mv", "nproc", "realpath", "sed", "sha256sum", "sort", "tee", "touch",
    "tr",
)  # fmt: skip
PREFIX = re.compile(r"^--profile precise --device (cpu|cuda) --set runtime\.threads=(\d+) ")
GPU = "2"
RECORD_SHA = "rec0123"
# pv calls go to the case below; the helpers answer from files (agreement_query) or run for real on
# run-errors unless the run holds the clean rows and REAL_QUERY is unset (its import costs
# seconds). freeze writes a key=value pre-registration, compare appends to the comparison log and
# writes the items.csv of its verify run. With CACHE_GATED set, extract marks CACHE_DIR for its
# subset and the preflight of a subset is uncached until it does.
FAKE_PYTHON = r"""#!/bin/bash
DATA="$PHYSVERIFY__PATHS__DATA_DIR"
CAL="$PHYSVERIFY__PATHS__CALIBRATION_DIR"
PREREG="$DATA/agreement/preregistration.json"
LOG="$DATA/agreement/preregistration.test_comparisons.jsonl"
if [[ "$1" != -c ]]; then
    script="$(basename "$1")"
    printf 'query %s %s\n' "$script" "${*:2}" >> "$CALLS"
    case "$script ${*:2}" in
      "run_query.py run-errors "*)
        [[ -z "${REAL_QUERY:-}" ]] && cmp -s "$CLEAN_CHECKS" "${!#}/checks.jsonl" && exit 0
        PYTHONPATH="$REAL_SRC" exec "$REAL_PYTHON" "$@" ;;
      "agreement_query.py rule") [[ -n "${RULE_EXIT:-}" ]] && exit "$RULE_EXIT"
        echo "${CONFIG_RULE:-calibrated}" ;;
      "agreement_query.py tested")
        if [[ -f "$LOG" ]]; then grep -c . "$LOG" || true; else echo 0; fi ;;
      "agreement_query.py prereg "*) sed -n "s/^$4=//p" "$3" ;;
      "agreement_query.py gaps "*) echo "- no object query: 32 rows (32 with video) of $3" ;;
      "agreement_query.py ready "*) echo v2-ready000001 ;;
      "agreement_query.py log-path") echo "$LOG" ;;
      "freeze_query.py field "*) exec "$REAL_PYTHON" "$@" ;;
      "freeze_query.py artifacts "*) exit "${ARTIFACTS_EXIT:-0}" ;;
    esac
    exit 0
fi
shift 2
printf 'pv %s\n' "$*" >> "$CALLS"
printf 'env cuda=%s|%s|%s|%s|%s|%s|%s|%s|%s\n' "${CUDA_VISIBLE_DEVICES-unset}" \
    "${PHYSVERIFY__PATHS__DATA_DIR-unset}" "${PHYSVERIFY__PATHS__CACHE_DIR-unset}" \
    "${PHYSVERIFY__PATHS__RAW_DIR-unset}" "${PHYSVERIFY__PATHS__CALIBRATION_DIR-unset}" \
    "${PHYSVERIFY__PATHS__ERROR_PROFILE-unset}" "${PHYSVERIFY__PATHS__THRESHOLDS-unset}" \
    "${PHYSVERIFY__CHECK_SCOPE__REQUIRE_DECISION_RECORD-unset}" \
    "${PHYSVERIFY__PATHS__RUNS_DIR-unset}" >> "$CALLS"
RUNS="${PHYSVERIFY__PATHS__RUNS_DIR:-$DATA/runs}"
SUBSET="$(sed -n 's/.* --subset \([^ ]*\).*/\1/p' <<<"$*")"
CACHED="$PHYSVERIFY__PATHS__CACHE_DIR/fake-extracted-$SUBSET"
new_run() {
    local n
    n=$(( $(cat "$RUNS/counter" 2>/dev/null || echo 0) + 1 ))
    mkdir -p "$RUNS/r$n"
    echo "$n" > "$RUNS/counter"
    RUN="$RUNS/r$n"
    echo "run: $RUN"
}
case "$*" in
  *" doctor")
    echo "config_dir: ${DOCTOR_CONFIG_DIR:-$REPO_CONFIGS}"
    echo "device: ${DOCTOR_DEVICE:-$4} (--device)"
    echo "config_hash: ${CONFIG_HASH:-cafe12345678}"
    echo "paths.data_dir: ${DOCTOR_DATA_DIR:-$DATA}"
    echo "paths.runs_dir: ${DOCTOR_RUNS_DIR:-$RUNS}"
    echo "paths.cache_dir: ${DOCTOR_CACHE_DIR:-$PHYSVERIFY__PATHS__CACHE_DIR} (env)"
    echo "paths.raw_dir: ${DOCTOR_RAW_DIR:-$PHYSVERIFY__PATHS__RAW_DIR}"
    echo "paths.calibration_dir: ${DOCTOR_CALIBRATION_DIR:-$CAL}"
    echo "paths.error_profile: ${PROFILE_FILE:-$CAL/stage_errors.precise.json}"
    echo "paths.thresholds: ${THRESHOLDS_FILE:-$CAL/thresholds.json}" ;;
  *" agreement subset --check") exit "${CHECK_EXIT:-0}" ;;
  *" agreement fetch "*) ;;
  *" --clips "*)
    new_run
    checks="${PREFLIGHT_CHECKS:-$CLEAN_CHECKS}"
    [[ -z "${CACHE_GATED:-}" || -f "$CACHED" ]] || checks="$UNCACHED_CHECKS"
    cp "$checks" "$RUN/checks.jsonl"
    exit "${PREFLIGHT_EXIT:-0}" ;;
  *" extract run "*) new_run; mkdir -p "${CACHED%/*}"; touch "$CACHED"; exit "${CLIP_EXIT:-0}" ;;
  *" verify "*) new_run; exit "${CLIP_EXIT:-0}" ;;
  *" agreement freeze "*)
    state="${FROZEN_STATE:-a}"
    if [[ -f "$PREREG" ]] && ! grep -qx "state=$state" "$PREREG"; then
        echo "the frozen state changed" >&2; exit 2
    fi
    if [[ ! -f "$PREREG" ]]; then
        mkdir -p "$DATA/agreement"
        calib="$(sha256sum "$CAL/calibration.json" | cut -d ' ' -f 1)"
        printf '%s\n' "state=$state" "rule=${PREREG_RULE:-calibrated}" \
            "calibration_gate=${PREREG_GATE:-}" "device=${PREREG_DEVICE:-$4}" "profile=precise" \
            "rule_label=calibrated/cal-1" "n_dev=619" "dev_majority=0" "dev_cut=0.42" \
            "dev_tuned_cut=" "calibration_artifact_sha256=${PREREG_CALIB:-$calib}" \
            "exclusion_record_sha256=${PREREG_RECORD:-rec0123}" > "$PREREG"
    fi
    echo "preregistration: $PREREG" ;;
  *" agreement compare "*)
    [[ -z "${COMPARE_EXIT:-}" ]] || exit "$COMPARE_EXIT"
    echo '{"run": "test"}' >> "$LOG"
    report="$(sed -n 's/.* --verify-run \([^ ]*\).*/\1/p' <<<"$*")/agreement"
    mkdir -p "$report"
    echo clip_id > "$report/items.csv"
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


class AgreementSandbox(Sandbox):
    def __init__(self, root: Path) -> None:
        super().__init__(root.resolve(), SCRIPT_NAMES, TOOLS)
        self.out = self.root / "h2"
        self.data = self.root / "h2-data"
        self.chain_dir = self.root / "a8v2"
        self.calibration = self.root / "kubric-data" / "calibration"
        executable(self.scripts / "setup_cuda.sh", FORBIDDEN)
        executable(self.fakes / "python", FAKE_PYTHON)
        executable(self.fakes / "git", FAKE_GIT)
        for name in ("uv", "nvidia-smi"):
            executable(self.fakes / name, FORBIDDEN)
        self.chain_dir.mkdir()
        (self.chain_dir / "state.env").write_text(f"CHAIN_DEVICE=cpu\nRECORD_SHA={RECORD_SHA}\n")
        (self.chain_dir / "steps.tsv").write_text(
            "".join(f"{k}\tstep\t{c}\t1\t-\n" for k in range(12) for c in ("running", "0"))
        )
        self.calibration.mkdir(parents=True)
        for name in ("calibration.json", "exclusions.json", "stage_errors.precise.json"):
            (self.calibration / name).write_text(f"{name}\n")
        self.env.update(
            {
                "REAL_PYTHON": sys.executable,
                "REAL_SRC": str(REPO / "src"),
                "CLEAN_CHECKS": str(write_check_rows(self.root / "clean.jsonl")),
                "PHYSVERIFY_H200_ROOT": str(self.root),
                "PHYSVERIFY_H200_ENV": str(self.root / "no-env.sh"),
                "PHYSVERIFY_AGREEMENT_PYTHON": str(self.fakes / "python"),
                "PHYSVERIFY_AGREEMENT_EXPECT_COMMIT": "c0ffee1",
                "PHYSVERIFY_LOCK_HELD": "1",
                "REPO_CONFIGS": str(self.repo / "configs"),
            }
        )
        self.uncached = write_check_rows(self.root / "uncached.jsonl", error=UNCACHED)
        self.env["UNCACHED_CHECKS"] = str(self.uncached)

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return self.call("run_human_agreement.sh", *args, **env)

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
def sandbox(tmp_path: Path) -> AgreementSandbox:
    return AgreementSandbox(tmp_path)


def _on(device: str) -> dict[str, str]:
    """The environment of a run on DEVICE (cuda with one GPU of 0-3)."""
    gpu = {"CUDA_VISIBLE_DEVICES": GPU} if device == "cuda" else {}
    return {"PHYSVERIFY_AGREEMENT_DEVICE": device, **gpu}


def _chain_on(sandbox: AgreementSandbox, device: str) -> None:
    (sandbox.chain_dir / "state.env").write_text(
        f"CHAIN_DEVICE={device}\nCHAIN_PROFILE=precise\nRECORD_SHA={RECORD_SHA}\n"
    )


def _dev_done(sandbox: AgreementSandbox, **env: str) -> None:
    result = sandbox.run("dev", **env)
    assert result.returncode == 0, result.stderr
    sandbox.calls.unlink()


def _pass(subset: str) -> list[str]:
    return [
        f"agreement fetch --subset {subset}",
        f"extract run --dataset videophy --subset {subset}",
        f"verify --dataset videophy --level perceived --subset {subset}",
    ]


@pytest.mark.parametrize("args", [(), ("all",), ("dev", "test")])
def test_a_missing_or_unknown_phase_stops_before_any_call(sandbox: AgreementSandbox, args) -> None:
    result = sandbox.run(*args)
    assert result.returncode == 2 and "usage: run_human_agreement.sh dev|test" in result.stderr
    assert sandbox.lines() == []


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PHYSVERIFY_AGREEMENT_EXPECT_COMMIT": ""}, "PHYSVERIFY_AGREEMENT_EXPECT_COMMIT"),
        ({"HEAD_SHA": "deadbeef0000"}, "expected c0ffee1"),
        (
            {"DIRTY_FILES": " M src/physverify/human_agreement/report.py\n"},
            "uncommitted or untracked",
        ),
        ({"DIRTY_FILES": "?? annotations/videophy_queries.new.json\n"}, "annotations/videophy"),
        ({"PHYSVERIFY_AGREEMENT_DEVICE": "gpu"}, "PHYSVERIFY_AGREEMENT_DEVICE must be cpu or cuda"),
        ({"PHYSVERIFY_AGREEMENT_DEVICE": "cuda"}, "needs CUDA_VISIBLE_DEVICES in 0-3"),
        (
            {"PHYSVERIFY_AGREEMENT_DEVICE": "cuda", "CUDA_VISIBLE_DEVICES": "5"},
            "needs CUDA_VISIBLE_DEVICES in 0-3",
        ),
        ({"PHYSVERIFY__AGREEMENT__RULE": "provisional_any_fail"}, "PHYSVERIFY__AGREEMENT__RULE"),
        ({"PHYSVERIFY__VERIFICATION__X": "1"}, "PHYSVERIFY__VERIFICATION__X"),
        ({"PHYSVERIFY__PATHS__THRESHOLDS": "/t.json"}, "PHYSVERIFY__PATHS__THRESHOLDS"),
        ({"PHYSVERIFY_AGREEMENT_THREADS": "0"}, "PHYSVERIFY_AGREEMENT_THREADS"),
        ({"PHYSVERIFY_CONFIG_DIR": "/other/configs"}, "PHYSVERIFY_CONFIG_DIR"),
    ],
)
def test_a_startup_guard_stops_before_any_call(
    sandbox: AgreementSandbox, env: dict[str, str], message: str
) -> None:
    for phase in ("dev", "test"):
        result = sandbox.run(phase, **env)
        assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.lines() == []


def test_the_variables_the_script_exports_itself_are_allowed(sandbox: AgreementSandbox) -> None:
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
        (lambda s: _chain_on(s, "cuda"), "never mix devices or profiles"),
        (
            lambda s: (s.chain_dir / "state.env").write_text(
                "CHAIN_DEVICE=cpu\nCHAIN_PROFILE=fast\n"
            ),
            "never mix devices or profiles",
        ),
        (
            lambda s: (s.chain_dir / "state.env").write_text("CHAIN_DEVICE=cuda\n"),
            "(no CHAIN_PROFILE: precise assumed) is of device cuda",
        ),
        (lambda s: (s.chain_dir / "state.env").write_text("CHAIN_DEVICE=cpu\n"), "RECORD_SHA"),
        (lambda s: _append(s.chain_dir / "steps.tsv", "4\tstep\t1\t1\t-\n"), "step 4"),
        (lambda s: _append(s.chain_dir / "steps.tsv", "9\tstep\trunning\t1\t-\n"), "step 9"),
        (lambda s: (s.calibration / "calibration.json").unlink(), "calibration.json"),
        (lambda s: (s.calibration / "exclusions.json").unlink(), "exclusions.json"),
    ],
)
def test_an_unfinished_or_mixed_calibration_chain_stops_before_any_call(
    sandbox: AgreementSandbox, change, message
) -> None:
    change(sandbox)
    result = sandbox.run("dev")
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == []


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"DOCTOR_CONFIG_DIR": "/other/configs"}, "config_dir"),
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
    sandbox: AgreementSandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run("dev", **env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == ["doctor"]


@pytest.mark.parametrize(("device", "gpus"), [("cpu", ""), ("cuda", GPU)])
def test_every_call_runs_on_the_chains_device_through_the_interpreter(
    sandbox: AgreementSandbox, device: str, gpus: str
) -> None:
    _chain_on(sandbox, device)
    _dev_done(sandbox, **_on(device))
    assert sandbox.run("test", **_on(device)).returncode == 0
    lines = sandbox.lines()
    assert not [line for line in lines if line.startswith("forbidden")]
    pv = [PREFIX.match(line[3:]) for line in lines if line.startswith("pv ")]
    assert pv and all(call and call.groups() == (device, "32") for call in pv)
    paths = (sandbox.data, *(sandbox.root / "Repo" / "data" / n for n in ("cache", "raw")))
    expected = f"env cuda={gpus}|{'|'.join(map(str, paths))}|{sandbox.calibration}|unset|unset|true"
    expected += f"|{sandbox.out / 'runs'}"
    envs = [line for line in lines if line.startswith("env ")]
    assert len(envs) == len(pv) and set(envs) == {expected}


def test_the_thread_budget_can_be_set(sandbox: AgreementSandbox) -> None:
    assert sandbox.run("dev", PHYSVERIFY_AGREEMENT_THREADS="8").returncode == 0
    pv = [line[3:] for line in sandbox.lines() if line.startswith("pv ")]
    assert {PREFIX.match(call)[2] for call in pv} == {"8"}


def test_dev_freezes_without_touching_test(sandbox: AgreementSandbox) -> None:
    result = sandbox.run("dev", CLIP_EXIT="1", REAL_QUERY="1")
    assert result.returncode == 0, result.stderr
    assert "run_human_agreement.sh test" in result.stdout
    assert "WARNING" not in result.stderr
    runs = sandbox.out / "runs"
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


def _freeze(sandbox: AgreementSandbox, **change: str) -> dict[str, str]:
    """run_freeze.sh's manifest in place of the calibration chain's run, which is removed."""
    shutil.rmtree(sandbox.chain_dir)
    manifest = sandbox.root / "final" / "freeze.json"
    manifest.parent.mkdir()
    data = {
        "commit": "c0ffee1234567",
        "device": "cpu",
        "profile": "precise",
        "calibration_dir": str(sandbox.calibration),
        "record_sha": RECORD_SHA,
        **change,
    }
    manifest.write_text(json.dumps(data), encoding="utf-8")
    return {"PHYSVERIFY_AGREEMENT_FREEZE": str(manifest)}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_the_freeze_manifest_stands_for_the_calibration_chain(
    sandbox: AgreementSandbox, device: str
) -> None:
    """The manifest's record_sha is the one the pre-registration holds, on the frozen device."""
    env = {**_freeze(sandbox, device=device), **_on(device)}
    _dev_done(sandbox, **env)
    result = sandbox.run("test", **env)
    assert result.returncode == 0, result.stderr
    queries = [line for line in sandbox.lines() if line.startswith("query freeze_query.py")]
    assert f"query freeze_query.py artifacts {env['PHYSVERIFY_AGREEMENT_FREEZE']}" in queries
    assert sandbox.state()["TEST_COMPARED"] == "1"


@pytest.mark.parametrize(
    ("change", "env", "message"),
    [
        ({"device": "cuda"}, {}, "never mix devices or profiles"),
        ({}, _on("cuda"), "never mix devices or profiles"),
        ({"profile": "fast"}, {}, "never mix devices or profiles"),
        ({"commit": "b" * 40}, {}, "froze commit"),
        ({"calibration_dir": "/elsewhere"}, {}, "froze /elsewhere"),
        ({}, {"ARTIFACTS_EXIT": "2"}, "frozen files"),
        ({}, {"PHYSVERIFY_AGREEMENT_FREEZE": "/no/freeze.json"}, "no freeze manifest"),
    ],
)
def test_a_freeze_of_another_state_stops_before_any_call(
    sandbox: AgreementSandbox, change: dict[str, str], env: dict[str, str], message: str
) -> None:
    result = sandbox.run("dev", **{**_freeze(sandbox, **change), **env})
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == []


def test_a_pre_registration_off_the_frozen_record_fails_the_freeze(
    sandbox: AgreementSandbox,
) -> None:
    result = sandbox.run("dev", **_freeze(sandbox, record_sha="other"))
    assert result.returncode == 2 and "not the freeze's other" in result.stderr, result.stderr
    assert sandbox.last_rows()["freeze"] == "2"


def test_dev_refuses_once_a_test_comparison_is_logged(sandbox: AgreementSandbox) -> None:
    sandbox.comparison_log().parent.mkdir(parents=True)
    sandbox.comparison_log().write_text('{"run": "earlier"}\n')
    result = sandbox.run("dev")
    assert result.returncode == 2 and "post hoc" in result.stderr
    assert sandbox.commands() == []


def test_an_uncached_preflight_clip_stops_before_any_extraction(sandbox: AgreementSandbox) -> None:
    result = sandbox.run("dev", PREFLIGHT_CHECKS=str(sandbox.uncached))
    assert result.returncode == 2 and "preflight" in result.stderr
    assert "heavy stage is not cached" in result.stderr
    assert "PHYSVERIFY_AGREEMENT_EXPECT_COMMIT=c0ffee1" in result.stderr
    assert "bash scripts/h200/run_human_agreement.sh prefetch" in result.stderr
    assert not any(" extract run " in call for call in sandbox.commands())
    assert sandbox.last_rows()["pre"] == "2"


def test_a_cuda_prefetch_hint_goes_through_the_gpu_slot(sandbox: AgreementSandbox) -> None:
    """The command to paste never pins a GPU by hand: the GPU slot launcher picks it."""
    _chain_on(sandbox, "cuda")
    result = sandbox.run("dev", PREFLIGHT_CHECKS=str(sandbox.uncached), **_on("cuda"))
    assert result.returncode == 2 and "preflight" in result.stderr, result.stderr
    hint = next(line for line in result.stderr.splitlines() if "prefetch" in line and "cd " in line)
    assert hint.endswith(
        "~/work/scratch/gpu_slot.sh h2-prefetch bash scripts/h200/run_human_agreement.sh prefetch"
    )
    assert "CUDA_VISIBLE_DEVICES" not in hint


def test_an_unrecorded_review_stops_before_any_download(sandbox: AgreementSandbox) -> None:
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
    sandbox: AgreementSandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run("dev", **env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.last_rows()["freeze"] == "2"
    sandbox.calls.unlink()
    assert sandbox.run("test").returncode == 2
    assert sandbox.commands() == []


def test_a_rerun_with_another_frozen_state_stops_at_the_freeze(sandbox: AgreementSandbox) -> None:
    _dev_done(sandbox)
    result = sandbox.run("dev", FROZEN_STATE="b")
    assert result.returncode == 2 and "frozen state changed" in result.stdout
    assert sandbox.last_rows()["freeze"] == "2"


def test_test_without_a_completed_dev_phase_stops_before_any_call(
    sandbox: AgreementSandbox,
) -> None:
    result = sandbox.run("test")
    assert result.returncode == 2 and "run run_human_agreement.sh dev" in result.stderr
    assert sandbox.commands() == []


def test_test_compares_once_with_the_calibrated_rule(sandbox: AgreementSandbox) -> None:
    _dev_done(sandbox)
    result = sandbox.run("test", CLIP_EXIT="1")
    assert result.returncode == 0, result.stderr
    runs = sandbox.out / "runs"
    assert sandbox.commands() == [
        "doctor",
        "agreement subset --check",
        "verify --dataset videophy --level perceived --subset h2_test --clips v2-ready000001",
        *_pass("h2_test"),
        f"agreement compare --verify-run {runs}/r6 --subset h2_test --rule calibrated",
    ]
    assert sandbox.state()["TEST_COMPARED"] == "1"
    link = runs / "h2-latest"
    assert link.is_symlink() and str(link.readlink()) == "r6"
    copied = runs / "r6" / "agreement" / "preregistration.json"
    assert copied.read_bytes() == sandbox.prereg().read_bytes()
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "- H2: supported" in summary and "### Input gaps (h2_test)" in summary


def test_test_compares_with_the_configured_calibrated_rule(sandbox: AgreementSandbox) -> None:
    rule = {"CONFIG_RULE": "calibrated_fail_only", "PREREG_RULE": "calibrated_fail_only"}
    _dev_done(sandbox, **rule)
    result = sandbox.run("test", **rule)
    assert result.returncode == 0, result.stderr
    assert sandbox.commands()[-1].endswith("--subset h2_test --rule calibrated_fail_only")


@pytest.mark.parametrize("phase", ["dev", "test"])
def test_a_config_without_a_calibrated_rule_stops_the_phase(
    sandbox: AgreementSandbox, phase: str
) -> None:
    if phase == "test":
        _dev_done(sandbox)
    result = sandbox.run(phase, RULE_EXIT="2")
    assert result.returncode == 2, result.stderr
    assert not any(c.startswith("agreement compare") for c in sandbox.commands())


def test_a_pre_registration_of_another_calibrated_rule_fails_the_freeze(
    sandbox: AgreementSandbox,
) -> None:
    result = sandbox.run("dev", CONFIG_RULE="calibrated_fail_only")
    assert result.returncode == 2
    assert "rule 'calibrated', not calibrated_fail_only" in result.stderr, result.stderr


def test_the_test_preflight_clip_can_be_given(sandbox: AgreementSandbox) -> None:
    _dev_done(sandbox)
    assert sandbox.run("test", PHYSVERIFY_AGREEMENT_PREFLIGHT_TEST_CLIP="v1-chosen").returncode == 0
    assert any(call.endswith("--clips v1-chosen") for call in sandbox.commands())
    assert not any("ready" in line for line in sandbox.lines())


def test_a_second_test_run_or_a_logged_comparison_stops_before_any_call(
    sandbox: AgreementSandbox,
) -> None:
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


def test_a_compare_that_fails_before_logging_can_be_repeated(sandbox: AgreementSandbox) -> None:
    _dev_done(sandbox)
    failed = sandbox.run("test", COMPARE_EXIT="1")
    assert failed.returncode == 1
    assert "TEST_COMPARED" not in sandbox.state() and not sandbox.comparison_log().exists()
    assert not (sandbox.out / "runs" / "h2-latest").exists()
    again = sandbox.run("test")
    assert again.returncode == 0, again.stderr
    assert sandbox.state()["TEST_COMPARED"] == "1"


def test_a_gate_needs_the_explicit_acceptance(sandbox: AgreementSandbox) -> None:
    gate = {"PREREG_RULE": "provisional_any_fail", "PREREG_GATE": "aggregate weight 0"}
    _dev_done(sandbox, **gate)
    for value in ("", "yes", "true"):
        refused = sandbox.run("test", PHYSVERIFY_AGREEMENT_ACCEPT_NOT_EVALUABLE=value)
        assert refused.returncode == 2 and "not evaluable" in refused.stderr
        assert "post hoc" in refused.stderr
    assert sandbox.commands() == []
    accepted = sandbox.run("test", PHYSVERIFY_AGREEMENT_ACCEPT_NOT_EVALUABLE="1")
    assert accepted.returncode == 0, accepted.stderr
    assert sandbox.commands()[-1].endswith("--subset h2_test --rule calibrated")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s.prereg().write_text(s.prereg().read_text() + "x=1\n"), "changed since"),
        (
            lambda s: (s.calibration / "calibration.json").write_text("v3\n"),
            "the calibration chain's files",
        ),
        (
            lambda s: (s.calibration / "exclusions.json").write_text("v3\n"),
            "the calibration chain's files",
        ),
    ],
)
def test_a_change_after_the_dev_phase_stops_the_test_before_any_call(
    sandbox: AgreementSandbox, change, message: str
) -> None:
    _dev_done(sandbox)
    change(sandbox)
    result = sandbox.run("test")
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == []


def test_another_environment_or_commit_stops_the_test_before_any_call(
    sandbox: AgreementSandbox,
) -> None:
    _dev_done(sandbox)
    moved = sandbox.run("test", PHYSVERIFY_AGREEMENT_RAW_DIR=str(sandbox.root / "raw2"))
    assert moved.returncode == 2 and "environment differs" in moved.stderr
    later = sandbox.run("test", HEAD_SHA="c0ffee1999999")
    assert later.returncode == 2 and "the freeze was at c0ffee1234567" in later.stderr
    assert sandbox.commands() == []


def test_another_config_hash_stops_the_test_after_doctor(sandbox: AgreementSandbox) -> None:
    _dev_done(sandbox)
    result = sandbox.run("test", CONFIG_HASH="beef00000000")
    assert result.returncode == 2 and "config_hash" in result.stderr
    assert sandbox.commands() == ["doctor", "agreement subset --check"]


def test_an_uncommitted_local_config_stops_before_any_call(sandbox: AgreementSandbox) -> None:
    configs = sandbox.root / "repo" / "configs"
    configs.mkdir()
    (configs / "local.toml").write_text("[verification]\n")
    for phase in ("dev", "test"):
        result = sandbox.run(phase)
        assert result.returncode == 2 and "local.toml" in result.stderr
    assert sandbox.lines() == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_moving_the_output_and_data_directories_does_not_unspend_test(
    sandbox: AgreementSandbox, device: str
) -> None:
    _dev_done(sandbox)
    assert sandbox.run("test").returncode == 0
    ledger = sandbox.root / "h2-test-ledger.txt"
    assert ledger.read_text().splitlines() == [str(sandbox.comparison_log())]
    sandbox.calls.unlink()
    _chain_on(sandbox, device)
    moved = {
        "PHYSVERIFY_AGREEMENT_DIR": str(sandbox.root / "h2b"),
        "PHYSVERIFY_AGREEMENT_DATA": str(sandbox.root / "d2"),
        **_on(device),
    }
    dev = sandbox.run("dev", **moved)
    assert dev.returncode == 2 and "post hoc" in dev.stderr and str(ledger) in dev.stderr
    test = sandbox.run("test", **moved)
    assert (
        test.returncode == 2 and "spent" in test.stderr and "if the write completed" in test.stderr
    )
    assert sandbox.commands() == []


def test_dev_warns_when_the_calibration_gate_is_not_null(sandbox: AgreementSandbox) -> None:
    result = sandbox.run("dev", PREREG_RULE="provisional_any_fail", PREREG_GATE="weights 0")
    assert result.returncode == 0, result.stderr
    assert "WARNING: calibration_gate not null (weights 0)" in result.stderr


def test_prefetch_caches_the_dev_split_outside_the_data_dir(sandbox: AgreementSandbox) -> None:
    result = sandbox.run("prefetch")
    assert result.returncode == 0, result.stderr
    assert sandbox.commands() == ["doctor", "agreement subset --check", *_pass("h2_dev")[:2]]
    prefetch = sandbox.out / "prefetch"
    assert (prefetch / "runs" / "r1").is_dir() and not (sandbox.data / "runs").exists()
    assert not (sandbox.out / "steps.tsv").exists() and not sandbox.prereg().exists()
    rows = (prefetch / "steps.tsv").read_text(encoding="utf-8")
    assert "prefetch-extract\tprefetch\t0" in rows
    assert sandbox.run("dev").returncode == 0


def test_prefetch_test_caches_the_test_split_without_spending_it(
    sandbox: AgreementSandbox,
) -> None:
    """Fetch and extract only, in its own book: no verify, compare or ledger line."""
    _dev_done(sandbox)
    uncached = sandbox.run("test", CACHE_GATED="1")
    assert uncached.returncode == 2, uncached.stderr
    assert "bash scripts/h200/run_human_agreement.sh prefetch-test" in uncached.stderr
    sandbox.calls.unlink()
    result = sandbox.run("prefetch-test", CACHE_GATED="1")
    assert result.returncode == 0, result.stderr
    assert sandbox.commands() == ["doctor", "agreement subset --check", *_pass("h2_test")[:2]]
    book = sandbox.out / "prefetch-test"
    assert (book / "runs" / "r1").is_dir()
    assert "prefetch-test-extract\tprefetch-test\t0" in (book / "steps.tsv").read_text()
    assert sandbox.last_rows()["test-pre"] == "2"
    assert not sandbox.comparison_log().exists()
    assert not (sandbox.root / "h2-test-ledger.txt").exists()
    test = sandbox.run("test", CACHE_GATED="1")
    assert test.returncode == 0, test.stderr


def test_prefetch_checks_its_runs_directory_with_doctor(sandbox: AgreementSandbox) -> None:
    result = sandbox.run("prefetch", DOCTOR_RUNS_DIR="/elsewhere")
    assert result.returncode == 2 and "paths.runs_dir" in result.stderr, result.stderr
    assert sandbox.commands() == ["doctor"]


@pytest.mark.parametrize("phase", ["prefetch", "dev", "prefetch-test", "test"])
@pytest.mark.parametrize(
    ("variable", "subdir", "name"),
    [
        ("PHYSVERIFY_AGREEMENT_DIR", "h2", "paths.runs_dir"),
        ("PHYSVERIFY_AGREEMENT_DIR", "", "paths.runs_dir"),
        ("PHYSVERIFY_AGREEMENT_CACHE_DIR", "cache", "paths.cache_dir"),
        ("PHYSVERIFY_AGREEMENT_RAW_DIR", "raw", "paths.raw_dir"),
    ],
)
def test_runs_a_cache_or_raw_videos_under_the_data_directory_are_refused(
    sandbox: AgreementSandbox, phase: str, variable: str, subdir: str, name: str
) -> None:
    result = sandbox.run(phase, **{variable: str(sandbox.data / subdir)})
    assert result.returncode == 2, result.stderr
    assert f"{name} " in result.stderr and "is under the data directory" in result.stderr
    assert variable in result.stderr
    assert sandbox.commands() == []
    assert not sandbox.data.exists()


def test_dev_and_test_runs_default_to_the_output_directory(sandbox: AgreementSandbox) -> None:
    _dev_done(sandbox)
    assert sandbox.run("test").returncode == 0
    runs = sandbox.out / "runs"
    assert sorted(path.name for path in runs.glob("r*")) == [f"r{n}" for n in range(1, 7)]
    assert (runs / "h2-latest").is_symlink()
    assert sorted(path.name for path in sandbox.data.iterdir()) == ["agreement"]


def test_dev_finds_the_cache_prefetch_filled(sandbox: AgreementSandbox) -> None:
    uncached = sandbox.run("dev", CACHE_GATED="1")
    assert uncached.returncode == 2 and "run_human_agreement.sh prefetch" in uncached.stderr
    assert sandbox.run("prefetch", CACHE_GATED="1").returncode == 0
    dev = sandbox.run("dev", CACHE_GATED="1")
    assert dev.returncode == 0, dev.stderr
    caches = {line.split("|")[2] for line in sandbox.lines() if line.startswith("env ")}
    assert caches == {str(sandbox.root / "Repo" / "data" / "cache")}
    assert not (sandbox.data / "runs").exists()


def test_a_freeze_frozen_run_counts_only_freeze_lines_of_the_ledger(
    sandbox: AgreementSandbox,
) -> None:
    """A chain-frozen test look spends chain-frozen runs; a freeze-frozen run is a different
    verifier and is spent by its own freeze line only."""
    _dev_done(sandbox)
    assert sandbox.run("test").returncode == 0
    ledger = sandbox.root / "h2-test-ledger.txt"
    env = {
        **_freeze(sandbox),
        "PHYSVERIFY_AGREEMENT_DIR": str(sandbox.root / "h2v3"),
        "PHYSVERIFY_AGREEMENT_DATA": str(sandbox.root / "d3"),
    }
    _dev_done(sandbox, **env)
    assert sandbox.run("test", **env).returncode == 0
    lines = ledger.read_text().splitlines()
    freeze = Path(env["PHYSVERIFY_AGREEMENT_FREEZE"])
    sha = hashlib.sha256(freeze.read_bytes()).hexdigest()
    assert len(lines) == 2 and lines[1].startswith(f"freeze:{sha}\t")
    again = {
        **env,
        "PHYSVERIFY_AGREEMENT_DIR": str(sandbox.root / "h2v3b"),
        "PHYSVERIFY_AGREEMENT_DATA": str(sandbox.root / "d4"),
    }
    dev = sandbox.run("dev", **again)
    assert dev.returncode == 2 and "post hoc" in dev.stderr


FINAL_SCRIPTS = ("_guards.sh", "final_run.sh")


@pytest.fixture
def final(sandbox: AgreementSandbox) -> Path:
    """final_run.sh beside the runner, its stages 0-6 done and stage 2's freeze in $F with the
    calibration directory $F/calibration. The runner's defaults hold the v1 study (a logged
    comparison in h2-data and its chain line in the ledger) and the chain's calibration; every
    PHYSVERIFY_AGREEMENT_* variable comes from final_run.sh."""
    for name in FINAL_SCRIPTS:
        shutil.copy2(H200 / name, sandbox.scripts / name)
    (sandbox.tools / "find").symlink_to(shutil.which("find") or "find")
    final = sandbox.root / "final-v3"
    calibration = final / "calibration"
    calibration.mkdir(parents=True)
    for name in ("calibration.json", "exclusions.json", "stage_errors.precise.json"):
        (calibration / name).write_text(f"v3 {name}\n")
    freeze = {
        "commit": "c0ffee1234567",
        "device": "cpu",
        "profile": "precise",
        "calibration_dir": str(calibration),
        "record_sha": RECORD_SHA,
    }
    (final / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    (final / "run").mkdir()
    (final / "run" / "steps.tsv").write_text("".join(f"{k}\tstage\t0\t1\t-\n" for k in range(7)))
    saves = ("RUN_COMMIT=c0ffee1234567", "RUN_DEVICE=cpu", "KUBRIC_EVAL=k", "REAL_EVAL=r")
    (final / "run" / "state.env").write_text("\n".join((*saves, "ATTRIBUTION=a", "")))
    sandbox.comparison_log().parent.mkdir(parents=True)
    sandbox.comparison_log().write_text('{"run": "v1"}\n')
    (sandbox.root / "h2-test-ledger.txt").write_text(f"{sandbox.comparison_log()}\n")
    for name in ("PHYSVERIFY_AGREEMENT_PYTHON", "PHYSVERIFY_AGREEMENT_EXPECT_COMMIT"):
        del sandbox.env[name]
    sandbox.env.update(
        {
            "PHYSVERIFY_FINAL_EXPECT_COMMIT": "c0ffee1",
            "PHYSVERIFY_FINAL_CACHE_DIR": str(sandbox.root / "cache"),
            "PHYSVERIFY_FINAL_PYTHON": str(sandbox.fakes / "python"),
            "PV_DEVICE": "cpu",
            "CACHE_GATED": "1",
        }
    )
    return final


def _stage_7(sandbox: AgreementSandbox, **env: str) -> subprocess.CompletedProcess[str]:
    return sandbox.call("final_run.sh", "7", "7", **env)


def _prefetch(subset: str) -> list[str]:
    return ["doctor", "agreement subset --check", *_pass(subset)[:2]]


def test_final_run_stage_7_freezes_on_dev_then_compares_once_on_test(
    sandbox: AgreementSandbox, final: Path
) -> None:
    """Each subset is cached before its preflight, dev freezes before test compares, and every
    call has the freeze's calibration, a new data directory and the run's cache and threads."""
    result = _stage_7(sandbox)
    assert result.returncode == 0, result.stdout + result.stderr
    runs = final / "h2" / "runs"
    preflight = "verify --dataset videophy --level perceived --subset"
    assert sandbox.commands() == [
        *_prefetch("h2_dev"),
        "doctor",
        "agreement subset --check",
        f"{preflight} h2_dev --clips v1-a8208f4dd41e",
        *_pass("h2_dev"),
        f"agreement freeze --verify-run {runs}/r3",
        *_prefetch("h2_test"),
        "doctor",
        "agreement subset --check",
        f"{preflight} h2_test --clips v2-ready000001",
        *_pass("h2_test"),
        f"agreement compare --verify-run {runs}/r6 --subset h2_test --rule calibrated",
    ]
    pv = [PREFIX.match(line[3:]) for line in sandbox.lines() if line.startswith("pv ")]
    assert all(call and call.groups() == ("cpu", "16") for call in pv)
    paths = {tuple(line.split("|")[1:5]) for line in sandbox.lines() if line.startswith("env ")}
    raw = sandbox.root / "Repo" / "data" / "raw"
    data, cache, calibration = final / "h2-data", sandbox.root / "cache", final / "calibration"
    assert paths == {tuple(map(str, (data, cache, raw, calibration)))}
    state = (final / "run" / "state.env").read_text(encoding="utf-8")
    assert f"AGREEMENT_ITEMS={runs}/r6/agreement/items.csv\n" in state
    sha = hashlib.sha256((final / "freeze.json").read_bytes()).hexdigest()
    log = data / "agreement" / "preregistration.test_comparisons.jsonl"
    ledger = (sandbox.root / "h2-test-ledger.txt").read_text().splitlines()
    assert ledger == [str(sandbox.comparison_log()), f"freeze:{sha}\t{log}"]


def test_a_resumed_stage_7_repeats_neither_the_freeze_nor_the_comparison(
    sandbox: AgreementSandbox, final: Path
) -> None:
    failed = _stage_7(sandbox, COMPARE_EXIT="1")
    assert failed.returncode == 1, failed.stdout + failed.stderr
    sandbox.calls.unlink()
    resumed = _stage_7(sandbox)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    commands = sandbox.commands()
    assert commands[: len(_prefetch("h2_test"))] == _prefetch("h2_test")
    assert not [c for c in commands if "h2_dev" in c or c.startswith("agreement freeze")]
    assert commands[-1].startswith("agreement compare")
    sandbox.calls.unlink()
    again = _stage_7(sandbox)
    assert again.returncode == 0, again.stdout + again.stderr
    assert sandbox.commands() == []
    assert "AGREEMENT_ITEMS=" in (final / "run" / "state.env").read_text(encoding="utf-8")
