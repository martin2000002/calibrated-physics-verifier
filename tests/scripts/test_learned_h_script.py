"""scripts/h200/run_learned_h.sh and run_train_h.sh against a fake ``uv``: the train_h identity they
share, the order of the learned steps and what a dry run stops before."""

from __future__ import annotations

from pathlib import Path

import pytest
from shell_sandbox import Sandbox, executable

SCRIPT_NAMES = ("_common.sh", "_train_h.sh", "run_learned_h.sh", "run_train_h.sh")
TOOLS = ("bash", "dirname", "find", "mkdir", "nproc", "sort", "cat")
SEED = "3590338824"
ENCODER_SHA = "encsha"
# uv run --no-sync python ARGS: an inline program answers the seed or the table's encoder; any
# other call is logged with the dataset settings it sees.
FAKE_UV = rf"""#!/bin/bash
shift 3
if [[ "$1" == -c ]]; then
    [[ "$2" == *labelled_seed* ]] && echo {SEED}
    [[ "$2" == *encoder_sha256* ]] && echo {ENCODER_SHA}
    exit 0
fi
printf '%s|%s|%s|%s\n' "$PHYSVERIFY__DATASET__VERSION" "$PHYSVERIFY__DATASET__SEED" \
    "$PHYSVERIFY__DATASET__CALIB_FRACTION" "$*" >> "$CALLS"
"""
SETTINGS = f"v2-train-h|{SEED}|0.0|"
LEARNED = "tools/train_learned.py --profile precise --set runtime.device=cpu "
SELECT = "--dataset kubric_synthetic --subset train_h"


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    sandbox = Sandbox(tmp_path, SCRIPT_NAMES, TOOLS)
    executable(sandbox.fakes / "uv", FAKE_UV)
    return sandbox


def env(box: Sandbox, **extra: str) -> dict[str, str]:
    return {"PHYSVERIFY_H200_ENV": str(box.root / "no-env.sh"), **extra}


def test_the_dry_run_reads_the_inputs_only(box: Sandbox) -> None:
    out = box.root / "out"
    done = box.call("run_learned_h.sh", str(out), **env(box, PHYSVERIFY_LEARNED_DRY_RUN="1"))
    assert done.returncode == 0, done.stderr
    assert box.lines() == [
        f"{SETTINGS}{LEARNED}images {SELECT} --out {out}/images_plan.json --dry-run"
    ]


def test_training_warms_builds_the_table_then_trains_each_check(box: Sandbox) -> None:
    out = box.root / "out"
    done = box.call("run_learned_h.sh", str(out), "calib.jsonl", "b4.jsonl", **env(box))
    assert done.returncode == 0, done.stderr
    trained = [
        line
        for check in ("R1", "R2")
        for line in (
            (
                f"{SETTINGS}{LEARNED}samples --check {check} {SELECT} --images {out}/pairs.npz "
                f"--out {out}/{check}/samples.npz"
            ),
            (
                f"{SETTINGS}{LEARNED}train --check {check} {SELECT} --samples "
                f"{out}/{check}/samples.npz --encoder-sha {ENCODER_SHA} --evaluated calib.jsonl "
                f"b4.jsonl --out {out}/{check}"
            ),
        )
    ]
    assert box.lines() == [
        f"{SETTINGS}{LEARNED}warm {SELECT}",
        f"{SETTINGS}{LEARNED}images {SELECT} --out {out}/pairs.npz",
        *trained,
    ]


def test_a_filled_cache_skips_warm(box: Sandbox) -> None:
    done = box.call(
        "run_learned_h.sh", str(box.root / "out"), "calib.jsonl",
        **env(box, PHYSVERIFY_LEARNED_SKIP_WARM="1"),
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert not any(" warm " in line for line in box.lines())


def test_training_without_evaluated_manifests_is_refused(box: Sandbox) -> None:
    done = box.call("run_learned_h.sh", str(box.root / "out"), **env(box))
    assert done.returncode == 2
    assert "leakage guard" in done.stderr
    assert box.lines() == []


def test_the_build_checks_seeds_under_the_same_identity(box: Sandbox) -> None:
    data = box.root / "data"
    for version in ("v1-precise", "v2-train-h"):
        (data / "synthetic" / version).mkdir(parents=True)
        (data / "synthetic" / version / "manifest.jsonl").write_text("", encoding="utf-8")
    done = box.call(
        "run_train_h.sh",
        **env(box, PHYSVERIFY__PATHS__DATA_DIR=str(data), PHYSVERIFY_TRAIN_H_DRY_RUN="1"),
    )
    assert done.returncode == 0, done.stderr
    assert box.lines() == [
        (
            f"{SETTINGS}tools/seed_collisions.py --profile precise --subset train_h "
            f"{data}/synthetic/v1-precise/manifest.jsonl"
        )
    ]
