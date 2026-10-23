"""Script test fixtures: no network, the timing settings and clip factory, and directories of
run_iteration.sh as freeze_query.py select reads them."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from network import no_network as no_network
from timing_settings import clip_factory as clip_factory
from timing_settings import data_dir as data_dir
from timing_settings import make_settings as make_settings
from timing_settings import timing_environ as timing_environ

ITERATION_FILES = {
    "DEVSCORE_SHA": "devscore.json",
    "CALIBRATION_SHA": "calibration.json",
    "PROFILE_SHA": "calibration/stage_errors.precise.json",
    "STAGE_SIGMA_SHA": "calibration/stage_sigma.json",
}


def make_iteration(
    root: Path,
    name: str,
    n_met: int,
    auroc: float | None,
    *,
    device: str = "cpu",
    steps: Sequence[int] = (1, 2, 3, 4, 5),
    commit: str = "c" * 40,
) -> Path:
    """A run_iteration.sh directory: state.env written with printf %q, steps.tsv, the files."""
    directory = root / f"{name} {device}"
    (directory / "calibration").mkdir(parents=True)
    devscore = {"commit": commit, "selection": {"n_met": n_met, "tiebreak_auroc_cv": auroc}}
    contents = {
        "devscore.json": json.dumps(devscore).encode(),
        "calibration.json": f"calibration of {name}".encode(),
        "calibration/stage_errors.precise.json": f"profile of {name}".encode(),
        "calibration/stage_sigma.json": f"sigma of {name}".encode(),
    }
    for relative, data in contents.items():
        (directory / relative).write_bytes(data)
    state = [("CHAIN_DEVICE", device), ("CHAIN_ITER_ID", name), ("CHAIN_DEVICE", device)]
    state += [
        (key, hashlib.sha256(contents[relative]).hexdigest())
        for key, relative in ITERATION_FILES.items()
    ]
    (directory / "state.env").write_text("".join(f"{k}={shlex.quote(v)}\n" for k, v in state))
    rows = [f"{n}\tstep\trunning\t0\t-" for n in steps] + [f"{n}\tstep\t0\t9\t-" for n in steps]
    (directory / "steps.tsv").write_text("\n".join(rows) + "\n")
    return directory


@pytest.fixture
def iteration_factory() -> Callable[..., Path]:
    """Directories of run_iteration.sh, as freeze_query.py select reads them."""
    return make_iteration
