"""The raw oracle run the equivalence driver prepares is one ``--from-run`` accepts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from physverify.app.context import AppContext, GlobalOptions, Invocation
from physverify.app.services import checks_hash_for
from physverify.calibration.collect import require_same_checks
from physverify.core.errors import UsageError

DRIVER = Path(__file__).resolve().parents[2] / "tools" / "equivalence" / "driver.py"


def _load_driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("equivalence_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _session_hash(config_dir: Path, environ: dict[str, str]) -> str:
    return checks_hash_for(
        AppContext(Invocation((), environ), GlobalOptions(config_dir=config_dir))
    )


def test_the_raw_run_carries_the_sides_checks_hash_and_device(
    tmp_path: Path, config_dir: Path, environ: dict[str, str]
) -> None:
    folder = tmp_path / "raw-oracle"
    folder.mkdir()
    _load_driver().stamp_raw_run(folder, config_dir, environ)
    require_same_checks(folder, _session_hash(config_dir, environ), "cpu")


def test_an_unstamped_raw_run_is_refused(
    tmp_path: Path, config_dir: Path, environ: dict[str, str]
) -> None:
    folder = tmp_path / "raw-oracle"
    folder.mkdir()
    with pytest.raises(UsageError, match=r"no checks_hash\.txt"):
        require_same_checks(folder, _session_hash(config_dir, environ), "cpu")
