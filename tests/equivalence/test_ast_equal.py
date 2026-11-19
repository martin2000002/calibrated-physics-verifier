"""`tools/equivalence/ast_equal.py` accepts only the edits it is meant to accept."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "equivalence" / "ast_equal.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ast_equal", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ast_equal = _load_tool()

BASE = {
    "src/physverify/store.py": (
        '"""Store."""\n\nSTORE_SCHEMA = "physverify.store"\n\n\n'
        "def load(path: str, limit: int) -> list[str]:\n"
        "    lines = open(path).read().splitlines()\n"
        "    return [line.strip() for line in lines[:limit] if line]\n"
    ),
    "src/physverify/model.py": (
        '"""Model."""\n\n\nclass Model:\n    size: int = 0\n    """Size in pixels."""\n'
    ),
}


def _git(repo: Path, *args: str) -> None:
    identity = ("-c", "user.name=t", "-c", "user.email=t@t")
    subprocess.run(["git", "-C", str(repo), *identity, *args], check=True, capture_output=True)


def _commit(repo: Path, files: Mapping[str, str | None]) -> None:
    for path, text in files.items():
        target = repo / path
        if text is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")


def _new_repo(root: Path, files: Mapping[str, str | None] = BASE) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    _commit(root, files)
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _new_repo(tmp_path / "repo")


def _run(repo: Path, *flags: str) -> int:
    code: int = ast_equal.main([str(repo), "HEAD~1", "HEAD", "--paths", "src", *flags])
    return code


def test_comment_only_edit_is_equal(repo: Path) -> None:
    _commit(repo, {"src/physverify/store.py": BASE["src/physverify/store.py"] + "# a note\n"})
    assert _run(repo) == 0


def test_docstrings_are_equal_only_when_bare_strings_are_stripped(repo: Path) -> None:
    model = BASE["src/physverify/model.py"]
    edited = model.replace('"""Model."""', '"""A model."""').replace("in pixels", "of the box")
    _commit(repo, {"src/physverify/model.py": edited})
    assert _run(repo, "--strip-bare-strings") == 0
    assert _run(repo) == 1
    for statement in ('f"{Model.size}"', "Model.size"):
        _commit(repo, {"src/physverify/model.py": edited + statement + "\n"})
        assert _run(repo, "--strip-bare-strings") == 1, statement
        _commit(repo, {"src/physverify/model.py": edited})


def test_changed_literal_differs(repo: Path) -> None:
    store = BASE["src/physverify/store.py"].replace('"physverify.store"', '"physverify.other"')
    _commit(repo, {"src/physverify/store.py": store})
    assert _run(repo) == 1


def test_schema_id_does_not_follow_a_module_rename(tmp_path: Path) -> None:
    for name in ("STORE_SCHEMA", "SCHEMA"):
        store = BASE["src/physverify/store.py"].replace("STORE_SCHEMA", name)
        repo = _new_repo(tmp_path / name, {**BASE, "src/physverify/store.py": store})
        moved = store.replace('"physverify.store"', '"physverify.archive"')
        _commit(repo, {"src/physverify/store.py": None, "src/physverify/archive.py": moved})
        assert _run(repo) == 1, name


def test_schema_id_kept_through_a_module_rename_is_equal(repo: Path) -> None:
    _commit(repo, {"src/physverify/caller.py": 'ENTRY = "physverify.store.load"\n'})
    kept = BASE["src/physverify/store.py"]
    _commit(
        repo,
        {
            "src/physverify/store.py": None,
            "src/physverify/archive.py": kept,
            "src/physverify/caller.py": 'ENTRY = "physverify.archive.load"\n',
        },
    )
    assert _run(repo) == 0


def test_allowed_substitution_is_equal(repo: Path, tmp_path: Path) -> None:
    model = BASE["src/physverify/model.py"].replace("size", "width")
    _commit(repo, {"src/physverify/model.py": model})
    allow = tmp_path / "allow.tsv"
    allow.write_text("src/physverify/model.py\tname\tsize\twidth\n", encoding="utf-8")
    assert _run(repo, "--allow", str(allow)) == 0


def test_unused_allow_line_is_a_problem(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _commit(repo, {"src/physverify/store.py": BASE["src/physverify/store.py"] + "# a note\n"})
    allow = tmp_path / "allow.tsv"
    allow.write_text("*\tname\tabsent\tpresent\n", encoding="utf-8")
    assert _run(repo, "--allow", str(allow)) == 1
    assert "unused allow line: *\tname\tabsent" in capsys.readouterr().out
