from pathlib import Path

import pytest

from physverify.core.atomic import write_atomic, write_bytes_atomic, write_text_atomic


def test_writes_replace_the_file_and_leave_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.bin"
    write_bytes_atomic(target, b"first")
    write_text_atomic(target, "second")
    assert target.read_bytes() == b"second"
    assert [entry.name for entry in target.parent.iterdir()] == ["out.bin"]


def test_a_failed_write_keeps_the_old_file_and_removes_the_temp(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")

    def fail(temp: Path) -> None:
        temp.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_atomic(target, fail)
    assert target.read_text(encoding="utf-8") == "old"
    assert [entry.name for entry in tmp_path.iterdir()] == ["out.txt"]
