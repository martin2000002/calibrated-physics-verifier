import hashlib
from pathlib import Path

from physverify.core.digests import CHUNK_BYTES, file_sha256, sha256_if_file, text_sha256


def test_file_digest_streams_the_whole_content(tmp_path: Path) -> None:
    data = bytes(range(256)) * (CHUNK_BYTES // 128 + 3)
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()
    assert sha256_if_file(path) == file_sha256(path)


def test_a_missing_file_has_no_digest(tmp_path: Path) -> None:
    assert sha256_if_file(tmp_path / "absent") is None
    assert sha256_if_file(tmp_path) is None


def test_text_digest_is_over_utf8() -> None:
    assert text_sha256("\u00e9") == hashlib.sha256("\u00e9".encode()).hexdigest()
