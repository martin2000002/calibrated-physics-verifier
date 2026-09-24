"""Pinned checkpoints: where they live, whether they are present, and verified downloads
(ADR 0005, 3.5). huggingface_hub is imported only inside the network downloader."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from physverify.config.models import PathsSettings
from physverify.core.errors import ConfigError
from physverify.extraction._base import StageProvider, WeightKind, WeightSpec

ALLOW_PATTERNS = ("*.json", "*.safetensors", "*.txt", "*.model")
DEFAULT_HF_FILE = "model.safetensors"
HF_REPO_PARTS = 2
CHUNK_BYTES = 1 << 20


class Downloader(Protocol):
    def hf_snapshot(
        self, repo_id: str, revision: str, cache_dir: Path, allow_patterns: Sequence[str]
    ) -> Path: ...

    def http_file(self, url: str, dest: Path) -> None: ...

    def git_checkout(self, url: str, revision: str, dest: Path) -> None: ...


class NetworkDownloader:
    def hf_snapshot(
        self, repo_id: str, revision: str, cache_dir: Path, allow_patterns: Sequence[str]
    ) -> Path:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id, revision=revision, cache_dir=cache_dir, allow_patterns=list(allow_patterns)
            )
        )

    def http_file(self, url: str, dest: Path) -> None:
        partial = dest.with_name(dest.name + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response, partial.open("wb") as out:  # noqa: S310
            shutil.copyfileobj(response, out, CHUNK_BYTES)
        partial.replace(dest)

    def git_checkout(self, url: str, revision: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", revision], check=True)


def hf_repo_and_file(ref: str) -> tuple[str, str]:
    """``org/name`` -> (repo, model.safetensors); ``org/name/path/file`` names another file."""
    parts = ref.split("/")
    repo = "/".join(parts[:HF_REPO_PARTS])
    file = "/".join(parts[HF_REPO_PARTS:]) or DEFAULT_HF_FILE
    return repo, file


def hub_cache(paths: PathsSettings) -> Path:
    return paths.hf_home / "hub"


def local_path(spec: WeightSpec, paths: PathsSettings) -> Path:
    if spec.kind is WeightKind.HF:
        org, name = hf_repo_and_file(spec.ref)[0].split("/")
        return hub_cache(paths) / f"models--{org}--{name}" / "snapshots" / spec.revision
    basename = Path(urllib.parse.urlparse(spec.ref).path).name
    if spec.kind is WeightKind.URL:
        return paths.weights_dir / basename
    return paths.vendor_dir / basename.removesuffix(".git")


def weight_file(spec: WeightSpec, paths: PathsSettings) -> Path:
    if spec.kind is WeightKind.HF:
        return local_path(spec, paths) / hf_repo_and_file(spec.ref)[1]
    return local_path(spec, paths)


def pin_main_ref(spec: WeightSpec, paths: PathsSettings) -> None:
    """Point a missing ``refs/main`` at the pinned snapshot.

    A download by commit hash writes no ref, but transformers asks for ``main`` offline (EdgeTAM
    resolves ``timm/repvit_m1.dist_in1k`` while formatting a config); an existing ref is kept."""
    ref = local_path(spec, paths).parent.parent / "refs" / "main"
    if not ref.exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(spec.revision, encoding="utf-8")


def git_head(repo: Path) -> str:
    head = repo / ".git" / "HEAD"
    return head.read_text(encoding="utf-8").strip() if head.is_file() else ""


def is_present(spec: WeightSpec, paths: PathsSettings) -> bool:
    if spec.kind is WeightKind.GIT:
        return git_head(local_path(spec, paths)) == spec.revision
    return weight_file(spec, paths).is_file()


def missing_weights(
    providers: Iterable[type[StageProvider]], paths: PathsSettings
) -> tuple[WeightSpec, ...]:
    specs = dict.fromkeys(spec for provider in providers for spec in provider.weights)
    return tuple(spec for spec in specs if not is_present(spec, paths))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discard(path: Path) -> None:
    target = path.resolve()
    path.unlink(missing_ok=True)
    if target != path:
        target.unlink(missing_ok=True)


def _verify_file(spec: WeightSpec, path: Path) -> Path:
    if not path.is_file():
        raise ConfigError(f"{spec.ref}: {path} was not downloaded")
    actual = sha256_of(path)
    if actual != spec.sha256:
        _discard(path)
        raise ConfigError(f"{spec.ref}: sha256 {actual} != pinned {spec.sha256}; file removed")
    return path


def fetch(spec: WeightSpec, paths: PathsSettings, downloader: Downloader) -> Path:
    """Download (unless present) and verify sha256 (hf, url) or HEAD == revision (git)."""
    target = weight_file(spec, paths)
    if spec.kind is WeightKind.HF:
        repo, file = hf_repo_and_file(spec.ref)
        if not target.is_file():
            downloader.hf_snapshot(repo, spec.revision, hub_cache(paths), (*ALLOW_PATTERNS, file))
        verified = _verify_file(spec, target)
        pin_main_ref(spec, paths)
        return verified
    if spec.kind is WeightKind.URL:
        if not target.is_file():
            downloader.http_file(spec.ref, target)
        return _verify_file(spec, target)
    if not is_present(spec, paths):
        shutil.rmtree(target, ignore_errors=True)
        downloader.git_checkout(spec.ref, spec.revision, target)
    head = git_head(target)
    if head != spec.revision:
        shutil.rmtree(target, ignore_errors=True)
        raise ConfigError(f"{spec.ref}: HEAD {head!r} != pinned {spec.revision}; clone removed")
    return target


@contextmanager
def vendored_import_path(root: Path) -> Iterator[None]:
    """Make a vendored checkout importable only while its modules are being imported."""
    entry = str(root)
    sys.path.insert(0, entry)
    try:
        yield
    finally:
        if entry in sys.path:
            sys.path.remove(entry)


@contextmanager
def hub_offline() -> Iterator[None]:
    """Forbid Hugging Face Hub network calls while a model loads from its pinned snapshot."""
    from huggingface_hub import constants

    previous = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = previous
