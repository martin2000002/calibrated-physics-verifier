"""On-disk layout of one dataset version, atomic writes and per-step completion markers.

<synthetic_dir>/<version>/
  plan.json
  scenes/<sim_id>/{spec.json, states.npz, raw_collisions.npz, sim_meta.json}
  clips/<clip_id>/{states_variant.npz, contacts_provisional.json, record.json, closest.npz,
                   contacts_world.json, layers.npz, kubric_meta.json, frames/%06d.png, video.mp4,
                   camera.json, states_world.npz, states_camera.npz, contacts_camera.json,
                   gt_meta.json, manifest.json}
  manifest.jsonl
  jobs/<task>-<hash>.jsonl
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from physverify.core import jsonio
from physverify.core.atomic import write_atomic, write_bytes_atomic
from physverify.core.cache_keys import params_hash
from physverify.core.errors import PhysverifyError
from physverify.schema.io import Record, dumps


class Step(StrEnum):
    """Pipeline steps; each one is done for an item when its marker file exists."""

    GENERATE = "generate"
    SIMULATE = "simulate"
    APPLY = "apply"
    PERTURB = "perturb"
    RENDER = "render"
    EXPORT = "export"


SPEC = "spec.json"
SIM_STATES = "states.npz"
RAW_COLLISIONS = "raw_collisions.npz"
SIM_META = "sim_meta.json"
SCENE_RECORD = "perturbation.json"
VARIANT_STATES = "states_variant.npz"
PROVISIONAL_CONTACTS = "contacts_provisional.json"
RECORD = "record.json"
CLOSEST = "closest.npz"
CONTACTS_WORLD = "contacts_world.json"
LAYERS = "layers.npz"
KUBRIC_META = "kubric_meta.json"
FRAMES_DIR = "frames"
VIDEO = "video.mp4"
CAMERA = "camera.json"
STATES_WORLD = "states_world.npz"
STATES_CAMERA = "states_camera.npz"
CONTACTS_CAMERA = "contacts_camera.json"
GT_META = "gt_meta.json"
CLIP_MANIFEST = "manifest.json"

MARKERS: dict[Step, str] = {
    Step.GENERATE: SPEC,
    Step.SIMULATE: SIM_META,
    Step.APPLY: RECORD,
    Step.PERTURB: CONTACTS_WORLD,
    Step.RENDER: KUBRIC_META,
    Step.EXPORT: GT_META,
}
SCENE_STEPS = frozenset({Step.GENERATE, Step.SIMULATE})
FAILURE_FILE = "failed.json"
SKIPPED_FILE = "skipped.json"


class ContainerRunner(Protocol):
    """What the pipeline needs from ``runtime.docker.DockerRunner`` (injected, faked in tests)."""

    def run(self, task: str, jobs: Path) -> Any: ...

    def container_path(self, host_path: Path) -> str: ...


@dataclass(frozen=True)
class DatasetStore:
    root: Path

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    def scene_dir(self, sim_id: str) -> Path:
        return self.root / "scenes" / sim_id

    def clip_dir(self, clip_id: str) -> Path:
        return self.root / "clips" / clip_id

    def jobs_path(self, task: str, key: str) -> Path:
        return self.root / "jobs" / f"{task}-{key}.jsonl"

    def item_dir(self, step: Step, item_id: str) -> Path:
        return self.scene_dir(item_id) if step in SCENE_STEPS else self.clip_dir(item_id)

    def is_done(self, step: Step, item_id: str) -> bool:
        return (self.item_dir(step, item_id) / MARKERS[step]).is_file()

    def is_skipped(self, clip_id: str) -> bool:
        return (self.clip_dir(clip_id) / SKIPPED_FILE).is_file()

    def failure(self, step: Step, item_id: str) -> dict[str, Any] | None:
        path = self.item_dir(step, item_id) / FAILURE_FILE
        return read_json(path) if path.is_file() else None

    def skip_reason(self, clip_id: str) -> str | None:
        path = self.clip_dir(clip_id) / SKIPPED_FILE
        return str(read_json(path)["reason"]) if path.is_file() else None

    def mark_skipped(self, clip_id: str, reason: str) -> None:
        atomic_write_json(self.clip_dir(clip_id) / SKIPPED_FILE, {"reason": reason})

    def clear_failure(self, step: Step, item_id: str) -> None:
        (self.item_dir(step, item_id) / FAILURE_FILE).unlink(missing_ok=True)

    def write_jobs(self, task: str, jobs: list[dict[str, Any]]) -> Path:
        """JSONL job file named by the hash of its content (identical batches share a file)."""
        path = self.jobs_path(task, params_hash({"jobs": jobs})[:12])
        write_bytes_atomic(path, "".join(dumps(job) + "\n" for job in jobs).encode("utf-8"))
        return path


def run_container(runner: ContainerRunner, task: str, jobs: Path) -> None:
    """Per-job errors land in ``failed.json``; a non-zero exit means the container itself failed."""
    result = runner.run(task, jobs)
    code = int(getattr(result, "returncode", 0))
    if code:
        raise PhysverifyError(f"container task {task!r} exited with {code} (jobs {jobs})")


def atomic_write_json(path: Path, payload: Any) -> None:
    write_bytes_atomic(path, dumps(payload).encode("utf-8"))


def read_json(path: Path) -> Any:
    """A JSON file the container or the store wrote; its shape is the reader's contract."""
    return jsonio.read_json(path)


def atomic_save(record: Record, path: Path) -> None:
    """``Record.save_json`` / ``save_npz`` (chosen by suffix) through a temporary file."""
    write_atomic(path, record.save_npz if path.suffix == ".npz" else record.save_json)
