"""VideoPhy-2 AutoEval as an external judge, run in its own pinned environment.

The pinned ``VIDEOPHY2/inference.py`` reads a CSV (``videopath``, ``caption``) and writes it back
with an added ``score`` column and pandas' unnamed index column. Its parser falls back to 0 when
the answer is not 1..5, so 0 is a parse failure, never a point on the scale.
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Final

from physverify.core.errors import MissingInputError, PhysverifyError
from physverify.generator_comparison.judges._base import Judge, JudgeInput, JudgeScore
from physverify.generator_comparison.settings import AutoevalSettings, ComparisonSettings

SCALE: Final = (1.0, 5.0)
INFERENCE: Final = "VIDEOPHY2/inference.py"
VIDEO_COLUMN: Final = "videopath"
CAPTION_COLUMN: Final = "caption"
SCORE_COLUMN: Final = "score"
INDEX_PREFIX: Final = "Unnamed"
SHOWN: Final = 12
STDERR_TAIL: Final = 2000
SETUP_HINT: Final = "run scripts/h200/setup_autoeval.sh"
PARSE_FAILURE: Final = "parse_failure"
OUT_OF_RANGE: Final = "out_of_range"
UNPARSEABLE: Final = "unparseable"
MISSING_ROW: Final = "missing_row"

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def stamp_of(cfg: AutoevalSettings, task: str) -> str:
    return (
        f"videophy2_autoeval:{cfg.hf_revision[:SHOWN]}@code:{cfg.code_commit[:SHOWN]}"
        f":{task}:f{cfg.num_frames}b{cfg.batch_size}"
    )


def write_input_csv(items: Sequence[JudgeInput], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([VIDEO_COLUMN, CAPTION_COLUMN])
        for item in items:
            writer.writerow([str(item.video.resolve()), item.caption])


def command(cfg: AutoevalSettings, task: str, csv_in: Path, csv_out: Path) -> list[str]:
    """``-s -E`` keep the user site and PYTHON* variables out; ``-I`` is not used because the
    upstream script imports modules from its own directory."""
    venv, repo, ckpt = _paths(cfg)
    return [
        str(venv / "bin" / "python"),
        "-s",
        "-E",
        str(repo / INFERENCE),
        "--input_csv",
        str(csv_in),
        "--output_csv",
        str(csv_out),
        "--checkpoint",
        str(ckpt),
        "--task",
        task,
        "--num_frames",
        str(cfg.num_frames),
        "--batch_size",
        str(cfg.batch_size),
    ]


def parse_value(text: str) -> tuple[float | None, str | None]:
    try:
        value = int(float(text))
    except (ValueError, OverflowError):
        return None, UNPARSEABLE
    if value == 0:
        return None, PARSE_FAILURE
    if not SCALE[0] <= value <= SCALE[1]:
        return None, OUT_OF_RANGE
    return float(value), None


def parse_output(path: Path) -> dict[str, tuple[float | None, str | None]]:
    """``{videopath: (value, reason)}``; the leading unnamed index column is ignored, and a blank
    or short row without a video path is skipped (its video then reads as a missing row)."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        offset = 1 if header and (header[0] == "" or header[0].startswith(INDEX_PREFIX)) else 0
        names = header[offset:]
        if SCORE_COLUMN not in names or VIDEO_COLUMN not in names:
            raise PhysverifyError(f"{path}: no {SCORE_COLUMN!r} or {VIDEO_COLUMN!r} column")
        rows = [dict(zip(names, row[offset:], strict=False)) for row in reader]
    return {
        row[VIDEO_COLUMN]: parse_value(row.get(SCORE_COLUMN) or "")
        for row in rows
        if row.get(VIDEO_COLUMN)
    }


NO_ROOT: Final = f"[h3.autoeval] root is not set (PHYSVERIFY__H3__AUTOEVAL__ROOT); {SETUP_HINT}"


def _resolved(cfg: AutoevalSettings) -> tuple[Path, Path, Path] | None:
    venv, repo, ckpt = (cfg.path(v) for v in (cfg.venv, cfg.repo_dir, cfg.checkpoint_dir))
    return None if venv is None or repo is None or ckpt is None else (venv, repo, ckpt)


def _paths(cfg: AutoevalSettings) -> tuple[Path, Path, Path]:
    paths = _resolved(cfg)
    if paths is None:
        raise MissingInputError(NO_ROOT)
    return paths


def missing_parts(cfg: AutoevalSettings) -> str | None:
    paths = _resolved(cfg)
    if paths is None:
        return NO_ROOT
    venv, repo, ckpt = paths
    needed = {
        "venv": venv / "bin" / "python",
        "clone": repo / INFERENCE,
        "checkpoint": ckpt,
    }
    missing = [f"{name} ({path})" for name, path in needed.items() if not path.exists()]
    return (
        f"AutoEval is not installed: missing {', '.join(missing)}; {SETUP_HINT}"
        if missing
        else None
    )


class Videophy2AutoevalJudge:
    name: ClassVar[str] = "videophy2_autoeval"

    def __init__(
        self, cfg: AutoevalSettings, workdir: Path, runner: Runner = subprocess.run
    ) -> None:
        self._cfg = cfg
        self._workdir = workdir
        self._runner = runner

    @classmethod
    def from_config(cls, cfg: ComparisonSettings, workdir: Path) -> Judge:
        return cls(cfg.autoeval, workdir)

    def unavailable(self) -> str | None:
        return missing_parts(self._cfg)

    def stamp(self, task: str) -> str:
        return stamp_of(self._cfg, task)

    def judge(self, items: Sequence[JudgeInput], task: str) -> Mapping[str, JudgeScore]:
        reason = self.unavailable()
        if reason is not None:
            raise MissingInputError(reason)
        self._workdir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(dir=self._workdir, prefix=f"{task}-"))
        csv_in, csv_out = run_dir / "input.csv", run_dir / "output.csv"
        write_input_csv(items, csv_in)
        self._run(command(self._cfg, task, csv_in, csv_out), run_dir)
        parsed = parse_output(csv_out)
        stamp = self.stamp(task)
        return {item.video_id: self._score(parsed, item, stamp) for item in items}

    def _run(self, argv: list[str], cwd: Path) -> None:
        try:
            done = self._runner(
                argv,
                cwd=cwd,
                timeout=self._cfg.timeout_s,
                capture_output=True,
                text=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PhysverifyError(f"AutoEval timed out after {self._cfg.timeout_s} s") from exc
        if done.returncode != 0:
            tail = (done.stderr or "")[-STDERR_TAIL:]
            raise PhysverifyError(f"AutoEval exited with {done.returncode}:\n{tail}")

    @staticmethod
    def _score(
        parsed: Mapping[str, tuple[float | None, str | None]], item: JudgeInput, stamp: str
    ) -> JudgeScore:
        value, reason = parsed.get(str(item.video.resolve()), (None, MISSING_ROW))
        return JudgeScore(value, SCALE, stamp, reason)


PLUGINS = (Videophy2AutoevalJudge,)
