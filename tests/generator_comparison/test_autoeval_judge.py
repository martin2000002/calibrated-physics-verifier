from __future__ import annotations

import csv
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from physverify.core.errors import MissingInputError, PhysverifyError
from physverify.generator_comparison.judges._base import JudgeInput
from physverify.generator_comparison.judges.videophy2_autoeval import (
    INFERENCE,
    MISSING_ROW,
    OUT_OF_RANGE,
    PARSE_FAILURE,
    UNPARSEABLE,
    Videophy2AutoevalJudge,
    command,
    parse_output,
    parse_value,
)
from physverify.generator_comparison.settings import AutoevalSettings

REVISION = "ca38370e1dcc79c99c7017dd8755d0a087299afc"
COMMIT = "7d8b5ca65425758d2c4fc5435015609d429d9480"


def settings(root: Path | None) -> AutoevalSettings:
    return AutoevalSettings(
        root="" if root is None else str(root), hf_revision=REVISION, code_commit=COMMIT
    )


def install(root: Path) -> AutoevalSettings:
    cfg = settings(root)
    python = root / cfg.venv / "bin" / "python"
    script = root / cfg.repo_dir / INFERENCE
    for path in (python, script):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (root / cfg.checkpoint_dir).mkdir(parents=True)
    return cfg


class FakeAutoeval:
    """Writes the upstream output CSV: pandas' index, the input columns and ``score``."""

    def __init__(self, scores: Mapping[str, str], code: int = 0, timeout: bool = False) -> None:
        self.scores = scores
        self.code = code
        self.timeout = timeout
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if self.timeout:
            raise subprocess.TimeoutExpired(list(argv), kwargs["timeout"])
        if self.code == 0:
            self._write(
                Path(argv[argv.index("--input_csv") + 1]),
                Path(argv[argv.index("--output_csv") + 1]),
            )
        return subprocess.CompletedProcess(list(argv), self.code, "", "Traceback: boom")

    def _write(self, csv_in: Path, csv_out: Path) -> None:
        with csv_in.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "videopath", "caption", "score"])
            for i, row in enumerate(rows):
                name = Path(row["videopath"]).stem
                if name in self.scores:
                    writer.writerow([i, row["videopath"], row["caption"], self.scores[name]])


def items(tmp_path: Path, names: Sequence[str]) -> list[JudgeInput]:
    found = []
    for name in names:
        video = tmp_path / "videos" / f"{name}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"v")
        found.append(JudgeInput(name, video, f"caption of {name}"))
    return found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3", (3.0, None)),
        ("4.0", (4.0, None)),
        ("0", (None, PARSE_FAILURE)),
        ("7", (None, OUT_OF_RANGE)),
        ("", (None, UNPARSEABLE)),
        ("five", (None, UNPARSEABLE)),
        ("inf", (None, UNPARSEABLE)),
        ("-inf", (None, UNPARSEABLE)),
        ("nan", (None, UNPARSEABLE)),
    ],
)
def test_parse_value(text: str, expected: tuple[float | None, str | None]) -> None:
    assert parse_value(text) == expected


def test_scores_parse_failures_and_missing_rows(tmp_path: Path) -> None:
    cfg = install(tmp_path / "root")
    fake = FakeAutoeval({"a": "5", "b": "0"})
    judge = Videophy2AutoevalJudge(cfg, tmp_path / "runs", runner=fake)
    scores = judge.judge(items(tmp_path, ["a", "b", "c"]), "pc")
    assert scores["a"].value == 5.0 and scores["a"].scale == (1.0, 5.0)
    assert (scores["b"].value, scores["b"].reason) == (None, PARSE_FAILURE)
    assert (scores["c"].value, scores["c"].reason) == (None, MISSING_ROW)
    assert scores["a"].stamp == judge.stamp("pc")
    argv = fake.calls[0]
    assert argv[1:3] == ["-s", "-E"]
    assert argv[argv.index("--task") + 1] == "pc"
    assert argv[argv.index("--num_frames") + 1] == "32"


def test_the_stamp_names_every_pin() -> None:
    stamp = Videophy2AutoevalJudge(settings(None), Path("runs")).stamp("sa")
    assert stamp == f"videophy2_autoeval:{REVISION[:12]}@code:{COMMIT[:12]}:sa:f32b1"


def test_missing_installation_is_named(tmp_path: Path) -> None:
    assert "root is not set" in (
        Videophy2AutoevalJudge(settings(None), tmp_path).unavailable() or ""
    )
    judge = Videophy2AutoevalJudge(settings(tmp_path), tmp_path, runner=FakeAutoeval({}))
    reason = judge.unavailable()
    assert reason is not None and "venv" in reason and "setup_autoeval.sh" in reason
    with pytest.raises(MissingInputError):
        judge.judge(items(tmp_path, ["a"]), "pc")
    with pytest.raises(MissingInputError):
        command(settings(None), "pc", Path("i"), Path("o"))


def test_a_failed_run_raises_with_its_stderr(tmp_path: Path) -> None:
    cfg = install(tmp_path / "root")
    judge = Videophy2AutoevalJudge(cfg, tmp_path / "runs", runner=FakeAutoeval({}, code=3))
    with pytest.raises(PhysverifyError, match=r"exited with 3(.|\n)*boom"):
        judge.judge(items(tmp_path, ["a"]), "pc")


def test_a_timeout_raises(tmp_path: Path) -> None:
    cfg = install(tmp_path / "root")
    judge = Videophy2AutoevalJudge(cfg, tmp_path / "runs", runner=FakeAutoeval({}, timeout=True))
    with pytest.raises(PhysverifyError, match="timed out"):
        judge.judge(items(tmp_path, ["a"]), "pc")


def test_output_without_index_or_score_column(tmp_path: Path) -> None:
    plain = tmp_path / "plain.csv"
    plain.write_text("videopath,caption,score\n/v.mp4,c,2\n", encoding="utf-8")
    assert parse_output(plain) == {"/v.mp4": (2.0, None)}
    broken = tmp_path / "broken.csv"
    broken.write_text("videopath,caption\n/v.mp4,c\n", encoding="utf-8")
    with pytest.raises(PhysverifyError, match="score"):
        parse_output(broken)


def test_blank_and_short_rows_are_skipped(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"
    output.write_text(",videopath,caption,score\n0,/a.mp4,c,4\n\n1\n2,/b.mp4\n", encoding="utf-8")
    assert parse_output(output) == {"/a.mp4": (4.0, None), "/b.mp4": (None, UNPARSEABLE)}
