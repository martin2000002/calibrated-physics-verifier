"""Read-only queries run_a8v2.sh asks of the checkout: clip selections, the re-rendered clips of a
migration, the checks on an evaluation and the gravity rows of a stage report.

Exit codes: 0 ok, 2 a check failed or an input is unusable (message on stderr).

    python a8v2_query.py [--set K=V]... clips DATASET SUBSET [--split calib|eval]
    python a8v2_query.py rerendered MANIFEST [--missing-ok]
    python a8v2_query.py [--set K=V]... evaluation EVALUATION_JSON
    python a8v2_query.py gravity STAGES_JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = "precise"
DEVICE = "cpu"
H1_HEADING = "## H1 primary"
GRAVITY = ("lift", "gravity_angle")


class QueryError(Exception):
    """A check failed or an input is unusable (exit 2)."""


class Clip(Protocol):
    @property
    def clip_id(self) -> str: ...

    @property
    def split(self) -> Any: ...


class Source(Protocol):
    def clips(self, subset: str | None = None) -> Sequence[Clip]: ...


def clip_ids(source: Source, subset: str, split: str | None) -> list[str]:
    clips = source.clips(subset)
    return sorted(c.clip_id for c in clips if split is None or c.split.value == split)


def rerendered(manifest: Path, *, missing_ok: bool) -> list[str]:
    from physverify.core.errors import PhysverifyError
    from physverify.synth.migrate import rerender_clip_ids

    if not manifest.is_file():
        if missing_ok:
            return []
        raise QueryError(f"no migration manifest at {manifest}: run step 0 (migrate) first")
    try:
        return rerender_clip_ids(manifest)
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc


def evaluation_lines(path: Path, allowed: Sequence[Mapping[str, int]]) -> list[str]:
    """The real sets and the H1 section of ``path``; refused without the primary view or with a
    real set rule 1.5 does not allow."""
    data = _json(path)
    sizes = {ref["name"]: ref["n_clips"] for ref in data.get("real_sets", [])}
    if sizes not in [dict(a) for a in allowed]:
        raise QueryError(f"{path}: real_sets {sizes} not in the allowed sets {list(allowed)}")
    if data.get("h1_primary") is None:
        raise QueryError(f"{path} has no h1_primary: the evaluation predates WP-J2, rerun from 6")
    lines = ["Real sets: " + ", ".join(f"{k} {v}" for k, v in sorted(sizes.items())), ""]
    return lines + h1_section(path.with_suffix(".md"))


def h1_section(markdown: Path) -> list[str]:
    if not markdown.is_file():
        raise QueryError(f"no {markdown} next to the evaluation")
    lines = markdown.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(H1_HEADING)), None)
    if start is None:
        raise QueryError(f"{markdown} has no '{H1_HEADING}' section")
    return lines[start:]


def gravity_lines(path: Path) -> list[str]:
    """The calib rows of the LIFT gravity-angle metric of a ``diagnose stages`` report."""
    data = _json(path)
    split = data.get("profile_split", "calib")
    rows = [
        r
        for r in data.get("rows", [])
        if (r.get("stage"), r.get("metric")) == GRAVITY and r.get("split") == split
    ]
    if not rows:
        raise QueryError(f"{path} has no {split} row of {'/'.join(GRAVITY)}")
    lines = [f"LIFT gravity angle ({split}, {path}):", "", "| provider | n_clips | value |"]
    lines.append("|---|---|---|")
    for row in rows:
        values = ", ".join(f"{k} {v:.4g}" for k, v in sorted(row.get("values", {}).items()))
        lines.append(f"| {row.get('provider')} | {row.get('n_clips')} | {values} |")
    return lines


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QueryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise QueryError(f"{path} is not a JSON object")
    return data


def _settings(overrides: Sequence[str]) -> Any:
    from physverify.config.loader import load_settings

    return load_settings(
        REPO_ROOT / "configs",
        environ=os.environ,
        profile=PROFILE,
        device=DEVICE,
        overrides=overrides,
    )


def _source(settings: Any, dataset: str) -> Source:
    from physverify.cli.context import PLUGIN_KINDS

    registry = PLUGIN_KINDS["datasets"]()
    if dataset not in registry:
        raise QueryError(f"unknown dataset {dataset!r}; known: {sorted(registry)}")
    source: Source = registry[dataset](settings)
    return source


def _allowed_sets(settings: Any) -> Sequence[Mapping[str, int]]:
    from physverify.calibration.exclusions import SECTION, exclusion_rules
    from physverify.calibration.settings import check_registry

    rules = exclusion_rules(settings.sections.get(SECTION, {}), check_registry())
    return list(rules.real.allowed_sets)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    sub = parser.add_subparsers(dest="command", required=True)
    clips = sub.add_parser("clips")
    clips.add_argument("dataset")
    clips.add_argument("subset")
    clips.add_argument("--split", choices=("calib", "eval"))
    migrated = sub.add_parser("rerendered")
    migrated.add_argument("manifest", type=Path)
    migrated.add_argument("--missing-ok", action="store_true")
    sub.add_parser("evaluation").add_argument("path", type=Path)
    sub.add_parser("gravity").add_argument("path", type=Path)
    return parser


def _lines(args: argparse.Namespace) -> list[str]:
    commands: dict[str, Callable[[], list[str]]] = {
        "clips": lambda: clip_ids(
            _source(_settings(args.overrides), args.dataset), args.subset, args.split
        ),
        "rerendered": lambda: rerendered(args.manifest, missing_ok=args.missing_ok),
        "evaluation": lambda: evaluation_lines(
            args.path, _allowed_sets(_settings(args.overrides))
        ),
        "gravity": lambda: gravity_lines(args.path),
    }
    return commands[args.command]()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        lines = _lines(args)
    except QueryError as exc:
        print(f"a8v2_query: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
