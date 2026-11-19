"""On-disk layout of the generator-comparison outputs, under the campaign root; paths only, no
policy.

```
<data_dir>/finetune/<campaign>/h3/
  preregistration.json, preregistrations/<sha256>.json (the versions it replaced)
  scoring/<model_tag>__<scorer>.json
  judges/gate.json, judges/<split>/<model_tag>__<task>.json, judges/runs/
  h3_report.json, h3_report.md
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

COMPARISON_DIR: Final = "h3"
JUDGES_DIR: Final = "judges"


@dataclass(frozen=True)
class ComparisonLayout:
    root: Path

    @classmethod
    def of(cls, campaign_root: Path) -> ComparisonLayout:
        return cls(campaign_root / COMPARISON_DIR)

    @property
    def prereg(self) -> Path:
        return self.root / "preregistration.json"

    def replaced(self, sha256: str) -> Path:
        return self.root / "preregistrations" / f"{sha256}.json"

    def scoring(self, model_tag: str, scorer: str) -> Path:
        return self.root / "scoring" / f"{model_tag}__{scorer}.json"

    @property
    def gate(self) -> Path:
        return self.root / JUDGES_DIR / "gate.json"

    def judge_output(self, split: str, model_tag: str, task: str) -> Path:
        return self.root / JUDGES_DIR / split / f"{model_tag}__{task}.json"

    def judge_outputs(self, split: str) -> list[Path]:
        directory = self.root / JUDGES_DIR / split
        return sorted(directory.glob("*.json")) if directory.is_dir() else []

    @property
    def runs(self) -> Path:
        return self.root / JUDGES_DIR / "runs"

    @property
    def report_json(self) -> Path:
        return self.root / "h3_report.json"

    @property
    def report_md(self) -> Path:
        return self.root / "h3_report.md"
