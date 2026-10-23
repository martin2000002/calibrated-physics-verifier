"""V8 contacts of the fixture: the whole ground-truth chain down to ``contacts/ground_truth``,
written once by the code before it3 (commit 2a762ad) into
``golden/truth_contacts_before_it3.json`` and compared with the current code (it3 T11). The same
module runs in both trees:

    PYTHONPATH=<old>/src:tests/fixtures:tests/extraction \
        python tests/extraction/truth_contacts_golden_case.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from support import make_context, run_provider
from synthetic_clip import build_clip

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.models import RuntimeSettings
from physverify.extraction._base import GROUND_TRUTH
from physverify.schema.contacts import ContactTable
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import STAGE_ORDER, Stage

GOLDEN = Path(__file__).resolve().parent / "golden" / "truth_contacts_before_it3.json"


def truth_contacts() -> ContactTable:
    clip = build_clip(RuntimeSettings())
    providers = PLUGIN_KINDS["providers"]()
    outputs: dict[Stage, StageOutput] = {}
    for stage in STAGE_ORDER:
        provider = providers[f"{stage.value}/{GROUND_TRUTH}"]
        ctx = make_context(clip, upstream=outputs, gt=clip.ground_truth)
        outputs[stage] = run_provider(provider, ctx)
    table = outputs[Stage.CONTACTS]
    assert isinstance(table, ContactTable)
    return table


if __name__ == "__main__":
    truth_contacts().save_json(Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN)
