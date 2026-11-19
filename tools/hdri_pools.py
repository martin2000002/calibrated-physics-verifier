"""Print the ``[dataset.hdri_pools]`` TOML table of HDRI backgrounds outside OpenPano.

    PYTHONPATH=src python tools/hdri_pools.py HDRI_haven.json openpano-listing.txt

``HDRI_haven.json`` is Kubric's public HDRI manifest and the listing is ``unzip -l`` of the public
``openpano.zip``; ``physverify.synth.hdri_pools`` explains the pools.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from physverify.synth.hdri_pools import halves, outside_ids, training_names


def pools_table(manifest_path: Path, listing_path: Path) -> list[str]:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)["assets"]
    with listing_path.open(encoding="utf-8") as handle:
        seen = training_names(handle)
    dev, reserved = halves(outside_ids(manifest, seen))
    return [
        "[dataset.hdri_pools]",
        f"openpano_ood_dev = {json.dumps(dev)}",
        f"openpano_ood_reserved = {json.dumps(reserved)}",
    ]


if __name__ == "__main__":
    for line in pools_table(Path(sys.argv[1]), Path(sys.argv[2])):
        print(line)
