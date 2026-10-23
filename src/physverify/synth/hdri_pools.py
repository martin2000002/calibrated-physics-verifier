"""HDRI backgrounds outside a model's training panoramas.

GeoCalib was trained on OpenPano, whose PolyHaven panoramas include Kubric's single background.
The out-of-distribution pool is every id of Kubric's public ``HDRI_haven.json`` whose name is in
no OpenPano split, ordered by the sha256 of the id and cut into two fixed halves: the first for
development, the second reserved for the freeze. ``python -m physverify.synth.hdri_pools MANIFEST
LISTING`` prints both as the ``[dataset.hdri_pools]`` TOML table.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from physverify.core.digests import text_sha256

PANORAMA_DIR = "openpano/panoramas/"
RESOLUTION_SUFFIX = re.compile(r"_\d+k$")


def panorama_name(path: str) -> str:
    """``openpano/panoramas/train/abandoned_parking_16k.jpg`` -> ``abandoned_parking``."""
    return RESOLUTION_SUFFIX.sub("", PurePosixPath(path).stem)


def training_names(listing: Iterable[str]) -> set[str]:
    """Panorama names of every OpenPano split in a zip listing (one path per line, size after)."""
    paths = (line.split()[0] for line in listing if line.strip())
    return {panorama_name(p) for p in paths if p.startswith(PANORAMA_DIR) and "." in p[-6:]}


def outside_ids(manifest_ids: Iterable[str], seen: set[str]) -> list[str]:
    """Manifest ids not in ``seen``, ordered by the sha256 of the id."""
    unseen = {i for i in manifest_ids if i not in seen}
    return sorted(unseen, key=text_sha256)


def halves(ids: list[str]) -> tuple[list[str], list[str]]:
    """(development, reserved); the reserved half takes the extra id of an odd count."""
    cut = len(ids) // 2
    return ids[:cut], ids[cut:]


def main(manifest_path: str, listing_path: str) -> None:
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)["assets"]
    with Path(listing_path).open(encoding="utf-8") as handle:
        seen = training_names(handle)
    dev, reserved = halves(outside_ids(manifest, seen))
    print("[dataset.hdri_pools]")
    print(f"openpano_ood_dev = {json.dumps(dev)}")
    print(f"openpano_ood_reserved = {json.dumps(reserved)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
