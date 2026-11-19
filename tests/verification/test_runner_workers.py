"""VerificationRunner with several worker processes: the one-worker report, in clip order."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence

from synthetic_clip import SyntheticFixtureDataset

from physverify.app.plugins import PLUGIN_KINDS
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest
from physverify.verification.check_set import ChecksSettings, build_checks
from physverify.verification.inputs import ClipInputs, OracleSource
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import VerificationReport, VerificationRunner
from physverify.verification.thresholding import ThresholdTable

BROKEN = "copy-2"


class CopiesSource:
    """The fixture clip's oracle inputs under each copy's id; ``BROKEN`` has no inputs."""

    def __init__(self, dataset: SyntheticFixtureDataset) -> None:
        self._oracle = OracleSource(dataset)
        self._clip = dataset.clips()[0]

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for clip in clips:
            if clip.clip_id == BROKEN:
                yield ClipInputs(clip, None, "MissingInputError: broken copy", InputLevel.ORACLE)
                continue
            (item,) = self._oracle.load([self._clip])
            yield dataclasses.replace(item, clip=clip)


def report(workers: int) -> VerificationReport:
    dataset = SyntheticFixtureDataset()
    copies = [dataclasses.replace(dataset.clips()[0], clip_id=f"copy-{n}") for n in range(5)]
    checks = build_checks(PLUGIN_KINDS["checks"](), ChecksSettings({}))
    runner = VerificationRunner(checks, RawPolicy(), ThresholdTable.provisional({}), workers)
    return runner.run(CopiesSource(dataset), copies)


def test_three_workers_give_the_one_worker_report() -> None:
    """``repr`` because table rows carry NaN sigmas, which plain dataclass equality rejects."""
    serial, parallel = report(1), report(3)
    assert [c.clip_id for c in parallel.clips] == [f"copy-{n}" for n in range(5)]
    assert [row.clip_id for row in parallel.table] == [row.clip_id for row in serial.table]
    assert repr(parallel.table) == repr(serial.table)
    assert repr(parallel.results) == repr(serial.results)
    assert any(row.clip_id == BROKEN for row in parallel.table)
