"""Hand-built rows of the verifier-vs-VLM comparison: a Kubric part whose pair outcomes are known
and a small real part, with the settings that judge them."""

from __future__ import annotations

from physverify.baselines.records import ClipRow, Cuts, PartRows, SetRows
from physverify.baselines.settings import BootstrapSettings, Part, VlmComparisonSettings

U_B1 = 0.5
SHA = "0" * 64


def config(max_uninterpretable: float = 0.5) -> VlmComparisonSettings:
    boot = BootstrapSettings(n=300, seed=0, ci=0.95)
    return VlmComparisonSettings(
        clean_fpr=0.1,
        max_uninterpretable=max_uninterpretable,
        real_fpr=0.1,
        real_fpr_upper=0.5,
        agreement_partition="test",
        pairs=boot,
        agreement=boot,
        judges={},
        sets={},
    )


def clip(
    clip_id: str, scene: str, perturbed: bool, b1: float | None, judge: float | None
) -> ClipRow:
    return ClipRow(clip_id=clip_id, scene=scene, perturbed=perturbed, b1=b1, judge=judge)


def part(name: Part, rows: tuple[ClipRow, ...]) -> PartRows:
    cuts = Cuts(
        u_b1=U_B1,
        clean_fpr=0.1,
        n_clean=10,
        calibration_dataset="calib",
        calibration_sha256=SHA,
        excluded_checks=(),
    )
    found = SetRows(dataset=f"{name}_set", subset=None, scores_sha256=SHA, stamps=(), rows=rows)
    return PartRows(
        part=name,
        judge="judge",
        cut=4.0,
        cuts=cuts,
        cuts_sha256=SHA,
        freeze=None,
        looks=0,
        sets=(found,),
    )


# Per scene, (B1, judge) pair outcomes: s1 (1, 0), s2 (1, tie), s3 (0, unread so tie).
KUBRIC = part(
    "kubric",
    (
        clip("s1c", "s1", False, 0.1, 2.0),
        clip("s1p", "s1", True, 0.9, 5.0),
        clip("s2c", "s2", False, 0.2, 4.0),
        clip("s2p", "s2", True, 0.8, 4.0),
        clip("s3c", "s3", False, 0.7, 3.0),
        clip("s3p", "s3", True, 0.3, None),
    ),
)
REAL = part(
    "real",
    tuple(clip(f"r{i}", f"r{i}", False, b1, 3.0) for i, b1 in enumerate((0.1, 0.9, 0.2, None))),
)
