"""Training samples of a learned variant: one row per subject of the default check, with the
model inputs, a regression target and what the clip was (clean, positive or decoy).

Targets: a subject that holds every target object of a positive aimed at the check gets the
positive's recorded magnitude; every other subject, and every subject of a clean or decoy clip,
gets 0. A clip perturbed for another check is not a sample.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from physverify.checks._base import CheckInputs
from physverify.checks.variants._learned import (
    LearnedMixin,
    image_key,
    model_inputs,
    pair_image,
)
from physverify.core.feature_table import FeatureTable
from physverify.learned.image_table import ENCODER_KEY, FRAMES_KEY
from physverify.schema.scene import PerturbationRecord

FloatArray = NDArray[np.float64]
CLEAN_NAME = "none"
LABEL = "label"


class Role(StrEnum):
    CLEAN = "clean"
    POSITIVE = "positive"
    DECOY = "decoy"


def clip_role(record: PerturbationRecord | None, check_id: str) -> Role | None:
    """The clip's role for ``check_id``; None for a clip perturbed for another check."""
    if record is None or record.name == CLEAN_NAME:
        return Role.CLEAN
    if not record.target_checks:
        return Role.DECOY
    return Role.POSITIVE if check_id in record.target_checks else None


def target(record: PerturbationRecord | None, role: Role, subject: Sequence[str]) -> float:
    if role is not Role.POSITIVE or record is None:
        return 0.0
    return float(record.magnitude) if set(record.target_objects) <= set(subject) else 0.0


@dataclass(frozen=True)
class Sample:
    clip_id: str
    scene_id: str
    subject: tuple[str, ...]
    arm: str
    role: Role
    label: float
    dense: FloatArray
    tokens: FloatArray

    @property
    def key(self) -> str:
        return "|".join((self.clip_id, self.arm, *self.subject))


def subject_samples(
    check: LearnedMixin,
    inputs: CheckInputs,
    table: FeatureTable,
    columns: tuple[Sequence[str], Sequence[str]],
    arm: str,
) -> list[Sample]:
    """``columns``: the image table's dense and token columns the model reads."""
    clip = inputs.clip
    role = clip_role(clip.perturbation, type(check).check_id)
    if role is None:
        return []
    dense_columns, token_columns = columns
    samples = []
    for subject in check.subjects(inputs):
        keys = [image_key(clip.clip_id, inputs.provenance, a, b) for a, b in subject.pairs]
        image = pair_image(table, keys, dense_columns, token_columns)
        dense, tokens = model_inputs(subject.track, image, len(dense_columns), len(token_columns))
        label = target(clip.perturbation, role, subject.base.subject)
        scene = clip.scene_id or clip.clip_id
        samples.append(
            Sample(clip.clip_id, scene, subject.base.subject, arm, role, label, dense, tokens)
        )
    return samples


def row_keys(samples: Sequence[Sample]) -> tuple[str, ...]:
    """Unique table row keys: a check can score the same subject more than once in a clip (one
    result per resting run), so a repeated key gets its occurrence number appended."""
    seen: Counter[str] = Counter()
    keys = []
    for sample in samples:
        n = seen[sample.key]
        seen[sample.key] += 1
        keys.append(sample.key if n == 0 else f"{sample.key}|#{n}")
    return tuple(keys)


@dataclass(frozen=True)
class TableSource:
    """The image table a sample set's rows were read from: which frames they read and the crop
    encoder's sha256 (empty when unknown)."""

    frames: str = ""
    encoder_sha256: str = ""

    @classmethod
    def of(cls, table: FeatureTable) -> TableSource:
        return cls(table.meta.get(FRAMES_KEY, ""), table.meta.get(ENCODER_KEY, ""))


@dataclass(frozen=True)
class SampleSet:
    samples: tuple[Sample, ...]
    dense_columns: tuple[str, ...]
    token_columns: tuple[str, ...]
    source: TableSource = TableSource()

    def arrays(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        dense = np.vstack([s.dense for s in self.samples])
        tokens = np.vstack([s.tokens for s in self.samples])
        return dense, tokens, np.array([s.label for s in self.samples])

    def subset(self, keep: Sequence[bool]) -> SampleSet:
        chosen = tuple(s for s, k in zip(self.samples, keep, strict=True) if k)
        return SampleSet(chosen, self.dense_columns, self.token_columns, self.source)

    def to_table(self) -> FeatureTable:
        columns = (*self.dense_columns, *self.token_columns, LABEL)
        values = np.vstack(
            [np.concatenate([s.dense, s.tokens, [s.label]]) for s in self.samples]
        ).reshape(len(self.samples), len(columns))
        described = [
            [s.clip_id, s.scene_id, list(s.subject), s.arm, s.role.value] for s in self.samples
        ]
        meta = {
            "samples": json.dumps(described),
            "dense_columns": json.dumps(list(self.dense_columns)),
            FRAMES_KEY: self.source.frames,
            ENCODER_KEY: self.source.encoder_sha256,
        }
        return FeatureTable(row_keys(self.samples), columns, values, meta)

    @classmethod
    def from_table(cls, table: FeatureTable) -> SampleSet:
        dense_columns = tuple(json.loads(table.meta["dense_columns"]))
        n_dense = len(dense_columns)
        token_columns = table.columns[n_dense:-1]
        samples = tuple(
            Sample(
                clip,
                scene,
                tuple(subject),
                arm,
                Role(role),
                float(row[-1]),
                row[:n_dense],
                row[n_dense:-1],
            )
            for (clip, scene, subject, arm, role), row in zip(
                json.loads(table.meta["samples"]), table.values, strict=True
            )
        )
        return cls(samples, dense_columns, tuple(token_columns), TableSource.of(table))
