"""Shared by the learned variants: a trained artifact that maps a subject's track features and the
image features of its contact zones to an estimated magnitude in the check's unit.

The artifact and the image feature table are files named by sha256 in the variant's params (so
they enter ``params_hash``); their paths are not hashed. The magnitude is an estimate: ``sigma``
is the artifact's residual rms on its own validation split, and every result says so in its
details. Calibration (tau, the sigma model, the clause) treats it as any other magnitude.

Leakage guard: the artifact lists the perturbation seeds of its training clips, and a clip whose
seed is among them is refused (``LeakageError``) instead of being scored.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from physverify.checks._base import Check, CheckInputs, abstain, unscored
from physverify.core.digests import file_sha256
from physverify.core.errors import ConfigError, PhysverifyError
from physverify.core.feature_table import FeatureTable
from physverify.core.models import FrozenModel
from physverify.schema.checks import CheckResult
from physverify.stats.mlp import Standardiser, TwoHeadWeights, predict

FloatArray = NDArray[np.float64]
SIGMA_SOURCE = "validation_residual_rms"
HAS_IMAGE = "has_image"


class LeakageError(PhysverifyError):
    """An evaluated clip was a training clip of the artifact."""


class LearnedFields(FrozenModel):
    artifact_sha256: str = ""
    features_sha256: str = ""
    artifact_path: str = Field(default="", exclude=True)
    features_path: str = Field(default="", exclude=True)


@dataclass(frozen=True)
class LearnedArtifact:
    check_id: str
    unit: str
    dense_columns: tuple[str, ...]
    token_columns: tuple[str, ...]
    weights: TwoHeadWeights
    dense: Standardiser
    tokens: Standardiser
    residual_rms: float
    encoder_sha256: str
    training_seeds: frozenset[int]

    @property
    def meta(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "unit": self.unit,
            "dense_columns": list(self.dense_columns),
            "token_columns": list(self.token_columns),
            "residual_rms": self.residual_rms,
            "encoder_sha256": self.encoder_sha256,
            "hidden_layers": len(self.weights.hidden),
        }

    def save(self, path: Path) -> str:
        """Write the artifact and return its sha256."""
        arrays: dict[str, NDArray[Any]] = {
            "meta": np.asarray(json.dumps(self.meta, sort_keys=True)),
            "training_seeds": np.asarray(sorted(self.training_seeds), dtype=np.int64),
            "out_w": self.weights.out[0],
            "out_b": self.weights.out[1],
            "token_w": self.weights.token[0],
            "token_b": self.weights.token[1],
            "dense_mean": self.dense.mean,
            "dense_scale": self.dense.scale,
            "token_mean": self.tokens.mean,
            "token_scale": self.tokens.scale,
        }
        for i, (w, b) in enumerate(self.weights.hidden):
            arrays[f"hidden_w_{i}"], arrays[f"hidden_b_{i}"] = w, b
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(handle, **arrays)  # type: ignore[arg-type]
        return file_sha256(path)

    @classmethod
    def load(cls, path: Path) -> LearnedArtifact:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            hidden = tuple(
                (data[f"hidden_w_{i}"].astype(np.float64), data[f"hidden_b_{i}"].astype(np.float64))
                for i in range(int(meta["hidden_layers"]))
            )
            weights = TwoHeadWeights(
                hidden,
                (data["out_w"].astype(np.float64), data["out_b"].astype(np.float64)),
                (data["token_w"].astype(np.float64), data["token_b"].astype(np.float64)),
            )
            return cls(
                check_id=str(meta["check_id"]),
                unit=str(meta["unit"]),
                dense_columns=tuple(meta["dense_columns"]),
                token_columns=tuple(meta["token_columns"]),
                weights=weights,
                dense=Standardiser(data["dense_mean"], data["dense_scale"]),
                tokens=Standardiser(data["token_mean"], data["token_scale"]),
                residual_rms=float(meta["residual_rms"]),
                encoder_sha256=str(meta["encoder_sha256"]),
                training_seeds=frozenset(int(s) for s in data["training_seeds"]),
            )

    def estimate(self, dense: FloatArray, tokens: FloatArray) -> FloatArray:
        """Estimated magnitudes for raw (n, dense) and (n, tokens) rows."""
        return predict(self.weights, self.dense.apply(dense), self.tokens.apply(tokens))


def verified_file(path: str, sha256: str, what: str) -> Path:
    if not path or not sha256:
        raise ConfigError(f"learned variant: the {what} path and sha256 must both be set")
    actual = file_sha256(Path(path))
    if actual != sha256:
        raise ConfigError(f"learned variant: {what} {path} has sha256 {actual}, not {sha256}")
    return Path(path)


def require_disjoint(training: Iterable[int], evaluated: Iterable[int]) -> None:
    shared = sorted(set(training) & set(evaluated))
    if shared:
        raise LeakageError(
            f"{len(shared)} evaluated clip seed(s) were trained on, e.g. {shared[0]}"
        )


def clip_seeds(inputs: CheckInputs) -> tuple[int, ...]:
    record = inputs.clip.perturbation
    return () if record is None else (int(record.seed),)


def image_key(clip_id: str, provenance: Mapping[str, str], a: str, b: str) -> str:
    """Row key of a pair's image features: the clip, the mask and depth providers, the pair."""
    first, second = sorted((a, b))
    return "|".join(
        (clip_id, provenance.get("track", ""), provenance.get("depth", ""), first, second)
    )


@dataclass(frozen=True)
class SubjectFeatures:
    """A default result with the subject's track features and the pairs its image rows name."""

    base: CheckResult
    track: FloatArray
    pairs: Sequence[tuple[str, str]]


def pair_image(
    table: FeatureTable,
    keys: Sequence[str],
    dense_columns: Sequence[str],
    token_columns: Sequence[str],
) -> tuple[FloatArray, FloatArray] | None:
    """Dense image features as the max over the found rows, tokens as their mean; None when no
    key has a row."""
    rows = [row for key in keys if (row := table.row(key)) is not None]
    if not rows:
        return None
    index = {c: i for i, c in enumerate(table.columns)}
    stacked = np.vstack(rows)
    dense = stacked[:, [index[c] for c in dense_columns]]
    tokens = stacked[:, [index[c] for c in token_columns]]
    with np.errstate(all="ignore"):
        return np.nanmax(dense, axis=0), np.nanmean(tokens, axis=0)


def model_inputs(
    track: FloatArray, image: tuple[FloatArray, FloatArray] | None, n_dense: int, n_tokens: int
) -> tuple[FloatArray, FloatArray]:
    """Dense row (track, has-image flag, image dense) and token row; a missing image is NaN."""
    dense, tokens = (
        image if image is not None else (np.full(n_dense, math.nan), np.full(n_tokens, math.nan))
    )
    return np.concatenate([track, [float(image is not None)], dense]), tokens


def dense_layout(track_columns: Sequence[str], image_columns: Sequence[str]) -> tuple[str, ...]:
    return (*track_columns, HAS_IMAGE, *image_columns)


class LearnedMixin:
    """``evaluate`` of a learned variant: the default check's results, each re-scored by the
    artifact; ``subjects`` (the variant's) gives each result's features."""

    learned_params: LearnedFields
    TRACK_COLUMNS: ClassVar[tuple[str, ...]]
    check_id: ClassVar[str]
    magnitude_unit: ClassVar[str]

    def subjects(self, inputs: CheckInputs) -> list[SubjectFeatures]:
        raise NotImplementedError

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        return [self._rescored(inputs, subject) for subject in self.subjects(inputs)]

    @cached_property
    def artifact(self) -> LearnedArtifact:
        p = self.learned_params
        path = verified_file(p.artifact_path, p.artifact_sha256, "artifact")
        artifact = LearnedArtifact.load(path)
        if artifact.dense_columns[: len(self.TRACK_COLUMNS) + 1] != (
            *self.TRACK_COLUMNS,
            HAS_IMAGE,
        ):
            raise ConfigError("learned variant: the artifact's track columns are not this check's")
        return artifact

    @cached_property
    def image_table(self) -> FeatureTable:
        p = self.learned_params
        return FeatureTable.load(verified_file(p.features_path, p.features_sha256, "features"))

    def _rescored(self, inputs: CheckInputs, subject: SubjectFeatures) -> CheckResult:
        artifact, base = self.artifact, subject.base
        require_disjoint(artifact.training_seeds, clip_seeds(inputs))
        check = cast("type[Check]", type(self))
        image_columns = artifact.dense_columns[len(self.TRACK_COLUMNS) + 1 :]
        keys = [image_key(inputs.clip.clip_id, inputs.provenance, a, b) for a, b in subject.pairs]
        image = pair_image(self.image_table, keys, image_columns, artifact.token_columns)
        if image is None and base.abstain_reason is not None:
            return abstain(
                check, inputs, base.subject, base.abstain_reason, base.frame_span, base.details
            )
        dense, tokens = model_inputs(
            subject.track, image, len(image_columns), len(artifact.token_columns)
        )
        value = float(artifact.estimate(dense[None, :], tokens[None, :])[0])
        details = {
            **base.details,
            "estimated": True,
            "sigma_source": SIGMA_SOURCE,
            "base_magnitude": base.magnitude,
            "image_pairs": 0 if image is None else len(subject.pairs),
        }
        span = base.frame_span
        return unscored(check, inputs, base.subject, value, artifact.residual_rms, span, details)
