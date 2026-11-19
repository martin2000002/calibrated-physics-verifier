"""Comparison labels: Kubric by manifest perturbation and scene, real clips clean and alone."""

from __future__ import annotations

from dataclasses import replace

from records import manifest

from physverify.baselines.labels import ClipLabel, labels_of
from physverify.synth.perturbations.none import NAME as NONE


def test_a_kubric_clip_is_perturbed_by_its_manifest_and_grouped_by_scene() -> None:
    perturbed = manifest()
    assert perturbed.perturbation is not None
    bare = replace(perturbed, clip_id="f1_0001_bare", perturbation=None)
    identity = replace(perturbed.perturbation, name=NONE)
    clean = replace(perturbed, clip_id="f1_0001__none", perturbation=identity)
    assert labels_of("kubric", [bare, clean, perturbed]) == (
        ClipLabel("f1_0001_bare", "f1_0001", False),
        ClipLabel("f1_0001__none", "f1_0001", False),
        ClipLabel(perturbed.clip_id, "f1_0001", True),
    )


def test_every_real_clip_is_clean_and_its_own_unit() -> None:
    clip = replace(manifest(), dataset="physics_iq_views")
    assert labels_of("real", [clip]) == (ClipLabel(clip.clip_id, clip.clip_id, False),)
