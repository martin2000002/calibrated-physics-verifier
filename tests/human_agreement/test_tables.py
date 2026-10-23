"""VideoPhy table helpers: lexical rule, URL-to-member mapping, ids and binarisation."""

from __future__ import annotations

import math

import pytest

from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.datasets._videophy_tables import (
    NO_RIGID_EVENT,
    FilterSettings,
    VideophySettings,
    clip_id_of,
    generator_of,
    has_term,
    lexical_reason,
    member_map,
    tokens,
    url_member,
)
from physverify.human_agreement.comparison_items import implausible

RULES = FilterSettings(
    rigid_event=("falls", "rolls", "knocked over"),
    human_agent=("man", "hand"),
    nonrigid=("cloth",),
    fracture=("shatters",),
    fluid=("water",),
)


@pytest.mark.parametrize(
    ("caption", "reason"),
    [
        ("A ball rolls across the table.", None),
        ("A BALL ROLLS.", None),
        ("A man's ball rolls away.", "human_agent"),
        ("A ball rolls into the water", "fluid"),
        ("A glass falls and shatters.", "fracture"),
        ("A cloth falls.", "nonrigid"),
        ("A cup sits on a shelf.", NO_RIGID_EVENT),
        ("The vase was knocked over.", None),
        ("The vase was knocked, then over it went.", NO_RIGID_EVENT),
        ("A ball enrolls in school.", NO_RIGID_EVENT),
        ("A handle falls.", None),
    ],
)
def test_the_lexical_rule_matches_whole_tokens_and_token_sequences(
    caption: str, reason: str | None
) -> None:
    assert lexical_reason(caption, RULES) == reason


def test_tokens_split_on_anything_but_letters_and_digits() -> None:
    assert tokens("Wind-up toy's 2nd spin!") == ("wind", "up", "toy", "s", "2nd", "spin")
    assert has_term(tokens("it was knocked over"), "knocked over")
    assert not has_term(tokens("anything"), "")


def test_member_map_keeps_plus_signs_and_cuts_at_the_first_semicolon() -> None:
    urls = [
        "https://x.org/gen/a+b.mp4",
        "https://x.org/gen/c;d.mp4",
        "https://x.org/gen/e;1.mp4",
        "https://x.org/gen/e;2.mp4",
        "https://x.org/gen/missing.mp4",
    ]
    names = ["p/gen/a+b.mp4", "p/gen/c", "p/gen/e"]
    assert member_map(urls, names, "p/") == {
        "https://x.org/gen/a+b.mp4": "p/gen/a+b.mp4",
        "https://x.org/gen/c;d.mp4": "p/gen/c",
    }
    assert url_member("https://x.org/gen/c;d.mp4?x=1", "") == "gen/c;d.mp4"


def test_clip_ids_are_short_stable_hashes_of_the_url() -> None:
    clip = clip_id_of("v1", "https://x.org/a.mp4")
    assert clip == clip_id_of("v1", "https://x.org/a.mp4")
    assert clip.startswith("v1-") and len(clip) == 15
    assert clip != clip_id_of("v2", "https://x.org/a.mp4")


def test_the_generator_comes_from_its_column_or_the_url_directory() -> None:
    record = {"video_url": "https://x.org/wan2.1/a.mp4", "model_name": "cogvideo"}
    assert generator_of(record, "model_name") == "cogvideo"
    assert generator_of(record, "") == "wan2.1"


def test_below_the_published_cut_is_implausible_and_nan_has_no_label() -> None:
    assert implausible(0.0, 1.0) is True
    assert implausible(1.0, 1.0) is False
    assert implausible(3.0, 4.0) is True
    assert implausible(math.nan, 4.0) is None


def test_unknown_subsets_are_usage_errors(settings: Settings) -> None:
    videophy = settings.section("videophy", VideophySettings)
    assert videophy.parts_of("smoke") == videophy.subsets["h2_dev"]
    with pytest.raises(UsageError, match="known"):
        videophy.parts_of("nope")
