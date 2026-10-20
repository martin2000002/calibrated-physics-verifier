"""The A8 v2 chain directory as ``run_a8v2.sh`` leaves it (state.env, steps.tsv, logs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from physverify.reporting.chain import (
    ChainState,
    Missing,
    ResultsInputError,
    abstain_curve_run,
    last_codes,
    parse_state,
)


def test_state_reads_quoted_values_and_the_last_line_wins() -> None:
    text = "CHAIN_DEVICE=cpu\nEVAL_JSON=/a\\ b/evaluation.json\n\nEVAL_JSON=/c/evaluation.json\n"
    assert parse_state(text) == {"CHAIN_DEVICE": "cpu", "EVAL_JSON": "/c/evaluation.json"}
    assert parse_state("X=/a\\ b\n") == {"X": "/a b"}
    with pytest.raises(ResultsInputError, match="unreadable"):
        parse_state("not an assignment\n")


def test_a_step_is_its_last_row() -> None:
    codes = last_codes("5\tfit\t0\t3\t-\n5\tfit\trunning\t0\t-\n6\tevaluate\t0\t1\tlog\n")
    assert codes == {5: "running", 6: "0"}


def _chain(root: Path, logs: dict[str, str]) -> ChainState:
    (root / "logs").mkdir(parents=True)
    (root / "state.env").write_text("CHAIN_DEVICE=cuda\n")
    (root / "steps.tsv").write_text("7\tabstain\t0\t1\t-\n")
    for name, text in logs.items():
        (root / "logs" / name).write_text(text)
    return ChainState.load(root)


def test_the_curve_run_is_the_one_run_line_of_abstain_curve(tmp_path: Path) -> None:
    chain = _chain(
        tmp_path,
        {
            "7-abstain-1.log": "$ pv abstain fit\nrun: /fit\n",
            "7-abstain-2.log": "$ pv abstain curve\nrun: /curve\n",
        },
    )
    assert abstain_curve_run(chain) == Path("/curve")
    assert chain.rehearsal and chain.completed(7) and not chain.completed(5)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("$ pv abstain curve\n", "no `run:`"),
        ("$ pv abstain curve\nrun: /a\nrun: /b\n", "2 ambiguous"),
    ],
)
def test_no_or_several_candidates_are_missing(tmp_path: Path, text: str, reason: str) -> None:
    found = abstain_curve_run(_chain(tmp_path, {"7-abstain-2.log": text}))
    assert isinstance(found, Missing) and found.reason.startswith(reason)


def test_a_directory_without_state_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ResultsInputError, match="not an A8 v2 chain directory"):
        ChainState.load(tmp_path)
