"""Tests over this repository's own release artifacts.

These run in CI with nothing to download: both files are committed here.

What is asserted and what is merely recorded is a deliberate split.

**Asserted** are the things that are arithmetic. The prediction grid has to be
complete, an item has to have one answer key, `is_correct` has to agree with the
two columns it is derived from, `accuracy_pct` has to equal correct/n, and every
published confidence interval has to be the Wilson interval for its own counts.
None of those can be a judgement call, and a reader treats all of them as
already checked.

**Recorded** are the things that are a choice: how many items a model failed to
answer, and how the published denominators relate to this file's scored counts.
Those live in a committed baseline, so the numbers are visible in the repository
and a change fails the test. The point is not that the current values are right;
it is that they cannot move without someone noticing.

    python -m pytest tests/test_release_artifacts.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qa.release_checks import (  # noqa: E402
    PREDICTIONS,
    SUBGROUPS,
    check_accuracy_matches_counts,
    check_is_correct_consistency,
    check_leaderboard_matches_the_table,
    check_one_key_per_item,
    check_prediction_grid,
    check_subgroup_partitions,
    check_tiers,
    check_unparsed_encoding,
    check_wilson_intervals,
    denominator_gaps,
    read_csv,
    scored_by_model,
    unparsed_by_model,
    wilson_interval,
)

BASELINE = Path(__file__).parent / "baselines" / "release_denominators.json"


@pytest.fixture(scope="module")
def predictions():
    assert PREDICTIONS.exists(), f"missing {PREDICTIONS}"
    return read_csv(PREDICTIONS)


@pytest.fixture(scope="module")
def subgroups():
    assert SUBGROUPS.exists(), f"missing {SUBGROUPS}"
    return read_csv(SUBGROUPS)


# ---------------------------------------------------------------------------
# the interval maths, checked against known values before it is trusted
# ---------------------------------------------------------------------------


class TestWilsonInterval:
    """The recomputation is only evidence if it is itself correct. These pin it
    against hand-checkable cases, so a failure in the table checks below means
    the table moved, not that this function is wrong."""

    def test_a_half_and_half_split_is_centred(self):
        lo, hi = wilson_interval(50, 100)
        assert lo < 50 < hi
        assert abs((lo + hi) / 2 - 50) < 1.0

    def test_known_value(self):
        # 90/100 -> roughly [82.6, 94.5] by the Wilson formula
        lo, hi = wilson_interval(90, 100)
        assert abs(lo - 82.6) < 0.2, lo
        assert abs(hi - 94.5) < 0.2, hi

    def test_a_perfect_score_does_not_produce_a_zero_width_interval(self):
        # the normal approximation collapses to [100, 100] here, which is the
        # reason the Wilson interval is the one the methods specify
        lo, hi = wilson_interval(40, 40)
        assert lo < 100
        assert hi == pytest.approx(100, abs=1e-6)

    def test_an_empty_subgroup_does_not_divide_by_zero(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_the_interval_always_contains_the_point_estimate(self):
        for n in (5, 17, 58, 160, 960):
            for k in (0, 1, n // 2, n - 1, n):
                lo, hi = wilson_interval(k, n)
                assert lo - 1e-9 <= 100 * k / n <= hi + 1e-9, (k, n)


# ---------------------------------------------------------------------------
# asserted: arithmetic
# ---------------------------------------------------------------------------


def test_every_model_answered_every_item_exactly_once(predictions):
    assert check_prediction_grid(predictions) == []


def test_each_item_has_one_answer_key(predictions):
    assert check_one_key_per_item(predictions) == []


def test_is_correct_agrees_with_the_columns_it_is_derived_from(predictions):
    assert check_is_correct_consistency(predictions) == []


def test_each_model_belongs_to_exactly_one_tier(predictions):
    assert check_tiers(predictions) == []


def test_published_accuracy_equals_its_own_counts(subgroups):
    assert check_accuracy_matches_counts(subgroups) == []


def test_every_published_interval_is_the_wilson_interval_for_its_counts(subgroups):
    """The one number in a supplementary table a reader cannot check by eye."""
    assert check_wilson_intervals(subgroups) == []


def test_the_three_dimensions_slice_the_same_item_set(subgroups):
    assert check_subgroup_partitions(subgroups) == []


def test_the_predictions_and_the_published_table_agree(predictions, subgroups):
    """The cross-artifact check. data/, supplementary/ and reproduce.py all have
    to describe one set of results; if one was regenerated and the others were
    not, a reader following the README gets different numbers from the paper."""
    assert check_leaderboard_matches_the_table(predictions, subgroups) == []


def test_the_table_covers_the_primary_tier_and_only_the_primary_tier(predictions, subgroups):
    primary = {r["model"] for r in predictions if r["tier"] == "primary"}
    tabled = {r["model"] for r in subgroups}
    assert tabled == primary, (
        f"in the table but not primary tier: {sorted(tabled - primary)}; "
        f"primary but not in the table: {sorted(primary - tabled)}"
    )


# ---------------------------------------------------------------------------
# recorded: choices, held to a committed baseline
# ---------------------------------------------------------------------------


def test_unparsed_answers_use_one_encoding(predictions):
    """When no answer can be parsed from a response, `is_correct` is blank.

    All 361 such rows now follow that one rule. Thirteen of them used to carry
    `0.0` instead, which meant a reader averaging the column directly got a
    different number from the paper for Claude Haiku 4.5, Gemini 2.5 Pro and
    Llama 3.3 70B, all primary tier. Blanking them changed no published number:
    reproduce.py's output is byte-for-byte identical and the subgroup table was
    not touched, because both already dropped rows with no parsed answer. The
    13 values were inert, and now they are also absent.

    This was an xfail while the decision was open. It is a plain assertion now,
    which is the point of writing the finding as a test rather than a comment:
    the day it is fixed, the test says so.
    """
    assert check_unparsed_encoding(predictions) == []


def test_the_scored_count_equals_the_published_denominator(predictions, subgroups):
    """Each model's scored rows and its published denominator are the same number.

    This is the invariant the fix above bought. Before it, three models differed
    by exactly their unparsed-but-scored rows, and the explanation had to be
    carried in a baseline file. Now it is arithmetic, so it is asserted.
    """
    gaps = {m: d for m, d in denominator_gaps(predictions, subgroups).items() if d["gap"]}
    assert not gaps, f"scored count and published denominator disagree: {gaps}"


def test_the_published_denominators_still_match_their_baseline(predictions, subgroups):
    """The gap between this file's scored count and the published denominator is
    fully explained: it is exactly the 13 unparsed-but-scored rows above. That
    explanation is worth freezing, because if the gap ever changes by a
    different amount, the explanation no longer holds and every accuracy in the
    table needs recomputing."""
    assert BASELINE.exists(), f"missing {BASELINE}; run python -m qa.release_checks"
    expected = json.loads(BASELINE.read_text())
    actual = denominator_gaps(predictions, subgroups)

    drift = {
        m: {"was": expected["per_model"].get(m), "now": d}
        for m, d in actual.items()
        if expected["per_model"].get(m) != d
    }
    assert not drift, (
        f"the published denominators moved relative to the predictions: {drift}\n"
        f"If this is intended, recompute every accuracy and interval in "
        f"supplementary/subgroup_accuracy_wilson_ci.csv and update this baseline."
    )


def test_the_unanswered_counts_still_match_their_baseline(predictions):
    """A model failing to answer is a result, not a defect, so this is recorded
    rather than asserted. What it catches is the count changing when the
    predictions did not."""
    expected = json.loads(BASELINE.read_text())["unparsed_by_model"]
    actual = unparsed_by_model(predictions)
    assert actual == expected, (
        f"unanswered-item counts changed.\n  was: {expected}\n  now: {actual}"
    )


def test_the_totals_in_the_baseline_are_self_consistent(predictions):
    """A guard on the baseline itself: a recorded number nobody re-derives is
    just a number in a file."""
    baseline = json.loads(BASELINE.read_text())
    scored = scored_by_model(predictions)
    assert sum(baseline["unparsed_by_model"].values()) == baseline["totals"]["unparsed_rows"]
    assert baseline["totals"]["prediction_rows"] == len(predictions)
    assert baseline["totals"]["models"] == len({r["model"] for r in predictions})
    assert baseline["totals"]["items"] == len({r["item_id"] for r in predictions})
    assert sum(scored.values()) == baseline["totals"]["scored_rows"]
