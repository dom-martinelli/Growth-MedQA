"""Tests for the data-quality checks, and for the data itself.

Three layers, in the order they should be trusted:

1. Controls. The detectors run against hand-written fixtures where the answer
   is known: a clean file they must call clean, and corrupt files where every
   planted defect must be found and nothing else. Without these, a reported
   "0 findings" on the real pool means nothing, because a broken regex reports
   zero too.

2. Invariants. Things that must hold for any pool we would score, regardless of
   where it came from: unique ids, a key that is one of the offered options, no
   duplicated stems.

3. Baselines. The public source pools contain defects we did not introduce and
   cannot fix. Asserting zero there would fail by design, so the recorded count
   is committed and the test fails only when it *changes* — which is the signal
   that matters: the upstream data moved under us, and any number computed from
   it is now a different number.

Run:  python -m pytest tests -q
      python -m pytest tests -q -m "not needs_data"     # fixtures only
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qa.dataset_checks import (  # noqa: E402
    RT_DELETIONS,
    duplicate_options,
    key_conflicts,
    multi_answer_keys,
    placeholder_options,
    unreadable_options,
    RT_REJECTED,
    duplicate_questions,
    ids_unique,
    is_scoreable,
    keyed_answer_outside_options,
    options_for,
    partition_scoreable,
    read_rows,
    resolve_key,
    rt_deletions,
    scores_in_unit_interval,
    verify_index_convention,
)

FIXTURES = Path(__file__).parent / "fixtures"
BASELINES = Path(__file__).parent / "baselines"
# The source pools are not in this repository: they are redistributions of
# public datasets, hundreds of megabytes, and the release ships labels and
# predictions rather than item text. So the pool directory is found at runtime,
# and the tests that read it skip when it is absent. GROWTH_MEDQA_POOLS points
# at a working copy (in the research tree, pediatric_qa_outputs/).
def _find_pools() -> Path:
    env = os.environ.get("GROWTH_MEDQA_POOLS")
    candidates = [Path(env)] if env else []
    candidates += [REPO / "pediatric_qa_outputs", REPO / "exploration" / "pediatric_qa_outputs"]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[-1]


EXTRACTS = _find_pools()


def counts(findings) -> dict[str, int]:
    """Collapse findings to {detail: count} so a test can assert on exact
    numbers rather than on string formatting."""
    return {f.detail: f.count for f in findings}


# ---------------------------------------------------------------------------
# 1. controls
# ---------------------------------------------------------------------------


class TestNegativeControl:
    """The clean fixture must come back clean from every check.

    It is seeded with near misses on purpose — healthy, heat, shoulder, part,
    starts, Shortness, Arterial, important, cortical, vertebral, portal,
    quarter — because the deletion tokens are short and a careless pattern
    (substring instead of word boundary) would match inside all of them.
    """

    def test_no_rt_deletions_reported(self):
        assert rt_deletions(read_rows(FIXTURES / "clean_items.csv")) == []

    def test_no_bad_keys_reported(self):
        assert keyed_answer_outside_options(read_rows(FIXTURES / "clean_items.csv")) == []

    def test_no_duplicates_reported(self):
        assert duplicate_questions(read_rows(FIXTURES / "clean_items.csv")) == []
        assert ids_unique(read_rows(FIXTURES / "clean_items.csv")) == []


class TestRtDeletionDetector:
    """Every planted deletion is found, and the per-token counts are exact."""

    @pytest.fixture
    def found(self):
        return counts(rt_deletions(read_rows(FIXTURES / "rt_corrupt_items.csv")))

    def test_finds_every_affected_row(self, found):
        assert found["rows with at least one deletion (of 6)"] == 6

    @pytest.mark.parametrize(
        "token,expected",
        [("hea", 2), ("bih", 2), ("aery", 1), ("impoant", 1), ("sho", 1)],
    )
    def test_counts_each_token_exactly(self, found, token, expected):
        key = f"{token!r} where {RT_DELETIONS[token]!r} belongs"
        assert found.get(key) == expected, f"{token}: {found}"

    def test_reports_nothing_it_did_not_plant(self, found):
        # one line per token plus the summary line; a sixth token would mean a
        # false positive somewhere in the fixture
        assert len(found) == 6, found

    def test_every_token_is_actually_a_deletion(self):
        for token, word in RT_DELETIONS.items():
            assert "rt" not in token, f"{token!r} still contains rt, so it is not a deletion"
            assert word.replace("rt", "", 1) == token, (
                f"{token!r} is not {word!r} with one 'rt' removed"
            )

    def test_no_token_is_an_ordinary_word(self):
        # if a plain word lands in the table it matches ordinary text forever
        # after, and once that count is the baseline the inflation is invisible
        for token in RT_DELETIONS:
            assert token not in {"part", "parts", "heat", "short", "sport", "report", "start", "pass"}

    def test_the_rejected_tokens_stay_rejected(self):
        """Both were removed after reading the matched sentences rather than the
        counts: 'abo' matched ABO incompatibility, 'pas' matched the PAS stain.
        A comment would not stop someone adding them back."""
        for token in RT_REJECTED:
            assert token not in RT_DELETIONS, f"{token!r}: {RT_REJECTED[token]}"

    def test_a_medical_abbreviation_in_clean_text_is_not_flagged(self):
        rows = [
            {"id": "abo", "question": "A neonate with ABO incompatibility and jaundice."},
            {"id": "pas", "question": "The cytoplasm is PAS positive and diastase digestible."},
            {"id": "pa", "question": "The PA view shows the pulmonary artery."},
        ]
        assert rt_deletions(rows) == []


class TestKeyDetector:
    @pytest.fixture
    def found(self):
        return counts(keyed_answer_outside_options(read_rows(FIXTURES / "bad_key_items.csv")))

    def test_finds_the_unscoreable_items(self, found):
        # KEY-1 (key E), KEY-2 (key F), KEY-5 (answer text matching no option)
        assert found["unscoreable items (of 5 with options)"] == 3

    def test_finds_the_item_with_no_key_separately(self, found):
        # KEY-3. A missing key and a wrong key are different repairs: one is a
        # gap in the source, the other is a mis-parse on our side.
        assert found["items with no key"] == 1

    def test_does_not_flag_the_valid_item(self, found):
        # KEY-4 sits in the same file, which makes this a precision test and
        # not only a recall test
        assert sum(found.values()) == 4

    def test_a_letter_key_matching_an_option_letter_is_accepted(self):
        rows = [{"id": "x", "label": "b", "options": "{'A': 'one', 'B': 'two'}"}]
        assert keyed_answer_outside_options(rows) == []

    def test_an_item_with_no_options_is_not_called_defective(self):
        """AfriMed-QA mixes open questions into the same file. Three quarters of
        that extract has no key because it is not multiple choice, and counting
        those as defects is how a check loses its credibility."""
        rows = [{"id": "open-1", "question": "Describe the workup.", "answer": "free text"}]
        assert keyed_answer_outside_options(rows) == []
        assert is_scoreable(rows[0]) == (False, "no options offered")


class TestKeyColumnResolution:
    """Each source names its key differently. Reading the wrong column makes
    every item look unkeyed; reading the right one with the wrong convention
    makes every model look wrong by one position."""

    OPTIONS = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}

    @pytest.mark.parametrize(
        "row,expected,source",
        [
            ({"label": "C"}, "C", "medxpertqa"),
            ({"answer_idx": "D"}, "D", "medqa / medbullets"),
            ({"answer": "B"}, "B", "pediatricsmqa"),
            ({"cop": "2"}, "C", "medmcqa, 0-based integer"),
            ({"cop": "0"}, "A", "medmcqa, first option"),
            ({"correct_answer": "option3"}, "C", "afrimedqa, 1-based by its own naming"),
            ({"answer": "gamma"}, "C", "key given as the option text"),
        ],
    )
    def test_each_sources_convention(self, row, expected, source):
        letter, _ = resolve_key(row, self.OPTIONS)
        assert letter == expected, f"{source}: {row} should resolve to {expected}"

    def test_an_index_past_the_end_is_not_silently_wrapped(self):
        # python would happily accept options[-1]; a key of 9 over 4 options is
        # a defect, not the last option
        letter, why = resolve_key({"cop": "9"}, self.OPTIONS)
        assert letter is None and "out of range" in why


class TestIndexConvention:
    def test_zero_based_column_passes(self):
        rows = [{"cop": str(i), "opa": "a", "opb": "b", "opc": "c", "opd": "d"} for i in range(4)]
        assert verify_index_convention(rows) == []

    def test_one_based_column_is_caught(self):
        """The off-by-one that is invisible downstream: the data loads, every
        item scores, and every model is wrong in the same direction."""
        rows = [{"cop": str(i), "opa": "a", "opb": "b", "opc": "c", "opd": "d"} for i in (1, 2, 3, 4)]
        findings = verify_index_convention(rows)
        assert findings and "1-based" in findings[0].detail

    def test_a_range_that_fits_neither_convention_is_reported(self):
        rows = [{"cop": str(i), "opa": "a", "opb": "b", "opc": "c", "opd": "d"} for i in (0, 1, 2, 3, 4)]
        findings = verify_index_convention(rows)
        assert findings and "neither" not in findings[0].detail
        assert "does not fit" in findings[0].detail


class TestOptionParsing:
    def test_reads_a_dict_literal(self):
        row = {"options": "{'A': 'one', 'B': 'two'}"}
        assert options_for(row) == {"A": "one", "B": "two"}

    def test_reads_the_lettered_column_shape(self):
        # medmcqa uses opa..opd instead of a single options cell
        row = {"opa": "one", "opb": "two", "opc": "three", "opd": "four"}
        assert sorted(options_for(row)) == ["A", "B", "C", "D"]

    def test_a_malformed_cell_yields_nothing_rather_than_raising(self):
        # a parse failure must not take the whole run down, and must not be
        # silently treated as "no bad key" either: the caller sees an empty
        # option set and skips the row
        assert options_for({"options": "{'A': 'one'"}) == {}

    def test_does_not_execute_the_cell(self):
        # literal_eval, not eval. a data file does not get to run.
        assert options_for({"options": "__import__('os').getcwd()"}) == {}


class TestScoreRange:
    def test_accepts_the_endpoints(self):
        rows = [{"model": "m", "accuracy": "0.0", "macro_f1": "1.0"}]
        assert scores_in_unit_interval(rows, ["accuracy", "macro_f1"]) == []

    @pytest.mark.parametrize("value", ["1.4", "-0.01", "62.5", "nan", "", "n/a"])
    def test_rejects_anything_that_is_not_a_proportion(self, value):
        """A blank is included on purpose. Treating it as "nothing to check"
        is how a model silently leaves a panel: the table renders, the mean
        computes, and the count in the caption is wrong."""
        rows = [{"model": "m", "accuracy": value}]
        findings = scores_in_unit_interval(rows, ["accuracy"])
        assert findings, f"{value!r} should have been reported"

    def test_a_percentage_pasted_into_a_proportion_column_is_caught(self):
        # the accuracy figures are reported as percentages in prose (62.5 to
        # 91.0) and as proportions in the tables; one of the two conventions
        # landing in the other's column is the mistake this catches
        rows = [{"model": "m", "accuracy": "91.0"}]
        assert scores_in_unit_interval(rows, ["accuracy"])

    def test_a_non_number_is_reported_separately_from_a_bad_number(self):
        rows = [{"model": "m", "accuracy": "n/a"}]
        findings = scores_in_unit_interval(rows, ["accuracy"])
        assert [f.detail for f in findings] == ["values that are not numbers"]


# ---------------------------------------------------------------------------
# 2. invariants any pool we score has to satisfy
# ---------------------------------------------------------------------------


class TestScoreableGate:
    def test_the_gate_and_the_detector_agree(self):
        """The gate used by the pipeline and the check used by the tests have to
        agree on every row. If they ever disagree, one of the two produced a
        number that was then reported."""
        rows = list(read_rows(FIXTURES / "bad_key_items.csv"))
        _, dropped = partition_scoreable(rows)
        reported = sum(counts(keyed_answer_outside_options(rows)).values())
        assert len(dropped) == reported

    def test_the_gate_keeps_a_clean_file_whole(self):
        rows = list(read_rows(FIXTURES / "clean_items.csv"))
        kept, dropped = partition_scoreable(rows)
        assert dropped == []
        assert len(kept) == len(rows)

    def test_excluded_items_are_never_silently_dropped(self):
        """The count reported has to be the count scored, and an exclusion is
        only legitimate if it is stated with its size and its cause. A pipeline
        that returns the survivors alone cannot be audited."""
        rows = list(read_rows(FIXTURES / "bad_key_items.csv"))
        kept, dropped = partition_scoreable(rows)
        assert (len(rows), len(kept), len(dropped)) == (5, 1, 4)
        # every exclusion carries a reason, not just a row
        for _, why in dropped:
            assert why and isinstance(why, str)

    def test_the_gate_is_total(self):
        """Every row lands in exactly one side. A row that falls through both
        is a row that disappears between the curated and the scored count."""
        for name in ("clean_items.csv", "bad_key_items.csv", "rt_corrupt_items.csv"):
            rows = list(read_rows(FIXTURES / name))
            kept, dropped = partition_scoreable(rows)
            assert len(kept) + len(dropped) == len(rows), name


# ---------------------------------------------------------------------------
# 3. baselines on the real pools (skipped when the data is not on this machine)
# ---------------------------------------------------------------------------

needs_data = pytest.mark.needs_data

def _pool_baselines() -> list[Path]:
    """Only the source-pool baselines, which are the ones that name a `file`.

    tests/baselines/ also holds release_denominators.json, a different schema
    owned by test_release_artifacts.py. A bare glob picked that up and failed
    with a KeyError, which is a collection bug rather than a finding, so the
    selection is by shape instead of by directory listing.
    """
    if not BASELINES.exists():
        return []
    out = []
    for p in sorted(BASELINES.glob("*.json")):
        try:
            if "file" in json.loads(p.read_text()):
                out.append(p)
        except (json.JSONDecodeError, OSError):
            continue
    return out


BASELINE_FILES = _pool_baselines()


@needs_data
@pytest.mark.parametrize("baseline_path", BASELINE_FILES, ids=lambda p: p.stem)
def test_source_pool_matches_its_recorded_baseline(baseline_path: Path):
    """The upstream data has not changed under us.

    These pools are redistributions of public datasets. Their defects are not
    ours to fix, so the committed baseline records what they contained when the
    curation ran. A change here is not necessarily a regression — it may be
    upstream finally fixing the corruption — but it always means that every
    count downstream of this file has to be recomputed before it is quoted
    again.
    """
    baseline = json.loads(baseline_path.read_text())
    data = EXTRACTS / baseline["file"]
    if not data.exists():
        pytest.skip(f"{data} not on this machine (the pools are gitignored)")

    rows = list(read_rows(data))
    assert len(rows) == baseline["rows"], (
        f"{baseline['file']}: {len(rows)} rows now, {baseline['rows']} when the baseline was taken"
    )

    actual = {
        "rt_deletion_rows": _summary(rt_deletions(rows), "rows with at least one deletion"),
        "unscoreable_items": _summary(keyed_answer_outside_options(rows), "unscoreable items"),
        "items_with_no_key": _summary(keyed_answer_outside_options(rows), "items with no key"),
        "duplicate_stems": _summary(duplicate_questions(rows), "stems appearing more than once"),
        "repeated_ids": _summary(ids_unique(rows), "values used by more than one row"),
        "duplicate_option_items": _summary(duplicate_options(rows), "same answer twice"),
        "duplicate_option_keyed": _summary(duplicate_options(rows), "both the keyed answer"),
        "placeholder_option_items": _summary(placeholder_options(rows), "padded with a filler"),
        "placeholder_is_key": _summary(placeholder_options(rows), "key IS the filler"),
        "unreadable_option_items": _summary(unreadable_options(rows), "text-free options"),
        "multi_answer_keys": _summary(multi_answer_keys(rows), "single-answer scorer cannot score"),
        "key_conflict_stems": _summary(key_conflicts(rows), "identical stems whose keys disagree"),
        "key_conflict_different_answer": _summary(key_conflicts(rows), "different answer"),
    }
    expected = baseline["counts"]
    drift = {k: (expected.get(k), v) for k, v in actual.items() if expected.get(k) != v}
    assert not drift, (
        f"{baseline['file']}: upstream data moved. was -> now: {drift}\n"
        f"If this is expected, re-run scripts/record_qa_baselines.py and commit, "
        f"and recompute anything derived from this file."
    )


def _summary(findings, needle: str) -> int:
    for f in findings:
        if needle in f.detail:
            return f.count
    return 0
