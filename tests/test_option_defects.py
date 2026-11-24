"""Controls for the option-set and key-conflict checks.

Same discipline as the other detectors: a clean file they must call clean, and
fixtures where every planted defect is known, so "0 findings" on a real pool
means something. Every number asserted here was wrong in a first draft of the
check, and each one is a test because of that.

    python -m pytest tests/test_option_defects.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qa.dataset_checks import (  # noqa: E402
    classify_key_disagreement,
    duplicate_options,
    key_conflicts,
    multi_answer_keys,
    placeholder_options,
    read_rows,
    unreadable_options,
)

FIXTURES = Path(__file__).parent / "fixtures"
OPTIONS = FIXTURES / "option_defects.csv"
CONFLICTS = FIXTURES / "key_conflicts.csv"
CLEAN = FIXTURES / "clean_items.csv"


def counts(findings) -> dict[str, int]:
    return {f.detail: f.count for f in findings}


class TestNegativeControl:
    """None of the five fires on the clean fixture."""

    @pytest.mark.parametrize(
        "check",
        [duplicate_options, placeholder_options, unreadable_options, multi_answer_keys, key_conflicts],
        ids=lambda f: f.__name__,
    )
    def test_clean_file_is_called_clean(self, check):
        assert check(read_rows(CLEAN)) == []


class TestDuplicateOptions:
    @pytest.fixture
    def found(self):
        return counts(duplicate_options(read_rows(OPTIONS)))

    def test_finds_every_duplicate(self, found):
        # OPT-DUP-UNKEYED, OPT-DUP-KEYED, OPT-DUP-PREFIXED, and OPT-MARKUP,
        # whose three identical image tags are a duplicate as well as being
        # unreadable
        assert found["items offering the same answer twice (of 8)"] == 4

    def test_separates_the_ones_that_break_scoring(self, found):
        """Only a duplicate the key points at is a scoring defect: two options
        are both the right answer and one of them is marked wrong."""
        assert found["items where two options are both the keyed answer"] == 2

    def test_a_stray_enumeration_prefix_does_not_hide_a_duplicate(self, found):
        """OPT-DUP-PREFIXED writes one copy as "D. request the on-call..." and
        the other without the prefix. Comparing the raw strings would call them
        different, which is how this class of defect stays invisible."""
        examples = " ".join(
            e for f in duplicate_options(read_rows(OPTIONS)) for e in f.examples
        )
        assert "OPT-DUP-PREFIXED" in examples

    def test_one_item_can_carry_two_defects(self):
        """OPT-MARKUP is counted by duplicate_options and by
        unreadable_options, which is correct and worth stating: the per-check
        counts overlap, so they describe the same corpus from different angles
        and must never be added together into a total."""
        dup = " ".join(e for f in duplicate_options(read_rows(OPTIONS)) for e in f.examples)
        unread = " ".join(e for f in unreadable_options(read_rows(OPTIONS)) for e in f.examples)
        assert "OPT-MARKUP" in dup and "OPT-MARKUP" in unread


class TestPlaceholderOptions:
    @pytest.fixture
    def found(self):
        return counts(placeholder_options(read_rows(OPTIONS)))

    def test_finds_the_padding(self, found):
        assert found["items padded with a filler option"] == 2

    def test_flags_a_key_pointing_at_the_filler_separately(self, found):
        """The severe case: the key is "n/a", so the correct answer is not
        among the choices at all."""
        assert found["items whose key IS the filler option"] == 1


class TestUnreadableOptions:
    def test_finds_the_item_whose_options_are_empty_image_tags(self):
        found = counts(unreadable_options(read_rows(OPTIONS)))
        assert found["items with two or more text-free options"] == 1

    def test_one_markup_option_among_real_ones_is_not_flagged(self):
        """OPT-ONE-MARKUP has a single broken image beside three real options,
        so the item is still answerable. The threshold is two, deliberately:
        the defect being measured is an item with nothing to choose between."""
        findings = unreadable_options(read_rows(OPTIONS))
        text = " ".join(e for f in findings for e in f.examples)
        assert "OPT-ONE-MARKUP" not in text


class TestMultiAnswerKeys:
    def test_a_comma_in_the_answer_text_is_not_a_multi_answer_key(self):
        """The false positive this check shipped with: MedQA's answer column
        holds the answer as text, and "Non-enveloped with linear,
        double-stranded DNA" is one answer. It reported 94 select-all items in
        a single-answer USMLE set before the parts were required to resolve to
        options."""
        rows = [
            {
                "id": "medqa-like",
                "answer": "Non-enveloped with linear, double-stranded DNA",
                "answer_idx": "D",
                "options": "{'A': 'one', 'B': 'two', 'C': 'three', 'D': 'Non-enveloped with linear, double-stranded DNA'}",
            }
        ]
        assert multi_answer_keys(rows) == []

    def test_option_names_joined_by_commas_are_a_multi_answer_key(self):
        rows = [
            {
                "sample_id": "afri-like",
                "correct_answer": "option2,option3",
                "answer_options": "{'option1': 'a', 'option2': 'b', 'option3': 'c', 'option4': 'd'}",
            }
        ]
        findings = multi_answer_keys(rows)
        assert findings and findings[0].count == 1
        assert "2 answers" in findings[0].examples[0]


class TestKeyConflictClassifier:
    """Why two copies of one question disagree. The raw count is not a count of
    contradictions, and calling it one overstates the problem threefold."""

    @pytest.mark.parametrize(
        "answers,expected",
        [
            (["iga nephropathy", "wilms tumour"], "different answer"),
            (["craniofacial malformations", "nasal hypoplasia"], "different answer"),
            (["pertusis", "peusis"], "rt-deletion twin"),
            (["invertogram", "inveogram"], "rt-deletion twin"),
            (["panic attack", "pannic attack"], "typo twin"),
            (["meconium ileus", "meconiumileus"], "spacing or punctuation only"),
            (["vita", "vitamin a"], "one truncates the other"),
            (["", "meckels diveiculum"], "blank key"),
        ],
    )
    def test_each_reason(self, answers, expected):
        assert classify_key_disagreement(answers) == expected

    @pytest.mark.parametrize(
        "answers",
        [["350 g", "350 mg"], ["20 mgday", "5 mgday"], ["14", "4jan"], ["1500 gm", "2500 gm"]],
    )
    def test_a_numeric_answer_never_gets_the_typo_excuse(self, answers):
        """`350 g` and `350 mg` are one character apart and a thousandfold
        apart. In a numeric answer the character is the answer, so these stay
        disagreements however close the strings are."""
        assert classify_key_disagreement(answers) == "different answer"

    def test_an_rt_twin_is_found_without_a_token_table(self):
        """The table in RT_DELETIONS only holds words somebody thought of, and
        it missed both of these. Comparing the pair catches the pattern even in
        a word that was already misspelled."""
        assert classify_key_disagreement(["peusis", "pertusis"]) == "rt-deletion twin"


class TestKeyConflicts:
    @pytest.fixture
    def found(self):
        return counts(key_conflicts(read_rows(CONFLICTS)))

    def test_finds_every_disagreeing_pair(self, found):
        # six conflicting stems; KC-AGREE is the control pair that agrees
        assert found["identical stems whose keys disagree"] == 6

    def test_decomposes_them_rather_than_reporting_one_number(self, found):
        assert found["  different answer"] == 2        # the diagnosis pair and the dose pair
        assert found["  rt-deletion twin"] == 1
        assert found["  typo twin"] == 1
        assert found["  spacing or punctuation only"] == 1
        assert found["  one truncates the other"] == 1

    def test_the_agreeing_duplicate_is_not_counted(self, found):
        assert sum(v for k, v in found.items() if k.startswith("  ")) == 6
