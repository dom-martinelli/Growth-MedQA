"""Controls for the live scanner.

None of these touch the network. A test that needs Hugging Face to be up is a
test that goes red for reasons that have nothing to do with the code, and the
whole point of this suite is that red means something.

What is worth pinning here is the behaviour that decides whether a report can
be trusted: a failed fetch must raise rather than return nothing, a throttle
must be waited out rather than treated as an error, and a split that withholds
its answers must be reported as unlabelled rather than as entirely defective.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qa import hf_source  # noqa: E402
from qa.dataset_checks import keys_are_published  # noqa: E402
from qa.hf_source import Coverage, FetchError, Split, iter_rows, read_split  # noqa: E402


# ---------------------------------------------------------------------------
# the unlabelled-split guard
# ---------------------------------------------------------------------------


class TestKeysArePublished:
    OPTIONS = {"opa": "one", "opb": "two", "opc": "three", "opd": "four"}

    def test_a_labelled_split_is_labelled(self):
        rows = [{**self.OPTIONS, "cop": i % 4} for i in range(20)]
        assert keys_are_published(rows) is True

    def test_medmcqa_test_split_shape_is_recognised_as_unlabelled(self):
        """MedMCQA withholds its test answers by writing cop = -1 on every row.

        Before this guard the key check flagged 100% of that split, which is
        not a finding about the data: it is the scanner failing to notice there
        was nothing to check.
        """
        rows = [{**self.OPTIONS, "cop": -1} for _ in range(200)]
        assert keys_are_published(rows) is False

    def test_a_few_broken_keys_do_not_make_a_split_unlabelled(self):
        """The threshold is deliberately high. A labelled split with some bad
        keys is exactly the case worth reporting, so it must not be waved
        through as 'no keys published'."""
        rows = [{**self.OPTIONS, "cop": i % 4} for i in range(95)]
        rows += [{**self.OPTIONS, "cop": -1} for _ in range(5)]
        assert keys_are_published(rows) is True

    def test_rows_with_no_options_do_not_count_either_way(self):
        rows = [{"question": "free text, no options"} for _ in range(10)]
        assert keys_are_published(rows) is False


# ---------------------------------------------------------------------------
# fetching: failures must be loud
# ---------------------------------------------------------------------------


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def fake_page(rows, monkeypatch):
    """Serve a fixed page of rows in place of the network."""
    payload = json.dumps({"rows": [{"row": r} for r in rows]}).encode()
    monkeypatch.setattr(hf_source.urllib.request, "urlopen", lambda *a, **k: FakeResponse(payload))


class TestFetchFailsLoudly:
    def test_a_permanent_error_raises_rather_than_returning_empty(self, monkeypatch):
        """A helper that returns nothing on failure manufactures false
        negatives: the caller counts zero defects and cannot tell it counted
        zero rows."""
        def boom(*a, **k):
            raise urllib.error.HTTPError("u", 404, "Not Found", None, None)

        monkeypatch.setattr(hf_source.urllib.request, "urlopen", boom)
        with pytest.raises(FetchError):
            hf_source._get("rows", {"dataset": "x"})

    def test_it_does_not_retry_a_404(self, monkeypatch):
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise urllib.error.HTTPError("u", 404, "Not Found", None, None)

        monkeypatch.setattr(hf_source.urllib.request, "urlopen", boom)
        with pytest.raises(FetchError):
            hf_source._get("rows", {"dataset": "x"})
        assert len(calls) == 1, "a 404 will not become a 200 by asking again"

    def test_a_throttle_is_waited_out_and_then_succeeds(self, monkeypatch):
        """429 is the normal response to reading a large split unauthenticated.
        It is a pause, not a failure, and treating it as one was what made the
        first full scan report three splits as unreadable."""
        slept: list[float] = []
        monkeypatch.setattr(hf_source.time, "sleep", slept.append)
        state = {"n": 0}

        def flaky(*a, **k):
            state["n"] += 1
            if state["n"] < 3:
                raise urllib.error.HTTPError("u", 429, "Too Many Requests", {"Retry-After": "2"}, None)
            return FakeResponse(json.dumps({"rows": []}).encode())

        monkeypatch.setattr(hf_source.urllib.request, "urlopen", flaky)
        assert hf_source._get("rows", {"dataset": "x"}) == {"rows": []}
        assert slept, "a throttle should have paused before retrying"
        assert max(slept) >= 2, "Retry-After should be honoured rather than guessed at"

    def test_read_split_reports_the_error_instead_of_an_empty_result(self, monkeypatch):
        def boom(*a, **k):
            raise urllib.error.HTTPError("u", 404, "Not Found", None, None)

        monkeypatch.setattr(hf_source.urllib.request, "urlopen", boom)
        rows, cov = read_split(Split("d", "c", "s"))
        assert rows == []
        assert cov.error, "an unread split must carry its error, not look clean"
        assert cov.complete is False


# ---------------------------------------------------------------------------
# coverage: a sample must never look like a whole split
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_a_capped_read_is_marked_incomplete(self, monkeypatch):
        fake_page([{"id": i} for i in range(100)], monkeypatch)
        split = Split("d", "c", "s", rows=5000)
        rows, cov = read_split(split, limit=100)
        assert len(rows) == 100
        assert cov.rows_in_split == 5000
        assert cov.complete is False, "100 of 5,000 rows is a sample and must say so"

    def test_a_full_read_is_marked_complete(self, monkeypatch):
        fake_page([{"id": i} for i in range(100)], monkeypatch)
        rows, cov = read_split(Split("d", "c", "s", rows=100), limit=100)
        assert cov.complete is True

    def test_an_unknown_split_size_is_not_claimed_as_complete_coverage(self, monkeypatch):
        fake_page([{"id": i} for i in range(100)], monkeypatch)
        _, cov = read_split(Split("d", "c", "s", rows=None), limit=100)
        # size unknown, so completeness is unknowable; the report carries None
        assert cov.rows_in_split is None

    def test_the_limit_is_respected_exactly(self, monkeypatch):
        fake_page([{"id": i} for i in range(100)], monkeypatch)
        got = list(iter_rows(Split("d", "c", "s"), limit=37, pause=0))
        assert len(got) == 37

    def test_coverage_serialises_every_field_the_page_reads(self):
        c = Coverage("d/c/s", 100, 50, False, None)
        assert set(c.as_dict()) == {"split", "rows_in_split", "rows_read", "complete", "error"}


class TestAuth:
    def test_a_token_is_optional(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        assert "Authorization" not in hf_source._headers()

    def test_a_token_is_used_when_present(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "  hf_example  ")
        assert hf_source._headers()["Authorization"] == "Bearer hf_example"
