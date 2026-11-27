"""Read benchmark rows straight from Hugging Face, without downloading anything.

The datasets-server serves any public dataset's rows as JSON, paginated, with
no API key and no `datasets` library:

    https://datasets-server.huggingface.co/rows?dataset=...&config=...&split=...&offset=0&length=100

That is what makes a continuous check cheap enough to run on a schedule. The
alternative, pulling the parquet files, means hundreds of megabytes and a
dependency stack, for data that is read once and thrown away.

Standard library only, so the scheduled job has nothing to install and cannot
break because a numeric library changed a default.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator

SERVER = "https://datasets-server.huggingface.co"
PAGE = 100  # the server's maximum
UA = "growth-medqa-benchmark-audit (+https://github.com/dom-martinelli/Growth-MedQA)"


class FetchError(RuntimeError):
    """A request that did not succeed.

    Raised rather than swallowed. A fetch helper that returns empty on failure
    manufactures false negatives: the caller counts zero defects and cannot
    tell that it counted zero rows.
    """


@dataclass
class Split:
    dataset: str
    config: str
    split: str
    rows: int | None = None

    @property
    def name(self) -> str:
        return f"{self.dataset}/{self.config}/{self.split}"


def _headers() -> dict[str, str]:
    """A token is optional and only raises the rate limit.

    Unauthenticated reads are throttled hard enough that a full pass over a
    large split hits HTTP 429 about twenty pages in. With a token the same pass
    completes. The scanner works either way, so the job never depends on a
    secret being present, it just reads less without one.
    """
    h = {"User-Agent": UA, "Accept": "application/json"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token.strip()}"
    return h


def _get(path: str, params: dict[str, object], *, tries: int = 6, timeout: int = 60) -> dict:
    url = f"{SERVER}/{path}?" + urllib.parse.urlencode(params)
    last = ""
    for attempt in range(tries):
        req = urllib.request.Request(url, headers=_headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            # 429 and 5xx are worth waiting out; 404 and 401 are not
            if e.code not in (429, 500, 502, 503, 504):
                raise FetchError(f"{url}: {last}") from e
            if e.code == 429:
                # the server says how long to wait; believe it rather than
                # guessing, and treat a throttle as a longer pause than an error
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    wait = float(retry_after) if retry_after else 0
                except ValueError:
                    wait = 0
                time.sleep(max(wait, min(5 * 2 ** attempt, 120)))
                continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(2 ** attempt, 30))
    raise FetchError(f"{url}: gave up after {tries} tries ({last})")


def splits(dataset: str) -> list[Split]:
    """Every config and split in a dataset, with row counts where published."""
    data = _get("splits", {"dataset": dataset})
    if "error" in data:
        raise FetchError(f"{dataset}: {data['error']}")
    out = [Split(dataset, s["config"], s["split"]) for s in data.get("splits", [])]

    # row counts live on a different endpoint, and are not always present
    for cfg in {s.config for s in out}:
        try:
            info = _get("info", {"dataset": dataset, "config": cfg})
        except FetchError:
            continue
        counts = info.get("dataset_info", {}).get("splits", {})
        for s in out:
            if s.config == cfg and s.split in counts:
                s.rows = counts[s.split].get("num_examples")
    return out


def iter_rows(split: Split, *, limit: int | None = None, pause: float = 0.25) -> Iterator[dict]:
    """Stream a split's rows.

    `limit` caps how many are read. The caller must record both the cap and the
    split's true size: a scan of the first 2,000 rows of a 182,822-row dataset
    is a sample, and reporting its counts as if they covered the dataset is the
    kind of quiet overclaim that makes an audit worthless.
    """
    offset = 0
    seen = 0
    while True:
        want = PAGE if limit is None else min(PAGE, limit - seen)
        if want <= 0:
            return
        page = _get(
            "rows",
            {
                "dataset": split.dataset,
                "config": split.config,
                "split": split.split,
                "offset": offset,
                "length": want,
            },
        )
        batch = page.get("rows", [])
        if not batch:
            return
        for entry in batch:
            # enforce the cap here rather than trusting the server to honour
            # `length`. A caller that asks for 37 rows and silently gets 100
            # has a coverage figure that does not match what was read, which
            # is the one number this module exists to get right.
            if limit is not None and seen >= limit:
                return
            yield entry["row"]
            seen += 1
        offset += len(batch)
        if limit is not None and seen >= limit:
            return
        # unauthenticated reads are throttled per burst, not per hour, so a
        # short constant pause is what keeps a long split from tripping it
        if pause:
            time.sleep(pause)


@dataclass
class Coverage:
    """What was actually read, so a count can never be quoted without it."""

    split: str
    rows_in_split: int | None
    rows_read: int
    complete: bool
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "rows_in_split": self.rows_in_split,
            "rows_read": self.rows_read,
            "complete": self.complete,
            "error": self.error,
        }


def read_split(split: Split, *, limit: int | None = None) -> tuple[list[dict], Coverage]:
    """Read a split and say honestly how much of it was read."""
    rows: list[dict] = []
    error = None
    try:
        rows = list(iter_rows(split, limit=limit))
    except FetchError as e:
        error = str(e)
    complete = error is None and (split.rows is None or len(rows) >= split.rows)
    return rows, Coverage(split.name, split.rows, len(rows), complete, error)
