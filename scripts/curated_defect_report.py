"""Which of the scored items carry a known defect.

The checks in qa/dataset_checks.py audit the upstream pools. This answers the
question that actually bears on the paper: how much of that damage reached the
items the models were scored on.

It needs the harmonised pool, which is not in this repository because it holds
item text. Point it at a working copy:

    python scripts/curated_defect_report.py \\
        --pool ~/Documents/GitHub/Growth-MedQA/pediatric_qa_outputs/clean_growthmedqa/\\
postnatal_pediatric_growth_prefilter_positive_1555.csv

The join is on `harmonized_id`, which is the same string as `item_id` in
data/growth_medqa_per_model_predictions.csv. That matters: the per-source
extracts use their own ids (MedMCQA UUIDs, AfriMed-QA sha hashes) and share no
id with the curated set at all. Joining those instead returns zero defects for
every check, which reads exactly like a clean result. The script asserts the
join covered every scored item rather than trusting it.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qa.dataset_checks import (  # noqa: E402
    MARKUP_ONLY,
    PLACEHOLDER_OPTIONS,
    RT_DELETIONS,
    _norm_option,
    read_rows,
)

PREDICTIONS = REPO / "data" / "growth_medqa_per_model_predictions.csv"
RT_PATTERNS = {bad: re.compile(rf"\b{re.escape(bad)}\b") for bad in RT_DELETIONS}


def scored_item_ids() -> set[str]:
    return {r["item_id"] for r in read_rows(PREDICTIONS)}


def defects_in(row: dict[str, str]) -> list[str]:
    """Every defect this one harmonised item carries.

    A list, not a label: an item can be both corrupted and unanswerable, and
    the counts below therefore overlap and must not be summed.
    """
    out = []
    try:
        options = json.loads(row.get("options_json") or "{}")
    except json.JSONDecodeError:
        options = {}
    options = {k: str(v) for k, v in options.items()} if isinstance(options, dict) else {}
    key = (row.get("correct_answer_label") or "").strip()

    haystack = " ".join(
        [row.get("question_stem") or "", row.get("explanation") or "", *options.values()]
    ).lower()
    if any(p.search(haystack) for p in RT_PATTERNS.values()):
        out.append("rt-deletion in the item text")

    texts = [_norm_option(v) for v in options.values()]
    counts = Counter(t for t in texts if t)
    repeated = {t for t, n in counts.items() if n > 1}
    if repeated:
        out.append("duplicate option text")
        if key in options and _norm_option(options[key]) in repeated:
            out.append("the key points at text offered twice")

    if any(t in PLACEHOLDER_OPTIONS for t in texts):
        out.append("padded with a filler option")
        if key in options and _norm_option(options[key]) in PLACEHOLDER_OPTIONS:
            out.append("the key IS the filler option")

    if sum(1 for v in options.values() if MARKUP_ONLY.match(v)) >= 2:
        out.append("two or more options are markup with no text")

    if key and key not in options:
        out.append("the key names no offered option")
    if "," in key:
        out.append("multi-answer key")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True, help="the harmonised pool csv")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any scored item carries a defect")
    args = ap.parse_args(argv)

    if not args.pool.exists():
        print(f"pool not found: {args.pool}", file=sys.stderr)
        return 2

    scored = scored_item_ids()
    pool = {r["harmonized_id"]: r for r in read_rows(args.pool) if r.get("harmonized_id")}

    covered = scored & set(pool)
    missing = scored - set(pool)
    print(f"scored items: {len(scored)}")
    print(f"pool rows: {len(pool)}")
    print(f"joined: {len(covered)}")
    if missing:
        # the guard that stops a broken join reading as a clean result
        print(
            f"\nWARNING: {len(missing)} scored items are not in this pool, so the "
            f"counts below cover only {len(covered)} of {len(scored)}. "
            f"e.g. {sorted(missing)[:3]}",
            file=sys.stderr,
        )

    tally: Counter[str] = Counter()
    by_source: defaultdict[str, Counter[str]] = defaultdict(Counter)
    affected: set[str] = set()
    examples: defaultdict[str, list[str]] = defaultdict(list)
    for item_id in sorted(covered):
        row = pool[item_id]
        found = defects_in(row)
        if not found:
            continue
        affected.add(item_id)
        source = item_id.split("::", 1)[0]
        for d in found:
            tally[d] += 1
            by_source[source][d] += 1
            if len(examples[d]) < 3:
                examples[d].append(item_id)

    print(f"\nscored items carrying at least one defect: {len(affected)} of {len(covered)} "
          f"({100 * len(affected) / max(1, len(covered)):.1f}%)")
    print("\nby defect (an item can carry several, so these overlap and do not sum):")
    for d, n in tally.most_common():
        print(f"  {n:4d}  {d}")
        print(f"        e.g. {', '.join(examples[d])}")
    print("\nby source dataset:")
    for source, c in sorted(by_source.items()):
        print(f"  {source}: {dict(c.most_common())}")

    return 1 if (args.strict and affected) else 0


if __name__ == "__main__":
    raise SystemExit(main())
