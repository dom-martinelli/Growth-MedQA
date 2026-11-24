"""Record what the public source pools contain right now, as committed baselines.

The pools under pediatric_qa_outputs/ are redistributions of public datasets.
They carry defects we did not introduce (see qa/dataset_checks.py), and the
curation already accounts for them. What must not happen is for one of those
files to change quietly: every count downstream, including the curated and
scored sizes, is computed from them.

So the counts are written down, committed, and compared on every test run.
A failure is not "the data is bad" but "the data is different from the data the
numbers came from".

    python scripts/record_qa_baselines.py                 # all known pools
    python scripts/record_qa_baselines.py medmcqa         # one of them

The baselines are small JSON files in tests/baselines/. Updating one is a
deliberate commit, with the recomputation that goes with it.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qa.dataset_checks import (  # noqa: E402
    duplicate_options,
    duplicate_questions,
    ids_unique,
    key_conflicts,
    keyed_answer_outside_options,
    multi_answer_keys,
    placeholder_options,
    read_rows,
    rt_deletions,
    unreadable_options,
)

EXTRACTS = Path(
    os.environ.get("GROWTH_MEDQA_POOLS") or REPO / "pediatric_qa_outputs"
)
BASELINES = REPO / "tests" / "baselines"

# the pools the curation actually draws on, and why each one is watched
POOLS = {
    "medmcqa": (
        "medmcqa_pediatric_questions.csv",
        "openlifescienceai/medmcqa. Carries the rt-deletion corruption in the "
        "source rows, which defeats keyword curation.",
    ),
    "medxpertqa_text": (
        "medxpertqa_text_pediatric_questions.csv",
        "TsinghuaC3I/MedXpertQA, text split. Watched for items whose key is "
        "not among the options offered.",
    ),
    "medqa": (
        "medqa_pediatric_questions.csv",
        "MedQA. The largest contributor to the pool; watched for duplicates "
        "against the other sources.",
    ),
    "afrimedqa_v2": (
        "afrimedqa_v2_pediatric_questions.csv",
        "AfriMed-QA v2. Localised from a shared parent, so duplicate stems "
        "across sources are expected and need to be counted, not assumed.",
    ),
    "medbullets": ("medbullets_pediatric_questions.csv", "Medbullets."),
    "pediatricsmqa_tqa": ("pediatricsmqa_tqa_pediatric_questions.csv", "PediatricsMQA, TQA split."),
}


def _count(findings, needle: str) -> int:
    for f in findings:
        if needle in f.detail:
            return f.count
    return 0


def record(key: str, filename: str, note: str) -> dict | None:
    path = EXTRACTS / filename
    if not path.exists():
        print(f"  skip {key}: {path} not here")
        return None

    rows = list(read_rows(path))
    rt = rt_deletions(rows)
    keys = keyed_answer_outside_options(rows)
    out = {
        "file": filename,
        "note": note,
        "recorded": date.today().isoformat(),
        "rows": len(rows),
        "counts": {
            "rt_deletion_rows": _count(rt, "rows with at least one deletion"),
            "unscoreable_items": _count(keys, "unscoreable items"),
            "items_with_no_key": _count(keys, "items with no key"),
            "duplicate_stems": _count(duplicate_questions(rows), "stems appearing more than once"),
            "repeated_ids": _count(ids_unique(rows), "values used by more than one row"),
            "duplicate_option_items": _count(duplicate_options(rows), "same answer twice"),
            "duplicate_option_keyed": _count(duplicate_options(rows), "both the keyed answer"),
            "placeholder_option_items": _count(placeholder_options(rows), "padded with a filler"),
            "placeholder_is_key": _count(placeholder_options(rows), "key IS the filler"),
            "unreadable_option_items": _count(unreadable_options(rows), "text-free options"),
            "multi_answer_keys": _count(multi_answer_keys(rows), "single-answer scorer cannot score"),
            "key_conflict_stems": _count(key_conflicts(rows), "identical stems whose keys disagree"),
            "key_conflict_different_answer": _count(key_conflicts(rows), "different answer"),
        },
    }
    BASELINES.mkdir(parents=True, exist_ok=True)
    (BASELINES / f"{key}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"  {key}: {out['rows']} rows, {out['counts']}")
    return out


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(POOLS)
    unknown = [w for w in wanted if w not in POOLS]
    if unknown:
        print(f"unknown pool(s): {unknown}. known: {sorted(POOLS)}", file=sys.stderr)
        return 2
    print(f"recording baselines into {BASELINES.relative_to(REPO)}")
    for key in wanted:
        record(key, *POOLS[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
