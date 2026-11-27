"""Scan published benchmarks on Hugging Face and write a report.

This is the continuous half of the audit. The checks in qa/dataset_checks.py
run against the committed extracts, which is a snapshot; this runs the same
checks against what the datasets actually serve today, so a defect introduced
or repaired upstream shows up without anyone re-downloading anything.

    python scripts/scan_live.py                    # the configured benchmarks
    python scripts/scan_live.py --limit 500        # a quick pass
    python scripts/scan_live.py --only medmcqa
    python scripts/scan_live.py --out docs/live_scan.json

Only mechanical checks run here: defects that are a property of the printed
item and can be re-derived by anyone from the same rows. Nothing that needs a
model's judgement is published from this script, because an unverified
judgement about somebody else's published work is not something to put on a
page automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qa.dataset_checks import (  # noqa: E402
    KEY_DEPENDENT,
    duplicate_options,
    duplicate_questions,
    key_conflicts,
    keyed_answer_outside_options,
    keys_are_published,
    multi_answer_keys,
    placeholder_options,
    rt_deletions,
    unreadable_options,
)
from qa.hf_source import FetchError, Split, read_split, splits  # noqa: E402

# Which splits to watch. Each is a benchmark a medical-AI paper is scored on.
# `limit` caps a split that is too big to read daily; None means read it all.
WATCH = [
    {"dataset": "openlifescienceai/medmcqa", "config": "default", "split": "validation", "limit": None},
    {"dataset": "openlifescienceai/medmcqa", "config": "default", "split": "test", "limit": None},
    {"dataset": "openlifescienceai/medmcqa", "config": "default", "split": "train", "limit": 5000},
    {"dataset": "GBaker/MedQA-USMLE-4-options", "config": "default", "split": "test", "limit": None},
    {"dataset": "TsinghuaC3I/MedXpertQA", "config": "Text", "split": "test", "limit": None},
    {"dataset": "adlbh/PediatricsMQA", "config": "tqa", "split": "test", "limit": None},
]

# Every check here is mechanical: same rows in, same answer out, no model.
CHECKS = (
    ("corrupted_text", rt_deletions, "letters deleted inside words"),
    ("duplicate_options", duplicate_options, "the same answer offered twice"),
    ("placeholder_options", placeholder_options, "padded with a filler choice"),
    ("unreadable_options", unreadable_options, "options that are markup with no text"),
    ("key_not_offered", keyed_answer_outside_options, "the key names no offered option"),
    ("multi_answer_keys", multi_answer_keys, "select-all items keyed as a list"),
    ("duplicate_questions", duplicate_questions, "the same question more than once"),
    ("key_conflicts", key_conflicts, "identical questions whose keys disagree"),
)


def scan_split(spec: dict) -> dict:
    split = Split(spec["dataset"], spec["config"], spec["split"])
    # ask the server for the true size first, so coverage can be stated
    try:
        for s in splits(spec["dataset"]):
            if s.config == split.config and s.split == split.split:
                split.rows = s.rows
                break
    except FetchError as e:
        return {"split": split.name, "error": str(e), "findings": {}}

    rows, coverage = read_split(split, limit=spec.get("limit"))
    out = {**coverage.as_dict(), "findings": {}}
    if coverage.error:
        return out

    # A split that withholds its answers cannot be checked for key defects, and
    # saying so is different from saying it has none.
    labelled = keys_are_published(rows)
    out["keys_published"] = labelled

    for key, fn, blurb in CHECKS:
        if key in KEY_DEPENDENT and not labelled:
            out["findings"][key] = {
                "label": blurb,
                "total": None,
                "not_applicable": "this split does not publish its answer keys",
                "detail": [],
            }
            continue
        findings = fn(rows)
        out["findings"][key] = {
            "label": blurb,
            "total": findings[0].count if findings else 0,
            "detail": [{"detail": f.detail, "count": f.count, "examples": f.examples[:5]} for f in findings],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "live_scan.json")
    ap.add_argument("--limit", type=int, default=None, help="override every split's cap, for a quick pass")
    ap.add_argument("--only", help="substring: scan only matching datasets")
    args = ap.parse_args(argv)

    todo = [w for w in WATCH if not args.only or args.only in w["dataset"]]
    if not todo:
        print(f"nothing matches --only {args.only!r}", file=sys.stderr)
        return 2

    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "https://datasets-server.huggingface.co",
        "checks": [{"key": k, "label": b} for k, _, b in CHECKS],
        "note": (
            "Mechanical checks only: every finding is a property of the printed item and can be "
            "re-derived from the same rows. Counts apply to rows_read, which is not always the "
            "whole split."
        ),
        "splits": [],
    }

    failed = 0
    for spec in todo:
        if args.limit is not None:
            spec = {**spec, "limit": args.limit}
        print(f"scanning {spec['dataset']} {spec['config']}/{spec['split']} ...", flush=True)
        result = scan_split(spec)
        report["splits"].append(result)
        if result.get("error"):
            failed += 1
            print(f"  FAILED: {result['error']}", file=sys.stderr)
            continue
        flagged = {k: v["total"] for k, v in result["findings"].items() if v["total"]}
        covered = f"{result['rows_read']} of {result['rows_in_split']}" if result["rows_in_split"] else str(result["rows_read"])
        print(f"  read {covered} rows{'' if result['complete'] else '  (SAMPLE, not the whole split)'}")
        print(f"  {flagged or 'nothing flagged'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}")

    # A failed fetch is not a clean scan. Exit non-zero so a scheduled run goes
    # red rather than quietly publishing a report with a hole in it.
    if failed:
        print(f"{failed} split(s) could not be read", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
