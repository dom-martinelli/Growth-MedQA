"""Data-quality checks for the question pools this repo curates from.

Why this is a library and not just a test file: these checks belong in the
pipeline as a pre-flight gate, not only in CI. A corrupt or mis-keyed item that
reaches the scoring stage costs real model calls and quietly moves an accuracy
number, and by then the only evidence is a figure that looks slightly wrong.
Running the same functions from the pipeline and from the tests means the gate
and the regression suite can never drift apart.

Each check answers a defect that was actually found in a public dataset:

  rt_deletions          HuggingFace's openlifescienceai/medmcqa deletes the
                        letters "rt" inside words: heart -> hea, birth -> bih,
                        artery -> aery, important -> impoant. It is in the
                        source rows, it affects roughly one item in ten, and
                        the corrupt spelling is sometimes commoner than the
                        correct one, so keyword curation silently misses items.

  keyed_answer_outside_options
                        Some MedXpertQA items carry a label that is not one of
                        the options offered. Such an item is unscoreable: every
                        model is wrong by construction, so leaving it in drags
                        every accuracy figure down by the same amount and the
                        panel still looks internally consistent.

  duplicate_questions   The same stem reached the pool from two source
                        datasets. A duplicate is double-weighted in any
                        aggregate, and if the two copies disagree on the key,
                        one of them is scoring the model wrong.

  scores_in_unit_interval
                        An accuracy or proportion outside [0, 1] means the
                        denominator moved: usually a merge that dropped rows
                        after the numerator was computed.

Run as a gate:
    python -m qa.dataset_checks pediatric_qa_outputs --strict
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

# Words whose correct spelling contains "rt". The key is what the corruption
# leaves behind; the value is what it should have been.
#
# Precision is the whole game here, because this table feeds a committed
# baseline: a token that also occurs legitimately inflates the count forever and
# the inflation is invisible once it is the baseline. So a token qualifies only
# if it is not an English word AND not a medical abbreviation.
#
# Two candidates were removed after reading the matched text rather than the
# counts, which is the only way this kind of error is visible:
#   "abo"  (from "abort") collides with ABO incompatibility, which is all over
#          neonatal jaundice questions. Every one of its matches was correct ABO.
#   "pas"  (from "parts") collides with the PAS stain ("cytoplasm is PAS
#          positive"). Its matches were genuinely mixed, so it is out: an
#          ambiguous token in a gate is worse than a missed defect.
RT_DELETIONS: Mapping[str, str] = {
    "hea": "heart",
    "heas": "hearts",
    "bih": "birth",
    "bihs": "births",
    "sho": "short",
    "shoness": "shortness",
    "aery": "artery",
    "aeries": "arteries",
    "aerial": "arterial",
    "aeriovenous": "arteriovenous",
    "aoa": "aorta",
    "impoant": "important",
    "repoed": "reported",
    "stas": "starts",
    "staed": "started",
    "coex": "cortex",
    "coical": "cortical",
    "paracoical": "paracortical",
    "glucocoicoids": "glucocorticoids",
    "coicosteroids": "corticosteroids",
    "quaer": "quarter",
    "suppo": "support",
    "inseion": "insertion",
    "inseed": "inserted",
    "veebra": "vertebra",
    "veebral": "vertebral",
    "moality": "mortality",
    "poal": "portal",
    "hypeension": "hypertension",
    "infeility": "infertility",
}

# Tokens deliberately excluded, kept here so the decision is not quietly undone
# by someone adding them back. Tested, not just commented.
RT_REJECTED: Mapping[str, str] = {
    "abo": "collides with ABO incompatibility",
    "pas": "collides with the PAS stain",
    "pa": "collides with pulmonary artery, PA view",
    "ot": "collides with occupational therapy",
    "ca": "collides with carcinoma, calcium",
}

DEFAULT_TEXT_COLUMNS = (
    "question",
    "unified_question",
    "opa",
    "opb",
    "opc",
    "opd",
    "ope",
    "options",
    "exp",
    "explanation",
    "answer",
)


@dataclass
class Finding:
    """One defect, in the shape a bug report needs: what, where, how much."""

    check: str
    detail: str
    count: int
    examples: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        head = f"[{self.check}] {self.detail}: {self.count}"
        if self.examples:
            head += "\n    e.g. " + "\n    e.g. ".join(self.examples[:3])
        return head


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def read_rows(path: Path) -> Iterator[dict[str, str]]:
    """Stream a CSV. csv, never splitlines: these stems contain embedded
    newlines, so a line count is not a row count (one 7,457-row file counts as
    14,834 lines)."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def row_text(row: Mapping[str, str], columns: Sequence[str] = DEFAULT_TEXT_COLUMNS) -> str:
    return " ".join(str(row.get(c) or "") for c in columns)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def rt_deletions(rows: Iterable[Mapping[str, str]], *, id_column: str = "id") -> list[Finding]:
    """Rows whose text has lost an "rt" inside a word.

    Matched on word boundaries, so "hea" fires and "healthy" does not.
    """
    hits: Counter[str] = Counter()
    examples: defaultdict[str, list[str]] = defaultdict(list)
    patterns = {bad: re.compile(rf"\b{re.escape(bad)}\b") for bad in RT_DELETIONS}
    affected: set[str] = set()
    total = 0

    for i, row in enumerate(rows):
        total += 1
        text = row_text(row).lower()
        rid = str(row.get(id_column) or f"row {i}")
        for bad, pattern in patterns.items():
            if pattern.search(text):
                hits[bad] += 1
                affected.add(rid)
                if len(examples[bad]) < 3:
                    m = pattern.search(text)
                    lo, hi = max(0, m.start() - 40), m.end() + 40
                    examples[bad].append(f"{rid}: ...{text[lo:hi]}...")

    findings = [
        Finding(
            check="rt_deletions",
            detail=f"{bad!r} where {RT_DELETIONS[bad]!r} belongs",
            count=n,
            examples=examples[bad],
        )
        for bad, n in hits.most_common()
    ]
    if findings:
        findings.insert(
            0,
            Finding(
                check="rt_deletions",
                detail=f"rows with at least one deletion (of {total})",
                count=len(affected),
            ),
        )
    return findings


def _parse_options(raw: str | None) -> dict[str, str]:
    """Options arrive as a python dict literal in a CSV cell. literal_eval, not
    eval: the file is data, and data does not get to run."""
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(parsed, dict):
        return {str(k).strip(): str(v) for k, v in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        letters = "ABCDEFGHIJKLMNOP"
        return {letters[i]: str(v) for i, v in enumerate(parsed) if i < len(letters)}
    return {}


# Every source names its options and its key differently, and getting the
# mapping wrong is not a cosmetic problem: reading the wrong column makes every
# item look unkeyed, and reading the right column with the wrong convention
# makes every model look wrong by exactly one position.
#
#   medmcqa          opa..opd        + cop = integer index
#   medqa            options dict    + answer_idx = letter (answer = text)
#   medxpertqa       options dict    + label = letter
#   pediatricsmqa    options dict    + answer = letter
#   medbullets       choicesA..E     + answer_idx = letter (answer = text)
#   afrimedqa v2     answer_options  + correct_answer = "optionN"
#
# cop is 0-based: the observed values across the pediatric MedMCQA extract are
# exactly {0, 1, 2, 3} with four options per item, which 1-based indexing could
# not produce. verify_index_convention() re-checks that rather than trusting
# this comment, because some mirrors of this dataset are 1-based.
OPTION_CELLS = ("options", "answer_options", "choices")
OPTION_PREFIXES = ("op", "choices", "option")
KEY_COLUMNS = ("label", "answer_idx", "cop", "correct_answer", "answer", "cor_option")

LETTERS = "ABCDEFGHIJ"


def options_for(row: Mapping[str, str]) -> dict[str, str]:
    """The option set, from whichever column shape this source uses."""
    for cell in OPTION_CELLS:
        if row.get(cell):
            parsed = _parse_options(row[cell])
            if parsed:
                return parsed
    for prefix in OPTION_PREFIXES:
        lettered = {}
        for letter in "abcdefghij":
            for name in (f"{prefix}{letter}", f"{prefix}{letter.upper()}"):
                v = row.get(name)
                if v not in (None, ""):
                    lettered[letter.upper()] = str(v)
                    break
        if len(lettered) >= 2:
            return lettered
    return {}


def resolve_key(row: Mapping[str, str], options: Mapping[str, str]) -> tuple[str | None, str]:
    """The option letter this row's key points at, and which column said so.

    Returns (None, reason) when no column carries a usable key, so the caller
    can tell "this item is unkeyed" from "this item is keyed to an option that
    does not exist" — different defects with different repairs.
    """
    if not options:
        return None, "no options offered"
    letters = {k.upper(): k for k in options}
    by_text = {str(v).strip(): k for k, v in options.items()}

    for col in KEY_COLUMNS:
        raw = row.get(col)
        if raw in (None, ""):
            continue
        key = str(raw).strip().strip("'\"")

        if key.upper() in letters:
            return letters[key.upper()], col
        # "option3" -> the third option
        m = re.fullmatch(r"option[_ ]?(\d+)", key, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1  # these are 1-based by their own naming
            ordered = list(options)
            return (ordered[idx], col) if 0 <= idx < len(ordered) else (None, f"{col}={key} out of range")
        # a bare integer: medmcqa's cop, 0-based
        if re.fullmatch(r"\d+", key):
            idx = int(key)
            ordered = list(options)
            return (ordered[idx], col) if 0 <= idx < len(ordered) else (None, f"{col}={key} out of range")
        if key in by_text:
            return by_text[key], col
        # a key that is text but matches nothing: report it against this column
        return None, f"{col}={key[:60]!r} matches no option"

    return None, "no key column present"


def verify_index_convention(rows: Iterable[Mapping[str, str]], *, column: str = "cop") -> list[Finding]:
    """Check that an integer key column really is 0-based.

    An off-by-one here is the most expensive silent defect in the pipeline: the
    data loads, every item scores, and every model's accuracy is wrong in the
    same direction, so nothing looks anomalous. The convention is inferred from
    the observed range instead of being assumed, and an ambiguous range is a
    finding rather than a guess.
    """
    values: set[int] = set()
    widths: set[int] = set()
    for row in rows:
        raw = row.get(column)
        if raw in (None, ""):
            continue
        if re.fullmatch(r"\d+", str(raw).strip()):
            values.add(int(str(raw).strip()))
            widths.add(len(options_for(row)))
    if not values:
        return []

    lo, hi, width = min(values), max(values), max(widths) if widths else 0
    if lo == 0 and hi == width - 1:
        return []
    if lo == 1 and hi == width:
        return [
            Finding(
                "verify_index_convention",
                f"{column} looks 1-based (range {lo}-{hi} over {width} options), but it is read as 0-based",
                hi - lo + 1,
            )
        ]
    return [
        Finding(
            "verify_index_convention",
            f"{column} range {lo}-{hi} does not fit {width} options under either convention",
            hi - lo + 1,
        )
    ]


def keyed_answer_outside_options(
    rows: Iterable[Mapping[str, str]],
    *,
    id_column: str = "id",
) -> list[Finding]:
    """Items whose answer key names an option that was never offered.

    These are unscoreable, not merely wrong: no model can select the key, so
    they subtract a constant from every model's accuracy. In the curated set
    this is the exclusion rule, and it is the reason the curated count and the
    scored count differ.

    Rows that offer no options at all are skipped rather than counted: the open
    questions in AfriMed-QA have no key by design, and counting them here would
    report three quarters of that file as defective.
    """
    bad: list[str] = []
    empty: list[str] = []
    empty_slot: list[str] = []
    total = 0
    scoreable = 0

    for i, row in enumerate(rows):
        total += 1
        rid = str(row.get(id_column) or row.get("sample_id") or f"row {i}")
        options = options_for(row)
        if not options:
            continue  # open-ended item, not an MCQ
        scoreable += 1

        letter, how = resolve_key(row, options)
        if letter is None:
            if how in {"no key column present"}:
                empty.append(rid)
            else:
                bad.append(f"{rid}: {how}, options {sorted(options)}")
            continue
        if not str(options[letter]).strip():
            # the key points at an option slot that exists but is blank
            empty_slot.append(f"{rid}: key {letter} is an empty option")

    findings = []
    if bad:
        findings.append(
            Finding("keyed_answer_outside_options", f"unscoreable items (of {scoreable} with options)", len(bad), bad)
        )
    if empty:
        findings.append(
            Finding("keyed_answer_outside_options", "items with no key", len(empty), empty)
        )
    if empty_slot:
        findings.append(
            Finding("keyed_answer_outside_options", "keys pointing at a blank option", len(empty_slot), empty_slot)
        )
    return findings


def is_scoreable(row: Mapping[str, str]) -> tuple[bool, str]:
    """Can this item be scored at all?

    One definition of the gate, in one place, used by the pipeline and by the
    tests. When the rule lives in two places it eventually means two things,
    and then the curated count and the scored count come from different rules.
    """
    options = options_for(row)
    if not options:
        return False, "no options offered"
    letter, how = resolve_key(row, options)
    if letter is None:
        return False, how
    if not str(options[letter]).strip():
        return False, f"key {letter} is an empty option"
    return True, f"keyed by {how}"


def partition_scoreable(rows: Iterable[Mapping[str, str]]):
    """(kept, dropped) where dropped carries the reason, so an exclusion can be
    reported with its size and its cause instead of just happening."""
    kept, dropped = [], []
    for row in rows:
        ok, why = is_scoreable(row)
        (kept if ok else dropped).append(row if ok else (row, why))
    return kept, dropped


def duplicate_questions(
    rows: Iterable[Mapping[str, str]],
    *,
    question_column: str = "question",
    id_column: str = "id",
) -> list[Finding]:
    """Stems that appear more than once, compared on normalised text so that
    whitespace and case differences do not hide a duplicate."""
    seen: defaultdict[str, list[str]] = defaultdict(list)
    for i, row in enumerate(rows):
        q = re.sub(r"\s+", " ", str(row.get(question_column) or "")).strip().lower()
        if not q:
            continue
        seen[q].append(str(row.get(id_column) or f"row {i}"))

    dupes = {q: ids for q, ids in seen.items() if len(ids) > 1}
    if not dupes:
        return []
    examples = [f"{len(ids)}x: {q[:90]}... ({', '.join(ids[:4])})" for q, ids in list(dupes.items())[:5]]
    return [
        Finding("duplicate_questions", "stems appearing more than once", len(dupes), examples),
        Finding(
            "duplicate_questions",
            "rows that are a repeat of an earlier row",
            sum(len(ids) - 1 for ids in dupes.values()),
        ),
    ]


def ids_unique(rows: Iterable[Mapping[str, str]], *, id_column: str = "id") -> list[Finding]:
    counts = Counter(str(row.get(id_column) or "") for row in rows)
    counts.pop("", None)
    repeated = {k: v for k, v in counts.items() if v > 1}
    if not repeated:
        return []
    return [
        Finding(
            "ids_unique",
            f"{id_column} values used by more than one row",
            len(repeated),
            [f"{k} x{v}" for k, v in list(repeated.items())[:5]],
        )
    ]


def scores_in_unit_interval(
    rows: Iterable[Mapping[str, str]], columns: Sequence[str], *, id_column: str = "model"
) -> list[Finding]:
    """Any accuracy or proportion that left [0, 1].

    Checked for every column named, and for the obvious impossibilities
    separately from the merely surprising, because 1.4 is a bug and 0.0 may be
    a result.
    """
    out_of_range: list[str] = []
    unparseable: list[str] = []
    missing: list[str] = []
    for i, row in enumerate(rows):
        who = str(row.get(id_column) or f"row {i}")
        for col in columns:
            raw = row.get(col)
            # A blank score is reported, not skipped. Skipping is how a model
            # leaves a panel without anyone noticing: the table still renders,
            # the mean still computes, and the count in the caption is wrong.
            if raw is None or str(raw).strip() == "":
                missing.append(f"{who}.{col} is blank")
                continue
            try:
                v = float(str(raw).strip())
            except (TypeError, ValueError):
                unparseable.append(f"{who}.{col} = {raw!r}")
                continue
            if v != v:  # NaN, which compares false against every bound
                unparseable.append(f"{who}.{col} = NaN")
            elif not (0.0 <= v <= 1.0):
                out_of_range.append(f"{who}.{col} = {v}")

    findings = []
    if out_of_range:
        findings.append(Finding("scores_in_unit_interval", "values outside [0, 1]", len(out_of_range), out_of_range))
    if unparseable:
        findings.append(Finding("scores_in_unit_interval", "values that are not numbers", len(unparseable), unparseable))
    if missing:
        findings.append(Finding("scores_in_unit_interval", "values that are blank", len(missing), missing))
    return findings


# ---------------------------------------------------------------------------
# cli gate
# ---------------------------------------------------------------------------

ALL_CHECKS = (
    "rt_deletions",
    "keyed_answer_outside_options",
    "verify_index_convention",
    "duplicate_questions",
    "ids_unique",
)


def check_file(path: Path, checks: Sequence[str] = ALL_CHECKS) -> list[Finding]:
    """Run the row-level checks over one CSV. Rows are read once per check, so
    the file is streamed more than once rather than held in memory: these pools
    run to hundreds of megabytes."""
    findings: list[Finding] = []
    if "rt_deletions" in checks:
        findings += rt_deletions(read_rows(path))
    if "keyed_answer_outside_options" in checks:
        findings += keyed_answer_outside_options(read_rows(path))
    if "verify_index_convention" in checks:
        findings += verify_index_convention(read_rows(path))
    if "duplicate_questions" in checks:
        findings += duplicate_questions(read_rows(path))
    if "ids_unique" in checks:
        findings += ids_unique(read_rows(path))
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="a CSV, or a directory of them")
    ap.add_argument("--glob", default="*.csv", help="which files, when target is a directory")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 on any finding, so this can gate a pipeline step",
    )
    args = ap.parse_args(argv)

    files = sorted(args.target.glob(args.glob)) if args.target.is_dir() else [args.target]
    if not files:
        print(f"no files matched {args.target}/{args.glob}", file=sys.stderr)
        return 2

    total = 0
    for f in files:
        findings = check_file(f)
        total += len(findings)
        print(f"\n== {f}")
        if not findings:
            print("   clean")
        for finding in findings:
            print("  ", finding)

    print(f"\n{total} finding(s) across {len(files)} file(s)")
    return 1 if (args.strict and total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
