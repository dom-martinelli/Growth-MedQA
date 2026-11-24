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


def read_rows(path: Path | str) -> Iterator[dict[str, str]]:
    """Stream a CSV. csv, never splitlines: these stems contain embedded
    newlines, so a line count is not a row count (one 7,457-row file counts as
    14,834 lines)."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    path = Path(path)  # accept a plain string: every caller writes one eventually
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
# option-set defects
# ---------------------------------------------------------------------------

PLACEHOLDER_OPTIONS = {"n/a", "na", "none", "-", "--", "nil", "nan"}
MARKUP_ONLY = re.compile(r"^\s*(<[^>]*>\s*)+$")


def _norm_option(s: str | None) -> str:
    """An option's text, with any leading enumeration stripped.

    Some sources prefix the letter into the text ("a.\tMay be the grandchild"),
    inconsistently and only on some options, so two options that read the same
    to a human differ as strings until the prefix is removed.
    """
    return re.sub(r"\s+", " ", re.sub(r"^[a-e][.)\t ]+", "", str(s or "").strip().lower())).strip()


def duplicate_options(rows: Iterable[Mapping[str, str]], *, id_column: str = "id") -> list[Finding]:
    """Items offering the same answer twice.

    Two cases, and only the second is a scoring defect. If the duplicate is not
    the keyed option, the item is merely sloppy: there are fewer real choices
    than it appears. If the key points at text that appears twice, then two
    options are both correct and only one is accepted, so a model picking the
    other identical option is marked wrong for giving the right answer.

    MedMCQA's own tracker carries this as issue #5, "713 examples have
    duplicate choices", for the full dataset.
    """
    dupes: list[str] = []
    keyed: list[str] = []
    total = 0
    for i, row in enumerate(rows):
        options = options_for(row)
        if len(options) < 2:
            continue
        total += 1
        rid = str(row.get(id_column) or row.get("sample_id") or f"row {i}")
        texts = [_norm_option(v) for v in options.values()]
        counts = Counter(t for t in texts if t)
        repeated = {t for t, n in counts.items() if n > 1}
        if not repeated:
            continue
        dupes.append(f"{rid}: {sorted(repeated)[0][:60]!r} appears {counts[sorted(repeated)[0]]}x")
        letter, _ = resolve_key(row, options)
        if letter is not None and _norm_option(options[letter]) in repeated:
            keyed.append(f"{rid}: key {letter} points at text offered twice")

    findings = []
    if dupes:
        findings.append(Finding("duplicate_options", f"items offering the same answer twice (of {total})", len(dupes), dupes))
    if keyed:
        findings.append(
            Finding("duplicate_options", "items where two options are both the keyed answer", len(keyed), keyed)
        )
    return findings


def placeholder_options(rows: Iterable[Mapping[str, str]], *, id_column: str = "id") -> list[Finding]:
    """Options that are filler rather than an answer, usually padding a
    four-option item out to five ("n/a").

    Reported because a filler option is a distractor no reader would pick, so
    an item with one is effectively shorter than the others it is averaged
    with. Flagged separately, and much more seriously, when the key points at
    one: then the correct answer does not exist.
    """
    present: list[str] = []
    keyed: list[str] = []
    for i, row in enumerate(rows):
        options = options_for(row)
        if not options:
            continue
        rid = str(row.get(id_column) or row.get("sample_id") or f"row {i}")
        filler = {k for k, v in options.items() if _norm_option(v) in PLACEHOLDER_OPTIONS}
        if not filler:
            continue
        present.append(f"{rid}: {sorted(filler)}")
        letter, _ = resolve_key(row, options)
        if letter in filler:
            keyed.append(f"{rid}: key {letter} is {options[letter]!r}, so the answer is not offered")

    findings = []
    if present:
        findings.append(Finding("placeholder_options", "items padded with a filler option", len(present), present))
    if keyed:
        findings.append(Finding("placeholder_options", "items whose key IS the filler option", len(keyed), keyed))
    return findings


def unreadable_options(rows: Iterable[Mapping[str, str]], *, id_column: str = "id") -> list[Finding]:
    """Items whose options carry no text, only markup.

    The case that prompted this: an item in the scored set whose options A, B
    and C are each the string `<img style="max-width: 100%" src=" />`, an image
    tag with an empty source. The figure never made it into the dataset, so
    there is nothing to choose between. Such an item is not hard, it is
    unanswerable, and it still contributes to every model's accuracy.
    """
    out: list[str] = []
    for i, row in enumerate(rows):
        options = options_for(row)
        if len(options) < 2:
            continue
        rid = str(row.get(id_column) or row.get("sample_id") or f"row {i}")
        empty = [k for k, v in options.items() if MARKUP_ONLY.match(str(v))]
        if len(empty) >= 2:
            out.append(f"{rid}: {len(empty)} of {len(options)} options are markup with no text ({sorted(empty)})")
    if not out:
        return []
    return [Finding("unreadable_options", "items with two or more text-free options", len(out), out)]


def multi_answer_keys(rows: Iterable[Mapping[str, str]], *, id_column: str = "id") -> list[Finding]:
    """Select-all-that-apply items, keyed as a list.

    AfriMed-QA writes these as `correct_answer = "option2,option3,option4"`.
    They are not defective: they are a different question type. They matter
    because a single-answer harness cannot score them. Marked wrong by default,
    they drag a model's accuracy down for answering a question nobody asked it;
    dropped silently, they move the denominator without saying so.
    """
    out: list[str] = []
    sizes: Counter[int] = Counter()
    for i, row in enumerate(rows):
        options = options_for(row)
        if not options:
            continue
        letters = {k.upper() for k in options}
        for col in KEY_COLUMNS:
            raw = str(row.get(col) or "").strip()
            if "," not in raw:
                continue
            # A comma alone proves nothing: MedQA's answer column holds the
            # answer text, and "Non-enveloped with linear, double-stranded DNA"
            # is one answer. It is a multi-answer key only when every part is
            # itself an option name, which is how AfriMed-QA writes them
            # ("option2,option3").
            parts = [x.strip() for x in raw.split(",") if x.strip()]
            if len(parts) < 2:
                continue
            resolved = [p for p in parts if p in options or p.upper() in letters
                        or re.fullmatch(r"option[_ ]?\d+", p, re.IGNORECASE)]
            if len(resolved) != len(parts):
                continue
            rid = str(row.get(id_column) or row.get("sample_id") or f"row {i}")
            sizes[len(parts)] += 1
            out.append(f"{rid}: {col}={raw!r} ({len(parts)} answers)")
            break
    if not out:
        return []
    return [
        Finding(
            "multi_answer_keys",
            f"select-all items a single-answer scorer cannot score (answers per item: {dict(sorted(sizes.items()))})",
            len(out),
            out,
        )
    ]


# ---------------------------------------------------------------------------
# keys that disagree with each other
# ---------------------------------------------------------------------------


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _is_rt_twin(a: str, b: str) -> bool:
    """Is one string the other with a single "rt" deleted?

    The pair is the evidence, so no token table is needed. This matters: the
    table in RT_DELETIONS only lists words somebody thought of, and it missed
    both `peusis`/`pertusis` and `inveogram`/`invertogram` when those turned up
    as disagreeing keys. Comparing the two strings catches every instance of
    the pattern, including corruptions of words that were already misspelled.
    """
    x, y = _squash(a), _squash(b)
    lo, hi = sorted((x, y), key=len)
    if len(hi) - len(lo) != 2:
        return False
    return any(hi[i : i + 2] == "rt" and hi[:i] + hi[i + 2 :] == lo for i in range(len(hi) - 1))


def _edit_distance(a: str, b: str, cap: int) -> int:
    """Levenshtein, abandoned once it exceeds cap, so a long pair costs little."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def classify_key_disagreement(answers: Sequence[str]) -> str:
    """Why two copies of one question carry different keys.

    The raw count of disagreeing keys is not a count of contradictions, and
    reporting it as one overstates the problem by roughly a factor of three.
    Most pairs are the same answer written twice: one copy corrupted, respaced,
    truncated or mistyped. Only what survives all of those is a real
    disagreement about the medicine.
    """
    vals = sorted(set(answers))
    if any(not v.strip() for v in vals):
        return "blank key"
    squashed = {_squash(v) for v in vals}
    if len(squashed) == 1:
        return "spacing or punctuation only"
    if len(vals) == 2:
        a, b = (_squash(vals[0]), _squash(vals[1]))
        if _is_rt_twin(*vals):
            return "rt-deletion twin"
        if a.startswith(b) or b.startswith(a):
            return "one truncates the other"
        # A numeric or dosed answer never gets the typo excuse. "350 g" and
        # "350 mg" are one character apart and a thousand-fold apart, and
        # "14" and "4jan" is a spreadsheet eating a fraction. In a numeric
        # answer the character IS the answer, so these stay disagreements.
        if not any(ch.isdigit() for ch in a + b):
            cap = max(2, int(0.15 * max(len(a), len(b))))
            if _edit_distance(a, b, cap) <= cap:
                return "typo twin"
    return "different answer"


def key_conflicts(
    rows: Iterable[Mapping[str, str]],
    *,
    question_column: str = "question",
    min_stem_chars: int = 25,
) -> list[Finding]:
    """The same question, keyed two different ways.

    A duplicate is already a problem (it is double-weighted in any aggregate).
    A duplicate whose copies disagree on the answer is worse: whichever copy a
    model is scored against, one of the two scorings is wrong, and no amount of
    model skill fixes it.

    Matching is on exact normalised stem text, which makes this a lower bound
    and uncontestable. Near-duplicates need embeddings and a judgement about
    whether two differently worded items ask the same clinical question; that
    is a separate and more expensive analysis, and it finds cases this cannot.
    """
    groups: defaultdict[str, set[str]] = defaultdict(set)
    ids: defaultdict[str, set[str]] = defaultdict(set)
    for i, row in enumerate(rows):
        stem = re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(row.get(question_column) or "").lower())).strip()
        if len(stem) < min_stem_chars:
            continue
        options = options_for(row)
        letter, _ = resolve_key(row, options)
        if letter is None:
            continue
        groups[stem].add(_norm_option(options[letter]))
        ids[stem].add(str(row.get("id") or row.get("sample_id") or f"row {i}"))

    conflicting = {stem: answers for stem, answers in groups.items() if len(answers) > 1}
    if not conflicting:
        return []

    reasons: Counter[str] = Counter()
    real: list[str] = []
    for stem, answers in conflicting.items():
        why = classify_key_disagreement(sorted(answers))
        reasons[why] += 1
        if why == "different answer":
            real.append(f"{sorted(ids[stem])[:2]}: {sorted(a[:45] for a in answers)} -- {stem[:60]}...")

    findings = [
        Finding("key_conflicts", "identical stems whose keys disagree", len(conflicting)),
    ]
    for why, n in reasons.most_common():
        findings.append(Finding("key_conflicts", f"  {why}", n, real if why == "different answer" else []))
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
    "duplicate_options",
    "placeholder_options",
    "unreadable_options",
    "multi_answer_keys",
    "key_conflicts",
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
    if "duplicate_options" in checks:
        findings += duplicate_options(read_rows(path))
    if "placeholder_options" in checks:
        findings += placeholder_options(read_rows(path))
    if "unreadable_options" in checks:
        findings += unreadable_options(read_rows(path))
    if "multi_answer_keys" in checks:
        findings += multi_answer_keys(read_rows(path))
    if "key_conflicts" in checks:
        findings += key_conflicts(read_rows(path))
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
