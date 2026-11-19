"""Checks over the release artifacts in this repository.

These are different in kind from the source-pool checks in dataset_checks.py.
Those ask whether somebody else's data is sound. These ask whether *our own
published numbers* are internally consistent: whether the per-model predictions
form a complete grid, whether every accuracy in the supplementary table is the
one its own counts imply, and whether every published confidence interval is
really the Wilson interval for that numerator and denominator.

That matters because a supplementary table is read as arithmetic the reader does
not have to redo. If an interval were computed with a different method, or from
a denominator that moved after the numerator was fixed, nothing about the table
would look wrong. Recomputing it costs milliseconds and is the only way to know.

Everything here runs on files committed to this repository, so it runs in CI on
every push with no data to fetch.

    python -m qa.release_checks           # report
    python -m qa.release_checks --strict  # exit 1 on any finding
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
PREDICTIONS = REPO / "data" / "growth_medqa_per_model_predictions.csv"
SUBGROUPS = REPO / "supplementary" / "subgroup_accuracy_wilson_ci.csv"

# 1.96 is the usual shorthand; the exact normal quantile is used so that a
# recomputation cannot disagree with the published value over rounding alone.
Z95 = 1.959963984540054

TIERS = {"primary", "additional", "excluded_reasoning_config"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def wilson_interval(correct: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval, as a percentage.

    Not the normal approximation: at the small subgroup sizes in this table
    (n = 58 in one cell) the two differ by enough to matter, and the Wilson
    interval is what the supplementary methods claim.
    """
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (centre - half) / denom, 100 * (centre + half) / denom)


# ---------------------------------------------------------------------------
# the prediction grid
# ---------------------------------------------------------------------------


def check_prediction_grid(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Every model answered every item exactly once.

    A hole in the grid is the quietest way for a leaderboard to go wrong: the
    mean is still computed, and it is computed over a different denominator for
    one model than for the rest.
    """
    out = []
    pairs = Counter((r["model"], r["item_id"]) for r in rows)
    dupes = [f"{m} / {i}" for (m, i), n in pairs.items() if n > 1]
    if dupes:
        out.append(f"{len(dupes)} (model, item) pairs appear more than once, e.g. {dupes[:3]}")

    items_by_model = defaultdict(set)
    for r in rows:
        items_by_model[r["model"]].add(r["item_id"])
    sizes = {m: len(s) for m, s in items_by_model.items()}
    if len(set(sizes.values())) > 1:
        out.append(f"models cover different numbers of items: {sorted(set(sizes.values()))}")

    all_items = set().union(*items_by_model.values()) if items_by_model else set()
    for m, s in items_by_model.items():
        missing = all_items - s
        if missing:
            out.append(f"{m} is missing {len(missing)} items, e.g. {sorted(missing)[:3]}")
    return out


def check_one_key_per_item(rows: Iterable[Mapping[str, str]]) -> list[str]:
    """An item has one answer key, whoever was being scored against it."""
    keys = defaultdict(set)
    for r in rows:
        keys[r["item_id"]].add(r["correct_answer_label"])
    bad = {i: sorted(v) for i, v in keys.items() if len(v) > 1}
    if not bad:
        return []
    return [f"{len(bad)} items carry more than one answer key, e.g. {list(bad.items())[:2]}"]


def check_is_correct_consistency(rows: Iterable[Mapping[str, str]]) -> list[str]:
    """is_correct is a derived column, so it must agree with the two columns it
    is derived from. A stored boolean that disagrees with its own inputs is the
    kind of defect that survives every spot check, because spot checks read the
    boolean."""
    out, disagree, blank_pred_with_score = [], [], []
    for r in rows:
        score = r["is_correct"].strip()
        pred = r["predicted_label"].strip()
        key = r["correct_answer_label"].strip()
        if not score:
            if pred:
                blank_pred_with_score.append(r["item_id"])
            continue
        if score not in {"0", "1", "0.0", "1.0"}:
            out.append(f"is_correct is neither 0 nor 1: {r['model']} / {r['item_id']} = {score!r}")
            continue
        if (pred == key) != (float(score) == 1.0):
            disagree.append(f"{r['model']} / {r['item_id']}: pred={pred!r} key={key!r} is_correct={score}")
    if disagree:
        out.append(f"{len(disagree)} rows where is_correct disagrees with predicted vs key, e.g. {disagree[:2]}")
    if blank_pred_with_score:
        out.append(
            f"{len(blank_pred_with_score)} rows have no score although a prediction was parsed, "
            f"e.g. {blank_pred_with_score[:3]}"
        )
    return out


def check_tiers(rows: Iterable[Mapping[str, str]]) -> list[str]:
    """A model belongs to one tier. The tier decides whether it is in the
    primary comparison, so a model in two tiers is a model counted twice."""
    out = []
    tiers = defaultdict(set)
    for r in rows:
        tiers[r["model"]].add(r["tier"])
    for m, t in tiers.items():
        if len(t) > 1:
            out.append(f"{m} appears under more than one tier: {sorted(t)}")
        unknown = t - TIERS
        if unknown:
            out.append(f"{m} has an unrecognised tier: {sorted(unknown)}")
    return out


def check_unparsed_encoding(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """One event, one encoding.

    When a model returns nothing an answer can be parsed from, this file does
    two different things: 348 of those rows carry a blank `is_correct`
    (excluded), and 13 carry `is_correct = 0.0` (counted wrong). Both are
    defensible rules. Having both in one file is not, because a reader who
    averages `is_correct` gets a different number from the one the paper
    reports, for three models, with nothing in the file to explain the
    difference. All 13 are primary-tier models, which is where it matters most.

    The gap it produces is exactly the gap between this file's scored count and
    the published denominators: Claude Haiku 4.5 by 11, Gemini 2.5 Pro by 1,
    Llama 3.3 70B by 1.
    """
    scored, excluded = [], 0
    for r in rows:
        if r["predicted_label"].strip():
            continue
        if r["is_correct"].strip():
            scored.append(f"{r['model']} / {r['item_id']} (tier {r['tier']})")
        else:
            excluded += 1
    if not scored or not excluded:
        return []
    by_model = Counter(s.split(" / ")[0] for s in scored)
    return [
        f"unparsed answers are encoded two ways: {excluded} left blank (excluded) and "
        f"{len(scored)} scored as incorrect ({dict(by_model)}). Pick one rule and apply it "
        f"to all {excluded + len(scored)}, or state the split in the methods."
    ]


def unparsed_by_model(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    """Items where no answer could be parsed from the response.

    Reported rather than asserted: a model that fails to answer is a result,
    not a defect. What would be a defect is that count changing without the
    predictions changing.
    """
    out: Counter[str] = Counter()
    for r in rows:
        if not r["predicted_label"].strip():
            out[r["model"]] += 1
    return dict(out)


def scored_by_model(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for r in rows:
        if r["is_correct"].strip():
            out[r["model"]] += 1
    return dict(out)


# ---------------------------------------------------------------------------
# the published subgroup table
# ---------------------------------------------------------------------------


def check_accuracy_matches_counts(rows: Iterable[Mapping[str, str]], tol: float = 0.011) -> list[str]:
    """accuracy_pct is correct/n. Tolerance is half a unit in the last printed
    place, so a rounding difference passes and a recomputation from different
    numbers does not."""
    out = []
    for r in rows:
        n, correct = int(r["n"]), int(r["correct"])
        if n == 0:
            out.append(f"{r['model']} / {r['subgroup']}: n = 0")
            continue
        if correct > n:
            out.append(f"{r['model']} / {r['subgroup']}: correct {correct} exceeds n {n}")
            continue
        implied = 100 * correct / n
        if abs(implied - float(r["accuracy_pct"])) > tol:
            out.append(
                f"{r['model']} / {r['dimension']} / {r['subgroup']}: "
                f"table says {r['accuracy_pct']}, {correct}/{n} is {implied:.2f}"
            )
    return out


def check_wilson_intervals(rows: Iterable[Mapping[str, str]], tol: float = 0.02) -> list[str]:
    """Every published interval is the Wilson interval for its own counts.

    This is the check worth having. An interval is the one number in a table a
    reader cannot sanity-check by eye, so an interval computed by a different
    method, or from a denominator that changed later, is invisible.
    """
    out = []
    for r in rows:
        lo, hi = wilson_interval(int(r["correct"]), int(r["n"]))
        published_lo, published_hi = float(r["ci95_low"]), float(r["ci95_high"])
        if abs(lo - published_lo) > tol or abs(hi - published_hi) > tol:
            out.append(
                f"{r['model']} / {r['dimension']} / {r['subgroup']}: "
                f"published [{published_lo}, {published_hi}], Wilson for {r['correct']}/{r['n']} "
                f"is [{lo:.2f}, {hi:.2f}]"
            )
        if not (published_lo <= float(r["accuracy_pct"]) <= published_hi):
            out.append(
                f"{r['model']} / {r['subgroup']}: point estimate {r['accuracy_pct']} "
                f"is outside its own interval [{published_lo}, {published_hi}]"
            )
    return out


def check_subgroup_partitions(rows: Iterable[Mapping[str, str]]) -> list[str]:
    """Within one model, each dimension slices the same set of items, so the
    three dimensions must sum to the same total. If they do not, an item is
    labelled on one axis and not another, and the slices are not comparable."""
    sums: defaultdict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        sums[(r["model"], r["dimension"])] += int(r["n"])
    by_model = defaultdict(dict)
    for (model, dim), total in sums.items():
        by_model[model][dim] = total
    out = []
    for model, dims in sorted(by_model.items()):
        if len(set(dims.values())) > 1:
            out.append(f"{model}: dimensions disagree on the item count: {dims}")
    return out


def check_leaderboard_matches_the_table(
    predictions: Sequence[Mapping[str, str]], subgroups: Sequence[Mapping[str, str]], tol: float = 0.011
) -> list[str]:
    """The predictions, the supplementary table and reproduce.py agree.

    This is the cross-artifact check: recompute each primary model's accuracy
    from data/ under the rule the scripts actually apply (an item counts only if
    an answer was parsed from the response), then compare it against the totals
    implied by supplementary/. If those two ever disagree, one of the three
    published artifacts was regenerated and the others were not, and a reader
    following the README gets different numbers from the paper.
    """
    out = []
    scored: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, n]
    for r in predictions:
        if r["tier"] != "primary" or not r["predicted_label"].strip():
            continue
        cell = scored[r["model"]]
        cell[1] += 1
        if r["is_correct"].strip() and float(r["is_correct"]) == 1.0:
            cell[0] += 1

    totals: defaultdict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in subgroups:
        cell = totals[(r["model"], r["dimension"])]
        cell[0] += int(r["correct"])
        cell[1] += int(r["n"])

    for (model, dimension), (correct, n) in sorted(totals.items()):
        if model not in scored:
            out.append(f"{model} is in the table but has no primary-tier predictions")
            continue
        p_correct, p_n = scored[model]
        if (correct, n) != (p_correct, p_n):
            out.append(
                f"{model} / {dimension}: table totals {correct}/{n}, "
                f"predictions give {p_correct}/{p_n}"
            )
            continue
        if n and abs(100 * correct / n - 100 * p_correct / p_n) > tol:
            out.append(f"{model} / {dimension}: accuracies disagree")
    return out


def denominator_gaps(
    predictions: Sequence[Mapping[str, str]], subgroups: Sequence[Mapping[str, str]]
) -> dict[str, dict[str, int]]:
    """Per model: how many items the predictions file scored, against the
    denominator the published table used.

    Returned rather than asserted, because the two can legitimately differ: the
    table may apply an exclusion the prediction file does not record. What this
    makes impossible is for that difference to go unnoticed, or to change
    silently. The committed baseline in tests/baselines/release_denominators.json
    holds the current gaps, and the test fails when one moves.
    """
    scored = scored_by_model(predictions)
    sums: defaultdict[tuple[str, str], int] = defaultdict(int)
    for r in subgroups:
        sums[(r["model"], r["dimension"])] += int(r["n"])
    table: dict[str, int] = {}
    for (model, _dim), total in sums.items():
        table[model] = max(table.get(model, 0), total)

    out = {}
    for model, published in sorted(table.items()):
        out[model] = {
            "scored_in_predictions": scored.get(model, 0),
            "denominator_in_table": published,
            "gap": scored.get(model, 0) - published,
        }
    return out


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

CHECKS = (
    ("prediction grid is complete", lambda p, s: check_prediction_grid(p)),
    ("one answer key per item", lambda p, s: check_one_key_per_item(p)),
    ("is_correct agrees with its inputs", lambda p, s: check_is_correct_consistency(p)),
    ("each model has one tier", lambda p, s: check_tiers(p)),
    ("accuracy_pct equals correct/n", lambda p, s: check_accuracy_matches_counts(s)),
    ("published intervals are Wilson intervals", lambda p, s: check_wilson_intervals(s)),
    ("dimensions agree on the item count", lambda p, s: check_subgroup_partitions(s)),
    ("unparsed answers use one encoding", lambda p, s: check_unparsed_encoding(p)),
    ("predictions and the table agree", check_leaderboard_matches_the_table),
)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="exit 1 on any finding")
    args = ap.parse_args(argv)

    for path in (PREDICTIONS, SUBGROUPS):
        if not path.exists():
            print(f"missing release artifact: {path}", file=sys.stderr)
            return 2

    predictions = read_csv(PREDICTIONS)
    subgroups = read_csv(SUBGROUPS)
    print(f"{len(predictions)} prediction rows, {len(subgroups)} subgroup rows\n")

    total = 0
    for label, fn in CHECKS:
        findings = fn(predictions, subgroups)
        total += len(findings)
        print(f"{'FAIL' if findings else ' ok '}  {label}")
        for f in findings:
            print(f"        {f}")

    print("\nreported, not asserted:")
    unparsed = unparsed_by_model(predictions)
    if unparsed:
        worst = sorted(unparsed.items(), key=lambda kv: -kv[1])[:5]
        print(f"  items with no parseable answer: {sum(unparsed.values())} across {len(unparsed)} models; worst {worst}")
    for model, d in denominator_gaps(predictions, subgroups).items():
        if d["gap"]:
            print(
                f"  {model}: predictions scored {d['scored_in_predictions']}, "
                f"table used {d['denominator_in_table']} (gap {d['gap']})"
            )

    print(f"\n{total} finding(s)")
    return 1 if (args.strict and total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
