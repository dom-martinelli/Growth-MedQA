# Fixtures

Three tiny synthetic CSVs. They exist so the checks in `qa/dataset_checks.py`
are themselves tested, which matters more here than it sounds.

A data-quality check is a detector, and a detector has two failure modes. It can
miss a defect that is present, and it can report one that is not. A check run
only against the real pool can be wrong in either direction and look fine:
"0 findings" reads the same whether the data is clean or the regex is broken.

So each check gets both controls:

| file | role | what the check must do |
|---|---|---|
| `clean_items.csv` | negative control | report nothing |
| `rt_corrupt_items.csv` | positive control | find every planted deletion, and no others |
| `bad_key_items.csv` | positive control | find every planted unscoreable item |

The real pools are gitignored (they are hundreds of megabytes and some are
redistributions), so the tests that read them skip when they are absent and the
fixture tests always run. That way CI proves the detector works on every push,
and a local run with the data present proves what the pool contains.

These rows are written by hand, not sampled from any dataset. The stems are
nonsense on purpose.
