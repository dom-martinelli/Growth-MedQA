# Growth-MedQA

Reproducibility and supplementary materials for **Growth-MedQA**, an auditable benchmark of pediatric growth-assessment multiple-choice questions curated from public medical QA sources.

> Can an LLM ace the test and fail the patient? A single aggregate accuracy score can obscure clinically meaningful variation in how a model handles pediatric growth. Growth-MedQA resolves model performance to task- and concept-specific slices, audits the source items themselves, and probes a decision behavior that accuracy cannot see.

## Links
- Dataset (full items, keys, provenance, labels): https://huggingface.co/datasets/dmartinelli/growth-medqa
- Interactive explorer: https://huggingface.co/spaces/dmartinelli/growth-medqa-explorer
- Manuscript: see citation below (link added on publication)

## What is here
This repository holds the release artifacts and reproducibility files. The full item text lives in the Hugging Face dataset above.

```
data/
  growth_medqa_per_model_predictions.csv   Per-model prediction and correctness for all 960 items, every model
supplementary/
  subgroup_accuracy_wilson_ci.csv          Accuracy with 95% Wilson CIs and denominators, by subgroup
  example_items_by_topic_and_task.csv      Example items per ICPED growth topic and per USMLE competency
  S2_eligibility_criteria.csv              Benchmark eligibility criteria, scope, and rationale
  S3_representative_excluded.csv           Representative benchmarks excluded under each criterion
  supplementary_methods_S1.md              Label construction and validation (few-shot classifier, reference set)
  external_comparison_methods.md           External medical-benchmark comparison: sampling, hashing, correlation
figures/
  figure3_within_model_subject_spread.png  Within-model accuracy spread across growth subjects (10 models)
  figure4_kappa_heatmap.png                Pairwise Cohen's kappa across models (10 models)
  supp_figure_S1_labeling_framework.png    The three-axis item-labeling schematic
reproduce.py                               Standalone: recompute the leaderboard and subgroups from the predictions
```

## Models
The primary comparison uses ten models scored under a uniform single-completion protocol: Gemini 2.5 Pro, GPT-5.5, Claude Sonnet 4.6, Claude Sonnet 4.5, Claude Haiku 4.5, GPT-4o, GPT-4o mini, Llama 3.3 70B, Llama 3.1 8B, Mistral Small 3.2 24B. Three additional models run under reasoning-enabled or reduced-token configurations (Kimi K2 Thinking, GLM 4.7, Claude Opus 4.8) are included in the prediction file with `tier = excluded_reasoning_config` and are not part of the primary comparison.

## Reproduce
```
pip install pandas
python reproduce.py
```

## Labels
Each item carries a binary growth-relevance label (growth is part of the tested reasoning, or a distractor to be excluded), a growth-topic category anchored to the ESPE ICPED and ABP content outline with growth-relevance grounded in the Human Phenotype Ontology, and a physician-competency label against the USMLE Physician Tasks/Competencies vocabulary. Labels place an item in a slice and never determine its answer.

## Checks

The numbers in this repository are checked against themselves on every push
(`.github/workflows/tests.yml`). The checks need no data beyond what is
committed here and no libraries beyond the standard library, so they run in a
few seconds.

```
pip install -r requirements-test.txt
python -m pytest tests -q          # 64 cases
python -m qa.release_checks        # the same checks as a readable report
```

What is verified about the release artifacts:

- Every model answered every item exactly once (22 models x 960 items, no
  duplicates and no holes), and each item carries one answer key.
- `is_correct` agrees with `predicted_label` and `correct_answer_label` on all
  21,120 rows, rather than being trusted as a stored boolean.
- Each model belongs to exactly one tier, and the subgroup table covers the
  primary tier and only the primary tier.
- Every `accuracy_pct` equals its own `correct`/`n`.
- **Every published 95% interval is the Wilson interval for its own numerator
  and denominator**, recomputed from scratch across all 160 rows. An interval is
  the one number in a supplementary table a reader cannot sanity-check by eye,
  so a different method or a denominator that moved after the numerator was
  fixed would otherwise be invisible. The recomputation is itself pinned against
  hand-checkable cases first.
- `data/`, `supplementary/` and `reproduce.py` describe one set of results: each
  primary model's accuracy recomputed from the predictions equals the totals the
  subgroup table implies. If one artifact is regenerated and the others are not,
  this fails.
- `reproduce.py` is run in a clean environment by CI, because a reproducibility
  package whose one documented command is never executed is a promise rather
  than a package.

One open item, recorded rather than hidden (`tests` reports it as an expected
failure, so it stays visible while the decision is pending):

> When no answer can be parsed from a response, this file does two different
> things. 348 such rows leave `is_correct` blank, and 13 write `0.0`. Both rules
> are defensible, but using both in one file means a reader who averages
> `is_correct` directly gets a different number from the paper for three models
> (Claude Haiku 4.5, Gemini 2.5 Pro, Llama 3.3 70B), all primary tier. The
> published tables and `reproduce.py` both drop rows with no parsed answer, so
> those 13 values are never used and the published numbers are unaffected.

### Source-pool checks

`qa/dataset_checks.py` carries the audit of the upstream question banks the
items were curated from. The pools themselves are not redistributed here (see
the Hugging Face dataset), so these tests skip unless a working copy is
available:

```
GROWTH_MEDQA_POOLS=/path/to/pediatric_qa_outputs python -m pytest tests -q
```

The committed baselines in `tests/baselines/` record what each pool contained
when the curation ran, so a test fails when upstream data *changes* rather than
when it is merely imperfect. The largest finding: **1,451 of 7,457 rows in the
pediatric MedMCQA extract (19.5%) have the letters "rt" deleted inside words**
(heart to hea, birth to bih, artery to aery, important to impoant). It is in the
source rows, it defeats keyword curation because the corrupt spelling is
sometimes commoner than the correct one, and it appears in none of the other
pools. The detectors are tested against hand-written fixtures before being
pointed at the data, because "0 findings" reads the same whether the pool is
clean or the pattern is broken.

### A note on the MedXpertQA exclusion

Items excluded for having a key outside the A-E options are not an upstream
defect. Every MedXpertQA item in the pediatric extract offers ten options (A-J)
with its key always inside its own set, and 234 of 491 key to F-J. The exclusion
is produced by normalising ten options onto five.

## Data provenance and license
Items are derived from public medical QA benchmarks; redistribution of item content follows the licenses of the underlying sources. The derived labels, predictions, and supplementary tables in this repository are released under CC BY 4.0. See DATA_PROVENANCE.md.

## Citation
Martinelli D, Lee J. Can an LLM Ace the Test and Fail the Patient? A Critical Appraisal of Medical LLM Benchmarks. 2026.
