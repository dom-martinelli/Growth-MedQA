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

## Data provenance and license
Items are derived from public medical QA benchmarks; redistribution of item content follows the licenses of the underlying sources. The derived labels, predictions, and supplementary tables in this repository are released under CC BY 4.0. See DATA_PROVENANCE.md.

## Citation
Martinelli D, Lee J. Can an LLM Ace the Test and Fail the Patient? A Critical Appraisal of Medical LLM Benchmarks. 2026.
