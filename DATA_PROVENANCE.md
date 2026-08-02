# Data provenance

The Growth-MedQA items were curated from publicly available medical question-answering benchmarks. Item text is redistributed through the Hugging Face dataset under the licenses of the underlying sources. This repository redistributes:

- Derived per-item labels (growth relevance as a binary include/exclude, growth topic, physician competency)
- Per-model predictions and correctness
- Aggregate and subgroup accuracy with confidence intervals
- A small number of truncated example stems for illustration

No patient-identifying information is present. The behavioral referral probe referenced in the manuscript uses synthetic short-but-healthy child profiles.

Derived labels, predictions, and tables: CC BY 4.0.
Underlying item content: governed by each source benchmark's own license.
