# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

### Papers
Total papers downloaded: 7

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Entity Tracking in Language Models | Najoung Kim; Sebastian Schuster | 2023 | papers/2305.02363_entity_tracking_in_language_models.pdf | Entity state tracking probe for LMs |
| Towards Coherent and Consistent Use of Entities in Narrative Generation | Pinelopi Papalampidi; Kris Cao; Tomas Kocisky | 2022 | papers/2202.01709_consistent_entities_in_narrative_generation.pdf | Entity coherence metrics for stories |
| Learning to Ignore: Long Document Coreference with Bounded Memory Neural Networks | Shubham Toshniwal; Sam Wiseman; Allyson Ettinger; Karen Livescu; Kevin Gimpel | 2020 | papers/2010.02807_long_doc_coreference_bounded_memory.pdf | Bounded-memory coreference for long docs |
| CoreLM: Coreference-aware Language Model Fine-Tuning | Nikolaos Stylianou; Ioannis Vlahavas | 2021 | papers/2111.02687_corelm_coreference_aware_lm_finetuning.pdf | Coreference-aware LM fine-tuning |
| (How) Do Language Models Track State? | Belinda Z. Li; Zifan Carl Guo; Jacob Andreas | 2025 | papers/2503.02854_how_do_lms_track_state.pdf | State tracking via permutation tasks |
| MET-Bench: Multimodal Entity Tracking for Evaluating the Limitations of Vision-Language and Reasoning Models | Vanya Cohen; Raymond Mooney | 2025 | papers/2502.10886_met_bench_multimodal_entity_tracking.pdf | Multimodal entity tracking benchmark |
| Modeling Human Mental States with an Entity-based Narrative Graph | I-Ta Lee; Maria Leonor Pacheco; Dan Goldwasser | 2021 | papers/2104.07079_entity_based_narrative_graphs.pdf | Entity-based narrative graphs for character states |

See `papers/README.md` for detailed descriptions.

### Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| NarrativeQA (HELM subset) | HuggingFace `lighteval/narrative_qa_helm` | ~3.6 MB | Narrative QA | datasets/narrative_qa_helm/ | Small subset; long passages |
| BookSum | HuggingFace `kmfoda/booksum` | ~227 MB | Long-form summarization | datasets/booksum/ | Long chapters with summaries |

See `datasets/README.md` for detailed descriptions.

### Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| coref-spanbert | https://github.com/mandarjoshi90/coref | SpanBERT coreference baseline | code/coref-spanbert/ | TensorFlow-based coreference |
| e2e-coref | https://github.com/kentonl/e2e-coref | End-to-end coreference baseline | code/e2e-coref/ | Classic baseline for coref |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used arXiv API to find papers on entity tracking, state tracking, narrative entity consistency, and coreference in long documents.
- Selected papers with direct relevance to entity state tracking, narrative coherence, and long-context entity representation.
- Used HuggingFace datasets with built-in data formats compatible with the installed `datasets` library.
- Selected coreference repositories as strong baselines for entity tracking in long text.

### Selection Criteria
- Direct relevance to tracking entities/characters and their attributes over long contexts.
- Availability of open-access PDFs on arXiv.
- Benchmarks or methods that can be adapted to character tracking capacity experiments.

### Challenges Encountered
- Some HuggingFace datasets use legacy dataset scripts that are unsupported by the installed `datasets` version.
- Limited code links found directly from abstracts; selected strong baseline repos instead.

### Gaps and Workarounds
- Narrative-focused coreference datasets were not directly available via this `datasets` version; used NarrativeQA and BookSum as long-context surrogates.
- Specific paper code repositories were not readily discoverable; substituted with widely-used coreference baselines.

## Experiments Run (This Project)
- Synthetic narrative entity-tracking benchmark with controlled character counts (N=2..32), update events, and distractor sentences.
- Model evaluated: GPT-4.1 via OpenRouter (temperature=0).
- Outputs saved to `results/model_outputs.jsonl`, summaries to `results/summary.csv`, plots to `results/plots/`.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: NarrativeQA (HELM subset) for QA over long passages; BookSum for long narrative chapters.
2. **Baseline methods**: SpanBERT-based coreference (coref-spanbert) and e2e-coref for entity resolution baselines.
3. **Evaluation metrics**: Entity state accuracy; coreference F1 (MUC/B^3/CEAF); entity consistency metrics from narrative generation work.
4. **Code to adapt/reuse**: `code/coref-spanbert/` for strong coref baseline; `code/e2e-coref/` for a lightweight baseline.
