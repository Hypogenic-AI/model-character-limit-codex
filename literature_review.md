# Literature Review

## Research Area Overview
Entity tracking in long-form narratives sits at the intersection of coreference resolution, discourse modeling, and long-context language modeling. Recent work probes whether LMs can track entity states through sequences of updates, evaluates coherence and consistency of entity mentions in generated stories, and introduces benchmarks for entity tracking (including multimodal settings). Related work on coreference in long documents addresses memory constraints and provides mechanisms for maintaining entity representations over long contexts.

## Key Papers

### Paper 1: Entity Tracking in Language Models
- **Authors**: Najoung Kim; Sebastian Schuster
- **Year**: 2023
- **Source**: arXiv
- **Key Contribution**: Introduces a probing task to evaluate entity state tracking in LMs.
- **Methodology**: Task requires inferring final entity states given initial states and state-changing operations.
- **Datasets Used**: Task data described in the paper (not specified in abstract).
- **Results**: Provides systematic evaluation of LMs on entity state tracking.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Directly measures the model's ability to track entity attributes.

### Paper 2: Towards Coherent and Consistent Use of Entities in Narrative Generation
- **Authors**: Pinelopi Papalampidi; Kris Cao; Tomas Kocisky
- **Year**: 2022
- **Source**: arXiv
- **Key Contribution**: Proposes automatic metrics for entity coherence/consistency in story generation.
- **Methodology**: Measures entity usage patterns in generated narratives using new metrics.
- **Datasets Used**: Narrative generation datasets (not specified in abstract).
- **Results**: Shows gaps in entity coherence and consistency for large LMs.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Provides evaluation metrics for character consistency in long text.

### Paper 3: Learning to Ignore: Long Document Coreference with Bounded Memory Neural Networks
- **Authors**: Shubham Toshniwal; Sam Wiseman; Allyson Ettinger; Karen Livescu; Kevin Gimpel
- **Year**: 2020
- **Source**: arXiv
- **Key Contribution**: Bounded-memory neural coreference that does not keep all entities in memory.
- **Methodology**: Memory-augmented model that tracks a small number of entities at a time.
- **Datasets Used**: Long-document coreference datasets (not specified in abstract).
- **Results**: Demonstrates practical benefits for long documents with constrained memory.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Addresses the memory bottleneck for tracking many entities.

### Paper 4: CoreLM: Coreference-aware Language Model Fine-Tuning
- **Authors**: Nikolaos Stylianou; Ioannis Vlahavas
- **Year**: 2021
- **Source**: arXiv
- **Key Contribution**: Adds coreference awareness during LM fine-tuning to handle long texts.
- **Methodology**: Coreference-informed fine-tuning to improve long-context understanding.
- **Datasets Used**: Not specified in abstract.
- **Results**: Claims improvements in long-text processing with coreference signals.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Suggests a modeling direction to improve entity tracking.

### Paper 5: (How) Do Language Models Track State?
- **Authors**: Belinda Z. Li; Zifan Carl Guo; Jacob Andreas
- **Year**: 2025
- **Source**: arXiv
- **Key Contribution**: Studies state tracking in LMs via permutation tracking tasks.
- **Methodology**: LMs track state changes from sequences of swap operations.
- **Datasets Used**: Synthetic state tracking tasks (not specified in abstract).
- **Results**: Analyzes how LMs represent and update latent state.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Provides insight into state tracking mechanisms.

### Paper 6: MET-Bench: Multimodal Entity Tracking for Evaluating the Limitations of Vision-Language and Reasoning Models
- **Authors**: Vanya Cohen; Raymond Mooney
- **Year**: 2025
- **Source**: arXiv
- **Key Contribution**: Introduces a multimodal entity tracking benchmark.
- **Methodology**: Uses structured domains (Chess, Shell Game) to test multimodal entity state tracking.
- **Datasets Used**: MET-Bench benchmark (two structured domains).
- **Results**: Evaluates VLMs on cross-modal entity tracking.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Offers controlled entity-tracking evaluations with state changes.

### Paper 7: Modeling Human Mental States with an Entity-based Narrative Graph
- **Authors**: I-Ta Lee; Maria Leonor Pacheco; Dan Goldwasser
- **Year**: 2021
- **Source**: arXiv
- **Key Contribution**: Introduces an Entity-based Narrative Graph (ENG) for character mental states.
- **Methodology**: Explicit entity modeling with task-adaptive pre-training and inference.
- **Datasets Used**: Not specified in abstract.
- **Results**: Improves modeling of character goals and internal states.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Adds a character-centric representation for narrative tracking.

## Common Methodologies
- Entity-state tracking tasks: Used in Entity Tracking in Language Models; (How) Do Language Models Track State?
- Coreference-aware modeling: Used in Learning to Ignore; CoreLM.
- Narrative entity coherence metrics: Used in Towards Coherent and Consistent Use of Entities in Narrative Generation.
- Graph-based entity modeling: Used in Entity-based Narrative Graph.

## Standard Baselines
- Coreference resolution models (span-based or e2e): Common baseline for entity tracking in long documents.
- Vanilla transformer LMs: Baseline for tracking state changes or narrative consistency.

## Evaluation Metrics
- Entity coherence/consistency metrics for narrative generation.
- Accuracy on state-tracking tasks (final state prediction).
- Coreference resolution metrics (e.g., MUC, B^3, CEAF) for entity linking in long texts.

## Datasets in the Literature
- MET-Bench (structured domains like Chess, Shell Game) for multimodal entity tracking.
- Long-document coreference datasets (not specified in abstracts).
- Narrative generation datasets for entity coherence evaluation (not specified in abstracts).

## Gaps and Opportunities
- Limited standardized benchmarks focused on character-level tracking in long narratives.
- Need for controlled datasets with many entities and long context to probe capacity limits.

## Recommendations for Our Experiment
- **Recommended datasets**: NarrativeQA (HELM subset) and BookSum for long-form narrative passages; consider MET-Bench if multimodal evaluation is desired.
- **Recommended baselines**: Strong coreference models (e2e-coref, SpanBERT-based) plus vanilla LMs for comparison.
- **Recommended metrics**: State-tracking accuracy; entity consistency metrics; coreference F1 (MUC/B^3/CEAF).
- **Methodological considerations**: Use controlled perturbations (entity swaps, attribute updates) to stress entity tracking limits as context length increases.
