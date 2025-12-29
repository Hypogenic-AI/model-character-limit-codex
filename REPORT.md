# REPORT

## 1. Executive Summary
This study tests how many distinct characters a modern LLM can track when their attributes change inside a long narrative.
Across 110 synthetic narrative trials (N=2..32, short vs long context), GPT-4.1 achieved 98.2% accuracy with only two confusion errors, showing no clear degradation up to 32 characters.
Practically, this suggests that for controlled attribute-tracking tasks of this type, GPT-4.1’s capacity exceeds the tested range; future work should increase difficulty and diversity to locate the true failure point.

## 2. Goal
We tested whether LLMs exhibit a capacity limit in character tracking as the number of characters and context length increase. This matters for long-form narrative understanding and state-tracking tasks, where failures can be subtle and hard to debug. The expected impact was a measurable accuracy drop and an identifiable inflection point that could inform model memory mechanisms.

## 3. Data Construction

### Dataset Description
- Source: NarrativeQA (HELM subset) and BookSum (local snapshots).
- Use: Only as a pool of distractor sentences to make synthetic narratives more realistic.
- Size: 2,000 distractor sentences sampled from the datasets.
- Biases: Distractors reflect narrative domain distributions in the source data and may include uneven topic coverage.

### Example Samples
Synthetic example (truncated):

```
Ulric is a artist who carries a orb. Mu Bai is also a good friend of Yu Shu Lien...
Later, Harper traded the ring for a locket. ...
Question: At the end of the story, what object does Harper carry?
Answer: locket
```

### Data Quality
- Missing values: 0% (synthetic generation).
- Outliers: None detected; all instances generated under the same template.
- Validation checks: ensured unique object assignments and valid updates per character.

### Preprocessing Steps
1. Load NarrativeQA and BookSum from disk.
2. Split passages into sentences with a simple regex-based splitter.
3. Filter short sentences and sample 2,000 distractors.

### Train/Val/Test Splits
No training split was used; this is an evaluation-only experiment.

## 4. Experiment Description

### Methodology

#### High-Level Approach
Generate synthetic stories with N characters. Each character starts with an object and then undergoes multiple “trade” updates. The model must answer the final object for a target character after reading the full story.

#### Why This Method?
It isolates character tracking and state updates while keeping narrative-like distractors. Alternatives (pure coreference or real QA datasets) provide less control over ground-truth entity states.

### Implementation Details

#### Tools and Libraries
- openai: 2.14.0
- datasets: 4.4.2
- numpy: 2.4.0
- pandas: 2.3.3
- statsmodels: 0.14.6
- matplotlib: 3.10.8
- seaborn: 0.13.2

#### Algorithms/Models
- Model: GPT-4.1 (OpenRouter).
- Sampling: temperature=0, max_tokens=32.

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| temperature | 0.0 | fixed |
| max_tokens | 32 | fixed |
| trials per condition | 5 | budgeted |
| N values | 2..32 | design grid |

#### Training Procedure or Analysis Pipeline
1. Generate synthetic stories with initial assignments and update events.
2. Call GPT-4.1 for each story-question pair.
3. Score exact-match on the final object.
4. Aggregate accuracy and bootstrap CIs.
5. Fit logistic regression and segmented regression for inflection detection.

### Experimental Protocol

#### Reproducibility Information
- Runs per condition: 5
- Seed: 42
- Hardware: CPU-only
- Execution time: ~2 minutes for 110 calls

#### Evaluation Metrics
- Accuracy: whether the final object is correctly identified.
- Baselines: random guess and recency heuristic.
- Confidence intervals: bootstrap 95% CIs per condition.

### Raw Results

#### Tables
| N | Length | Accuracy | 95% CI | Count |
|---|--------|----------|--------|-------|
| 2 | long | 1.00 | [1.00, 1.00] | 5 |
| 2 | short | 1.00 | [1.00, 1.00] | 5 |
| 4 | long | 1.00 | [1.00, 1.00] | 5 |
| 4 | short | 1.00 | [1.00, 1.00] | 5 |
| 6 | long | 1.00 | [1.00, 1.00] | 5 |
| 6 | short | 1.00 | [1.00, 1.00] | 5 |
| 8 | long | 0.80 | [0.40, 1.00] | 5 |
| 8 | short | 0.80 | [0.40, 1.00] | 5 |
| 10 | long | 1.00 | [1.00, 1.00] | 5 |
| 10 | short | 1.00 | [1.00, 1.00] | 5 |
| 12 | long | 1.00 | [1.00, 1.00] | 5 |
| 12 | short | 1.00 | [1.00, 1.00] | 5 |
| 16 | long | 1.00 | [1.00, 1.00] | 5 |
| 16 | short | 1.00 | [1.00, 1.00] | 5 |
| 20 | long | 1.00 | [1.00, 1.00] | 5 |
| 20 | short | 1.00 | [1.00, 1.00] | 5 |
| 24 | long | 1.00 | [1.00, 1.00] | 5 |
| 24 | short | 1.00 | [1.00, 1.00] | 5 |
| 28 | long | 1.00 | [1.00, 1.00] | 5 |
| 28 | short | 1.00 | [1.00, 1.00] | 5 |
| 32 | long | 1.00 | [1.00, 1.00] | 5 |
| 32 | short | 1.00 | [1.00, 1.00] | 5 |

#### Visualizations
- Accuracy vs N: `results/plots/accuracy_by_n.png`
- Error types: `results/plots/error_types.png`

#### Output Locations
- Results JSON: `results/metrics.json`
- Plots: `results/plots/`
- Raw outputs: `results/model_outputs.jsonl`

## 5. Result Analysis

### Key Findings
1. GPT-4.1 achieved 98.2% accuracy overall and 100% accuracy for most N values.
2. Only two errors occurred (both confusion errors), yielding no consistent trend with increasing N.
3. Length condition (short vs long) did not change accuracy.

### Hypothesis Testing Results
- Hypothesis: performance should degrade as N grows.
- Result: not supported in this range. Logistic regression showed no significant effect of N or length (p > 0.35). The segmented fit selected a breakpoint at N=8, but this was driven by only two errors and is not robust.

### Comparison to Baselines
- Random guess: 0.9% accuracy.
- Recency heuristic: 6.4% accuracy.
- GPT-4.1 significantly outperforms both baselines.

### Visualizations
- `results/plots/accuracy_by_n.png` shows flat accuracy across N.
- `results/plots/error_types.png` shows confusion as the only error category.

### Surprises and Insights
- The model remained near-ceiling even with updates and long distractor chains, suggesting the task may be too easy for GPT-4.1.

### Error Analysis
- Two failures were confusion errors where the model output another valid object from the story.

### Limitations
- Small sample size per condition (n=5) yields wide CIs at the single N where errors occurred.
- Only one model evaluated.
- Synthetic stories may not capture the complexity of real long-form narratives.

## 6. Conclusions
GPT-4.1 successfully tracked up to 32 characters with multiple state updates and long distractor text, showing no clear degradation. This does not support the hypothesis of a low character-tracking limit under these controlled conditions.

### Implications
- Practical: GPT-4.1 can handle moderate-to-large cast tracking for simple attribute queries.
- Theoretical: the task design needs higher difficulty (e.g., multiple attributes, adversarial distractors) to reveal a capacity limit.

### Confidence in Findings
Moderate. The experiment is reproducible but limited in scope and sample size.

## 7. Next Steps

### Immediate Follow-ups
1. Increase trials per condition to 30+ to estimate a reliable accuracy curve.
2. Introduce multiple attributes per character and cross-attribute questions.

### Alternative Approaches
- Use a coreference-heavy narrative dataset with real human annotations for entity states.
- Add paraphrased questions and indirect references (pronouns, epithets).

### Broader Extensions
- Compare multiple model families and sizes.
- Test long-context models with 100+ characters.

### Open Questions
- At what scale does entity tracking fail for real-world narratives?
- Do models rely on heuristics (recency, salience) or stable entity representations?

## References
- papers/2305.02363_entity_tracking_in_language_models.pdf
- papers/2202.01709_consistent_entities_in_narrative_generation.pdf
- papers/2010.02807_long_doc_coreference_bounded_memory.pdf
- papers/2503.02854_how_do_lms_track_state.pdf
- datasets/narrative_qa_helm/
- datasets/booksum/
