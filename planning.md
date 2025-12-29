# Research Plan

## Research Question
How many distinct characters and attributes can a modern LLM reliably track in long-form narratives, and where does performance degrade as character count increases?

## Background and Motivation
Entity tracking in narratives relates to coreference and state tracking, but there is limited controlled evaluation of how many distinct characters and attributes an LLM can track as stories grow longer. Understanding the failure point can shed light on the model's implicit memory mechanisms and practical limits for story QA and long-context tasks.

## Hypothesis Decomposition
1. As the number of characters (N) in a narrative increases, accuracy on character-attribute questions declines.
2. The drop in accuracy is nonlinear, with a noticeable inflection point indicating a capacity limit.
3. Longer contexts with distractor mentions further reduce accuracy at the same N.
4. Models differ in capacity, with larger models sustaining higher N before degradation.

## Proposed Methodology

### Approach
Create a controlled narrative generation and evaluation harness. For each N, generate stories with N unique characters, each with a fixed attribute (e.g., role, object, color). Insert distractor sentences to increase length. Query the model with direct questions about character attributes. Measure accuracy as a function of N and context length. Use at least one real LLM API (e.g., GPT-4.1 via OpenRouter) and optionally a second model for comparison.

### Experimental Steps
1. Load available narrative datasets (NarrativeQA and BookSum) for realistic sentence templates and entity name distributions. Rationale: ensures narrative-like phrasing and reduces artificiality.
2. Build a synthetic narrative generator that inserts N characters with attributes and interleaves distractors sampled from dataset passages. Rationale: controlled ground truth while preserving realistic text.
3. Construct QA prompts with a fixed template and controlled decoding parameters (temperature=0). Rationale: reduce randomness in evaluation.
4. Run experiments across N in {2,4,6,8,10,12,16} and two length conditions (short, long). Rationale: cover a range to locate the inflection point.
5. Collect model outputs and score exact-match (case-insensitive) for the target attribute.
6. Analyze accuracy vs N and context length, estimate the inflection point using segmented regression or logistic curve fitting.
7. Perform error analysis by sampling failures and categorizing error types (confusion, omission, hallucination).

### Baselines
- Random guess baseline based on attribute set size.
- Heuristic baseline that answers with the most recent mentioned attribute (recency).
- Optional comparison across two models (e.g., GPT-4.1 vs Claude Sonnet 4.5).

### Evaluation Metrics
- Accuracy by N and length condition.
- Confidence intervals via bootstrap.
- Inflection point estimate for accuracy drop (segmented regression).
- Error type distribution.

### Statistical Analysis Plan
- Use logistic regression or segmented regression to model accuracy vs N.
- Compare short vs long using two-proportion z-tests per N with Holm correction.
- Report effect sizes (difference in proportions) and 95% CIs.
- Significance level alpha=0.05.

## Expected Outcomes
- Support: Accuracy declines with larger N, with an identifiable drop around a specific N; longer contexts worsen performance.
- Refute: Accuracy remains flat across N or no consistent degradation with N or length.

## Timeline and Milestones
- Phase 1 (planning): 30 min
- Phase 2 (setup and data prep): 30 min
- Phase 3 (implementation): 60 min
- Phase 4 (experiments): 60-90 min
- Phase 5 (analysis): 45 min
- Phase 6 (documentation): 30 min

## Potential Challenges
- API rate limits or costs: mitigate by small pilot runs and caching outputs.
- Dataset parsing complexity: use minimal sampling, fall back to simple distractor templates if needed.
- Non-determinism: use temperature=0 and fixed seeds.

## Success Criteria
- Completed runs across all N and length conditions.
- Statistical analysis with confidence intervals and inflection point estimate.
- REPORT.md with actual results and error analysis examples.
