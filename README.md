# Model Character Tracking Limit

This project evaluates how many distinct characters a modern LLM can track in long-form narratives with explicit state updates. It uses synthetic stories with controlled character counts and distractor sentences sampled from NarrativeQA and BookSum.

Key findings:
- GPT-4.1 reached 98.2% accuracy across N=2..32 with only two confusion errors.
- No measurable degradation with longer contexts in this setup.
- Baselines (random, recency) performed far below the model.

## How to Reproduce
1. Create and activate the environment:
   ```bash
   uv venv
   source .venv/bin/activate
   uv add openai pandas numpy scipy statsmodels matplotlib seaborn datasets tenacity
   ```
2. Prepare distractors:
   ```bash
   python src/data_prep.py
   ```
3. Run experiments (requires `OPENAI_API_KEY` or `OPENROUTER_API_KEY`):
   ```bash
   python src/run_experiment.py
   ```
4. Analyze results:
   ```bash
   python src/analyze_results.py
   ```

## File Structure
- `src/data_prep.py`: build distractor sentence pool
- `src/run_experiment.py`: generate narratives and query GPT-4.1
- `src/analyze_results.py`: compute metrics and plots
- `results/model_outputs.jsonl`: raw model outputs
- `results/metrics.json`: aggregated metrics
- `results/plots/`: figures
- `REPORT.md`: full report

See `REPORT.md` for full methodology and results.
