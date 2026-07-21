# Off-the-Shelf Large Language Models Are Unreliable Judges

Replication code, source data, and generated results for Jonathan H. Choi,
"Off-the-Shelf Large Language Models Are Unreliable Judges."

- [Preregistration](https://osf.io/25gya/)

The manuscript, online appendix, and presentation sources are maintained separately and are not included in this public replication repository.

## What this repository replicates

The paper studies four related questions:

1. How much legal judgments change when an otherwise identical prompt is
   rephrased.
2. Whether alternative output-processing methods and model families produce
   consistent judgments.
3. How closely off-the-shelf LLM judgments track ordinary-reader judgments
   from a preregistered survey.
4. Whether assistant tuning shifts ordinary-meaning judgments relative to
   matched base checkpoints.

The updated replication package also reports within-question human-response
dispersion, binary cross entropy (log loss), a GPT-4.1 top-logprob truncation
audit, and irrelevant-information robustness checks.

## Repository layout

```text
analysis/          Prompt perturbation, model evaluation, and robustness code
survey_analysis/   Survey cleaning and human/model comparison code
data/              Raw survey exports and checked-in model outputs
results/           Generated figures, tables, CSV files, and metadata
requirements.txt   Python dependencies
```

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/jonathanhchoi/llm-interpretation-replication.git
cd llm-interpretation-replication
python -m pip install -r requirements.txt
```

The main survey and checked-in model-output analyses run offline. API keys are
needed only to collect fresh model responses. To enable API-backed runs, copy
`.env.example` to `.env` and set the applicable values:

```text
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
HUGGINGFACE_TOKEN=...
```

API-backed runs can incur charges, and provider-side model behavior may change
after the checked-in results were collected.

## Reproduce the added analyses offline

Run these commands from the repository root:

```bash
# Rebuild the closed-source-model metrics, tables, and figures from checked-in
# model outputs and the cleaned human survey sample.
python analysis/evaluate_closed_source_models.py

# Rebuild the human-response distribution figure and per-question table.
python survey_analysis/visualize_ground_truth_distribution.py

# Rebuild the matched base-versus-tuned 49-question analysis.
python survey_analysis/analyze_base_vs_tuned_49q.py

# Print the statistics from the checked-in 500-prompt GPT-4.1 top-k audit.
python analysis/audit_gpt_confidence_topk_bound.py --summary-only
```

The ordinary-meaning evaluator uses these exact model snapshots:

- OpenAI `gpt-4.1-2025-04-14`
- Anthropic `claude-opus-4-1-20250805`
- Google `gemini-2.5-pro`

The default evaluator is offline: it validates and consumes
`results/closed_source_evaluation/closed_source_evaluation_results.csv`.
To query only missing rows, use:

```bash
python analysis/evaluate_closed_source_models.py --fill-missing
```

To replace all model responses with a fresh API-backed run, use:

```bash
python analysis/evaluate_closed_source_models.py --refresh-model-results
```

## Survey sample and exclusions

The two survey exports contain the ten randomized forms used for the 100
substantive ordinary-meaning questions. The canonical cleaning helper is
`survey_analysis/survey_analysis_consolidated.py::load_cleaned_question_responses`.
It implements the preregistered exclusions:

- answer other than 100 on the form's attention check;
- completion time below 20% of the form median; or
- identical responses to all substantive questions on the form.

Of 1,008 recruited participants, 124 are excluded (115 attention-check
failures, 9 straight-line responses, and 0 duration failures), leaving
`N = 884`. Attention-check items are never treated as substantive questions.
The resulting battery contains exactly 100 questions, with about 88 responses
per question.

Passing `require_consent=True` to the shared helper applies an additional
consent-record sensitivity filter; it is not the paper's preregistered
`N = 884` specification.

## Principal generated outputs

| Analysis | Script | Main outputs |
|---|---|---|
| Closed-source models vs. humans | `analysis/evaluate_closed_source_models.py` | `results/closed_source_evaluation/human_comparisons.json`, `mae_results_tables.tex`, `per_question_errors.png` |
| Human-response dispersion | `survey_analysis/visualize_ground_truth_distribution.py` | `survey_analysis/human_response_distribution.csv`, `human_response_distribution_table.tex`, `ground_truth_distribution_simple.png` |
| Base vs. tuned checkpoints | `survey_analysis/analyze_base_vs_tuned_49q.py` | `results/base_vs_tuned_49q.json`, `base_vs_tuned_49q_table.tex` |
| GPT-4.1 top-k audit | `analysis/audit_gpt_confidence_topk_bound.py` | `results/gpt41_confidence_topk_audit.csv` |
| Three-model prompt sensitivity | `analysis/run_three_model_analysis.py` | `results/combined_analysis/` and three-model figures |
| Irrelevant-information robustness | `analysis/evaluate_irrelevant_perturbations.py` | `results/irrelevant_perturbations/` |

All bootstrap confidence intervals in the added ordinary-meaning analyses use
10,000 question-level resamples with deterministic seeds. Percentile ranges
over prompt outputs are labeled descriptive output intervals, not confidence
intervals.

## GPT-4.1 confidence top-k audit

`analysis/perturb_prompts.py` computes weighted numerical confidence from
returned confidence-token log probabilities and records the probability mass
covered by valid 0-100 tokens. If the API returns only the top `k` tokens, a
conservative absolute-error bound on the 0-100 confidence scale is:

```text
100 * (1 - returned valid-confidence probability mass)
```

The checked-in random audit contains 500 prompts. Its median bound is
0.000168 points and its 95th percentile is 0.121 points. The audit script is
resumable; the following command queries up to 500 previously unaudited prompts
and requires `OPENAI_API_KEY`:

```bash
python analysis/audit_gpt_confidence_topk_bound.py --sample --seed 42 --limit 500
```

## Other replication entry points

Prompt-sensitivity collection and analysis:

```bash
python analysis/perturb_prompts.py
python analysis/perturb_prompts_gpt.py
python analysis/perturb_prompts_claude_batch.py
python analysis/perturb_prompts_gemini_batch.py
python analysis/run_three_model_analysis.py
```

Irrelevant-information robustness:

```bash
python analysis/perturb_with_irrelevant_statements.py
python analysis/evaluate_irrelevant_perturbations.py
```

These collection pipelines use external APIs and can be substantially more
expensive than the offline commands above. Existing results are checked in so
that readers can inspect the reported analyses without rerunning collection.

## Validation

The repository includes offline invariant checks for the cleaned sample,
question battery, checked-in model outputs, and generated metadata:

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@unpublished{choi_off_the_shelf_llms,
  title  = {Off-the-Shelf Large Language Models Are Unreliable Judges},
  author = {Choi, Jonathan H.},
  note   = {Manuscript}
}
```

## License

This repository is released under the [MIT License](LICENSE).
