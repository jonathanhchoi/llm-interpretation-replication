# Off-the-Shelf Large Language Models Are Unreliable Judges

Replication code and data for "Off-the-Shelf Large Language Models Are Unreliable Judges" by Jonathan H. Choi.

**Paper:** Available at [SSRN link]
**Preregistration:** [OSF](https://osf.io/25gya/)

## Overview

This repository contains code to replicate the empirical analyses in the paper, which tests the reliability of large language models (LLMs) as legal interpreters through:

1. **Prompt Sensitivity Analysis** - Testing how LLM judgments vary with prompt phrasing
2. **Ordinary Meaning Survey** - Comparing LLM judgments to human survey responses
3. **Post-Training Effects** - Analyzing how instruction-tuning affects model alignment with human judgments

## Repository Structure

```
llm_interpretation_replication/
├── analysis/                    # Core analysis scripts
├── survey_analysis/             # Human survey analysis scripts
├── data/                        # Input data files
├── results/                     # Generated outputs (figures, tables)
├── main.tex                     # Main paper LaTeX source
├── main_online_appendix.tex     # Online appendix LaTeX source
└── requirements.txt             # Python dependencies
```

## Replication Guide by Paper Section

### Section 3: LLM Judgments Are Sensitive to Prompts

This section demonstrates that LLM legal interpretations are highly sensitive to prompt phrasing, output processing methods, and model choice.

#### Section 3.3-3.5: Prompt Perturbation Analysis (GPT-4.1)

**Scripts:**
- `analysis/perturb_prompts.py` - Generate prompt perturbations and collect GPT responses
- `analysis/analyze_perturbation_results.py` - Analyze relative probabilities and verbalized confidence

**Outputs:**
- Figure 1 (`combined_prompts_visualization.png`) - Distribution of relative probabilities
- Figure 2 (`combined_confidence_visualization.png`) - Distribution of verbalized confidence scores
- Table 1 & 2 in `main.tex`

#### Section 3.6 & Appendix B: Robustness to Alternative Models

Tests GPT-5, Claude Opus 4.1, and Gemini 2.5 Pro on the same perturbations.

**Scripts:**
- `analysis/perturb_prompts_gpt.py` - GPT model perturbation analysis
- `analysis/perturb_prompts_claude.py` - Claude model analysis
- `analysis/perturb_prompts_claude_batch.py` - Claude batch processing
- `analysis/perturb_prompts_gemini.py` - Gemini model analysis
- `analysis/perturb_prompts_gemini_batch.py` - Gemini batch processing
- `analysis/run_three_model_analysis.py` - Combined three-model analysis
- `analysis/create_three_model_stacked_visualization.py` - Generate comparison figures

**Outputs:**
- Figure 5 (`three_model_stacked_visualization.png`) - Three-model comparison
- Table in Appendix B (`main.tex` lines 539-574)

#### Section 3.6 & Appendix C: Irrelevant Information Perturbations

Tests whether inserting irrelevant factual statements affects model judgments.

**Scripts:**
- `analysis/perturb_with_irrelevant_statements.py` - Generate irrelevant statement perturbations
- `analysis/evaluate_irrelevant_perturbations.py` - Evaluate model responses to irrelevant info

**Data:**
- `data/irrelevant_statements.txt` - List of 200 irrelevant factual statements
- `data/perturbations_irrelevant.json` - Generated perturbations

**Outputs:**
- Figure 6 (`irrelevant_info_three_model_stacked_visualization.png`)
- Table in Appendix C

#### Appendix D: Normality Testing

**Scripts:**
- `analysis/analyze_perturbation_results.py` - Includes KS/AD tests and QQ plot generation

**Outputs:**
- Figures in Appendix D (`prompt_*_qq_plot.png`, `prompt_*_truncated_model.png`)

---

### Section 4: LLM Judgments Do Not Accurately Reflect Ordinary Meaning

This section compares LLM judgments on ordinary meaning questions to human survey responses.

#### Section 4.1-4.2: Survey Analysis and MAE Comparison

**Scripts:**
- `survey_analysis/survey_analysis_consolidated.py` - Main survey data processing
- `analysis/evaluate_closed_source_models.py` - Evaluate GPT, Claude, Gemini on survey questions
- `analysis/create_combined_visualization.py` - Generate error distribution figures

**Data:**
- `data/word_meaning_survey_results.csv` - Human survey responses (Part 1)
- `data/word_meaning_survey_results_part_2.csv` - Human survey responses (Part 2)
- `data/demographic_data.csv` - Survey participant demographics

**Outputs:**
- Figure 3 (`per_question_errors.png`) - Distribution of LLM errors vs human baseline
- Tables 3-4 in `main.tex` (MAE comparisons)

#### Section 4.3: Post-Training Effects on Ordinary Meaning

Compares base models vs instruction-tuned versions (Falcon, StableLM, RedPajama).

**Scripts:**
- `analysis/analyze_results_base_versus_instruct.py` - Base vs instruct model comparison
- `analysis/compare_base_vs_instruct.py` - Statistical analysis
- `analysis/run_base_vs_instruct_100q.py` - Run comparison on 100 questions
- `survey_analysis/analyze_base_vs_instruct_mae_100q.py` - MAE analysis

**Outputs:**
- Table 5 (`main.tex` lines 432-446) - MAE comparison base vs post-trained
- Figure 7 (`prompt_rel_prob_differences.png`) - Probability differences
- Figure 8 (`prompt_rel_prob_heatmap.png`) - Heatmap of changes

---

### Online Appendix

#### Section 1: Prompt Perturbation Examples

The detailed perturbation tables in the Online Appendix are generated from:
- `data/perturbations.json` - All 10,000 prompt perturbations

#### Section 2: Inter-Model Sensitivity (8 Open-Source Models)

**Scripts:**
- `analysis/compare_instruct_models.py` - Compare multiple open-source models
- `analysis/compare_instruct_models_survey2.py` - Extended model comparison
- `analysis/model_comparison_graph.py` - Generate correlation matrices

**Outputs:**
- Correlation matrices and distribution plots in Online Appendix Section 2

#### Section 5: Mixture-of-Experts and Speculative Decoding

Analysis of how architectural choices affect consistency (discussed in Online Appendix).

#### Section 6: Power Analysis

**Scripts:**
- `analysis/power_analysis.py` - Statistical power calculations for survey design

**Outputs:**
- `analysis/power_analysis_report.tex` - Power analysis results

---

## Setup

### Prerequisites

- Python 3.8+
- API keys for OpenAI, Anthropic, and Google (for model evaluation)

### Installation

```bash
git clone https://github.com/jonathanhchoi/llm-interpretation-replication.git
cd llm-interpretation-replication
pip install -r requirements.txt
```

### API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

Required keys:
- `OPENAI_API_KEY` - For GPT models
- `ANTHROPIC_API_KEY` - For Claude models
- `GEMINI_API_KEY` - For Gemini models
- `HUGGINGFACE_TOKEN` - For open-source model access

## Running the Analysis

### Quick Start: Reproduce Main Figures

```bash
# Prompt sensitivity analysis (Section 3)
python analysis/run_three_model_analysis.py

# Survey comparison (Section 4)
python analysis/evaluate_closed_source_models.py

# Base vs instruct comparison (Section 4.3)
python analysis/run_base_vs_instruct_100q.py
```

### Full Replication

1. **Generate prompt perturbations** (requires Claude API):
   ```bash
   python analysis/perturb_prompts.py
   ```

2. **Evaluate perturbations on multiple models**:
   ```bash
   python analysis/perturb_prompts_gpt.py
   python analysis/perturb_prompts_claude_batch.py
   python analysis/perturb_prompts_gemini_batch.py
   ```

3. **Run survey analysis**:
   ```bash
   python survey_analysis/survey_analysis_consolidated.py
   python analysis/evaluate_closed_source_models.py
   ```

4. **Generate visualizations**:
   ```bash
   python analysis/create_three_model_stacked_visualization.py
   python analysis/create_combined_visualization.py
   ```

## Data Files

| File | Description |
|------|-------------|
| `data/word_meaning_survey_results.csv` | Human survey responses (N=1003) |
| `data/demographic_data.csv` | Survey participant demographics |
| `data/perturbations.json` | 10,000 prompt perturbations (5 scenarios x 2000 each) |
| `data/model_comparison_results.csv` | Model evaluation results |
| `data/irrelevant_statements.txt` | 200 irrelevant factual statements for robustness tests |

## Configuration

Edit `analysis/config.py` to modify:
- Model selection and API endpoints
- Number of bootstrap iterations
- Output directory paths
- Statistical significance thresholds

## Citation

```bibtex
@article{choi2025llm,
  title={Off-the-Shelf Large Language Models Are Unreliable Judges},
  author={Choi, Jonathan H.},
  year={2025}
}
```

## License

MIT License. See `LICENSE` for details.

## Contact

For questions about the replication code, please open an issue on GitHub.
