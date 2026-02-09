# Reproducing Figures

This document provides verified commands for reproducing all figures in the paper.

---

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set API keys (only needed for running new experiments):
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export TOGETHER_API_KEY="your_together_api_key"
   export GEMINI_API_KEY="your_gemini_api_key"
   ```

---

## Quick Reference

| Figure | Script | Command |
|--------|--------|---------|
| Figure 2A, 2B, 2C | `visualization.py` | `python visualization.py` |
| Figure 4A, 4B | `postprocess.py` | `python postprocess.py --model gpt_4o_mini` |
| Figure 5 | `faithfulness_metric.py` | `python faithfulness_metric.py` |
| Figure S4A, S4B | `visualization.py` | `python visualization.py` |
| Figure S5 | `postprocess.py` | `python postprocess.py --model gpt_4o_mini --suffix _reversed` |
| Figure S6 | `postprocess.py` | `python postprocess.py --model gpt_4o_mini --temp 0` |
| Figure S7 | `postprocess.py` | See [Model Bar Plots](#model-bar-plots) |
| Figure S8 | `faithfulness_metric.py` | `python faithfulness_metric.py --comprehensive` |
| Figure S9 | `postprocess.py` | `python postprocess.py --model gpt_4o_mini` |
| Figure S10 | `success_rate_metric.py` | `python success_rate_metric.py --all-letters-only` |

---

## Detailed Commands

### Transition Matrix Heatmaps (Figure 2, S4)

**Script:** `visualization.py`

**Output:** `figures/`

```bash
# Generates all transition matrix figures at once:
# - Teaser_1_gpt_4_1_prompt_1.pdf (Figure 2A - Polarization)
# - Teaser_22_gpt_4_1_prompt_1.pdf (Figure 2B - Mixed Pattern)
# - ideal.pdf (Figure 2C - Perfect Preservation)
# - Teaser_1_gpt_4_1_prompt_1_se.pdf (Figure S4A)
# - Teaser_22_gpt_4_1_prompt_1_se.pdf (Figure S4B)

python visualization.py
```

---

### Model Bar Plots (Figure 4, S5, S6, S7, S9)

**Script:** `postprocess.py`

**Output:** `postprocess_results/{model}/` and `postprocess_results/model_comparison.csv`

```bash
# Figure 4A, 4B, S9 - GPT-4o-mini (main model)
python postprocess.py --model gpt_4o_mini

# Figure S5 - GPT-4o-mini with reversed option order
python postprocess.py --model gpt_4o_mini --suffix _reversed

# Figure S6 - GPT-4o-mini with temperature=0
python postprocess.py --model gpt_4o_mini --temp 0

# Figure S7 - Other models
python postprocess.py --model gpt_4_1
python postprocess.py --model gpt_3_5_turbo
python postprocess.py --model gemma_3n_e4b
python postprocess.py --model llama3_3_70b
python postprocess.py --model llama3_1_8b
python postprocess.py --model llama4_maverick
python postprocess.py --model qwen3_a3b
```

**Output files per model:**
- `barplot_mean_probabilities.pdf` — Mean diagonal probabilities
- `barplot_includes_1.pdf` — Confidence intervals including 1
- `barplot_includes_1_bonferroni.pdf` — Bonferroni-corrected CIs
- `barplot_success_rates.pdf` — Success rates
- `summary.txt` — Statistical summary
- `postprocess_detailed_results.csv` — Per-proposition detailed results
- `postprocess_success_rates_results.csv` — Success rates at multiple thresholds

**Cross-model output:**
- `postprocess_results/model_comparison.csv` — Accumulates mean and SE of all 4 metrics across models (updated on each run)

---

### SPR Comparison Bar Charts (Figure 5, S8)

**Script:** `faithfulness_metric.py`

**Output:** `figures/`

```bash
# Figure 5 - Stance Preservation Rate comparison (8 models)
python faithfulness_metric.py

# Figure S8 - Comprehensive comparison (11 models)
python faithfulness_metric.py --comprehensive
```

**Options:**
- `--comprehensive`: Include all 11 models (adds llama3_8b, gpt_4o_mini_reversed, gpt_4o_mini_temp0)
- `--errorbar`: Show error bars
- `--neutral-only`: Use only neutral (letter C) data

---

### Success Rate Analysis (Figure S10)

**Script:** `success_rate_metric.py`

**Output:** `metrics/`

**Prerequisites:** Run `postprocess.py` for all models first.

```bash
# Figure S10(A) - Success rate vs threshold curves
# Figure S10(B) - AUC comparison bar plot
python success_rate_metric.py --all-letters-only
```

**Output files:**
- `success_rate_vs_threshold_all_letters.pdf`
- `auc_barplot_all_letters.pdf`
- `auc_results_all_letters.csv`

---

### Ablation Studies

Generate data for ablation analysis (used by `success_rate_metric.py` and `faithfulness_metric.py`):

```bash
# Assertion prompt ablation (prompt_id 9)
python postprocess.py --model gpt_4o_mini --suffix _assert --prompt_id 9

# Multiple summarization ablation
python postprocess.py --model gpt_4o_mini --suffix _multiple_summarization

# In-context learning ablation (prompt_id 8)
python postprocess.py --model gpt_4o_mini --suffix _in_context --prompt_id 8
```

---

## Notes

- **LLaMA-3.1-8B** may show NaN warnings (handled gracefully)
- **Temperature=0** outputs save to `postprocess_results/{model}_temp0/`
- Some ablation studies and smaller models may have incomplete data for certain propositions
- Each `postprocess.py` run appends to `postprocess_results/model_comparison.csv`; re-running for the same model updates its rows

---

## Dependencies

See `requirements.txt` for full list:
- numpy, pandas, matplotlib, seaborn, scipy
- openai, together, google-genai (for running new experiments)
- sentence-transformers, scikit-learn (for clustering)

---

*Last updated: February 2026*

