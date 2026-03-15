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

| Table/Figure | Script | Command |
|--------------|--------|---------|
| Figure 2A, 2B | `visualization.py` | `python visualization.py --topic_id <ID> --model gpt_5_4 --title "..." --panel ...` |
| Figure 2C | `visualization.py` | `python visualization.py --ideal-only --topic_id dummy` |
| Table 1A, 1B | `faithfulness_metric.py` | `python faithfulness_metric.py --latex-tables` |
| Figure 5 | `faithfulness_metric.py` | `python faithfulness_metric.py` |
| Figure S3A, S3B | `visualization.py` | `python visualization.py --topic_id <ID> --model gpt_5_4 --title "..." --panel ...` |
| Table S1-S4 | `faithfulness_metric.py` | `python faithfulness_metric.py --latex-tables --comprehensive` |
| Figure S4 | `faithfulness_metric.py` | `python faithfulness_metric.py --comprehensive` |
| Figure S5 | `success_rate_metric.py` | `python success_rate_metric.py --all-letters-only` |
| Data S1 (all sheets) | `export_data_s1.py` | `python export_data_s1.py` |

---

## Detailed Commands

### Transition Matrix Heatmaps (Figure 2, S3)

**Script:** `visualization.py`

**Output:** `figures/`

```bash
# Figure 2C - Identity matrix (perfect stance preservation)
python visualization.py --ideal-only --topic_id dummy

# Figure 2A - Example: Polarization pattern (replace TOPIC_ID_1 with actual proposition ID)
python visualization.py --topic_id 2401 --model gpt_5_4 --title '\"We should increase fuel tax\"\n-Empirical Transition Matrix' --panel left

# Figure 2B - Example: Mixed pattern (replace TOPIC_ID_2 with actual proposition ID)
python visualization.py --topic_id 3234 --model gpt_5_4 --title '\"The use of AI should be abandoned\"\n-Empirical Transition Matrix' --panel center

# Figure S3A, S3B - SE-annotated versions are generated alongside each mean-only figure
# (outputs: debate_speech_{TOPIC_ID}_{model}_prompt_1.pdf and debate_speech_{TOPIC_ID}_{model}_prompt_1_se.pdf)
```

**Options:**
- `--topic_id`: Proposition topic_id from propositions.json (required unless `--ideal-only`)
- `--model`: Internal model name (default: `gpt_4o_mini`)
- `--prompt_id`: Prompt ID (default: `1`)
- `--title`: Plot title (default: `"Empirical Transition Matrix"`)
- `--panel`: Panel layout — `left`, `center`, `right`, or `single` (default: `single`)
- `--ideal-only`: Only generate the ideal identity matrix figure

---

### Model Bar Plots (prerequisite for Table 1, S1–S4, Figure S4-5)

**Script:** `postprocess.py`

**Output:** `postprocess_results/{model}/` and `postprocess_results/model_comparison.csv`

Run these commands to generate per-model bar plots and populate `model_comparison.csv`, which is used by `faithfulness_metric.py --latex-tables` to produce Table 1A,B and Table S1–S4.

```bash
# Base model
python postprocess.py --model gpt_4o_mini

# Variants (reversed option order, temperature=0)
python postprocess.py --model gpt_4o_mini --suffix _reversed
python postprocess.py --model gpt_4o_mini --temp 0

# Other models
python postprocess.py --model gpt_4_1
python postprocess.py --model gpt_5_4
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

### Model Comparison Tables (Table 1, S1-S4)

**Script:** `faithfulness_metric.py`

**Output:** `figures/`

**Prerequisites:** Run `postprocess.py` for all models first (generates `model_comparison.csv`).

```bash
# Table 1A, 1B - Subset (9 models): Mean probability & Bonferroni CI inclusion
python faithfulness_metric.py --latex-tables

# Table S1-S4 - Comprehensive (14 models): All 4 metrics
python faithfulness_metric.py --latex-tables --comprehensive
```

**Output files:**
- `figures/latex_tables_subset.tex` — 4 LaTeX table snippets (subset models)
- `figures/latex_tables_comprehensive.tex` — 4 LaTeX table snippets (all models)

---

### SPR Comparison Bar Charts (Figure 5, S4)

**Script:** `faithfulness_metric.py`

**Output:** `figures/`

```bash
# Figure 5 - Stance Preservation Rate comparison (9 models)
python faithfulness_metric.py

# Figure S4 - Comprehensive comparison (14 models)
python faithfulness_metric.py --comprehensive
```

**Options:**
- `--comprehensive`: Include all 14 models (adds llama3_3_70b, llama3_1_8b, gpt_4o_mini_reversed, gpt_4o_mini_temp0)
- `--latex-tables`: Generate LaTeX table snippets instead of bar plots
- `--errorbar`: Show error bars
- `--neutral-only`: Use only neutral (letter C) data

---

### Success Rate Analysis (Figure S5)

**Script:** `success_rate_metric.py`

**Output:** `figures/`

**Prerequisites:** Run `postprocess.py` for all models first.

```bash
# Figure S5(A) - Success rate vs threshold curves
# Figure S5(B) - AUC comparison bar plot
python success_rate_metric.py --all-letters-only
```

**Output files:**
- `success_rate_vs_threshold_all_letters.pdf`
- `auc_barplot_all_letters.pdf`
- `auc_results_all_letters.csv`

---

### Supplementary Data S1 (Multi-Sheet Excel)

**Script:** `export_data_s1.py`

**Output:** `figures/data_S1.xlsx`

**Prerequisites:** Run `postprocess.py` for all models and `success_rate_metric.py` first.

```bash
python export_data_s1.py
```

**Sheets (12 total):**

| Sheet | Corresponds To |
|-------|---------------|
| `Fig2A_S3A` | Transition matrix for Fig 2A & S3A (topic 2401, gpt-5.4) |
| `Fig2B_S3B` | Transition matrix for Fig 2B & S3B (topic 3234, gpt-5.4) |
| `Table1A` | Avg Probability of Stance Preservation (9 models) |
| `Table1B` | Bonferroni-Corrected CI Inclusion Rate (9 models) |
| `Fig5` | SPR bar chart (9 models) |
| `TableS1` | Avg Probability of Stance Preservation (14 models) |
| `TableS2` | Bonferroni-Corrected CI Inclusion Rate (14 models) |
| `TableS3` | CI Inclusion Rate (14 models) |
| `TableS4` | Success Rate (14 models) |
| `FigS4` | SPR bar chart (14 models) |
| `FigS5A` | Success-rate vs threshold curves (14 models) |
| `FigS5B` | AUC comparison (14 models) |

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

- **Temperature=0** outputs save to `postprocess_results/{model}_temp0/`
- Each `postprocess.py` run appends to `postprocess_results/model_comparison.csv`; re-running for the same model updates its rows

---

## Dependencies

See `requirements.txt` for full list:
- numpy, pandas, matplotlib, seaborn, scipy
- openai, together, google-genai (for running new experiments)
- sentence-transformers, scikit-learn (for clustering)

---

*Last updated: February 2026*

