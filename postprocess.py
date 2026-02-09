"""
Postprocess: Statistical Analysis of LLM Probability Outputs

This script performs postprocessing and statistical analysis on LLM output probability matrices
from debate speech experiments. It generates four bar plot visualizations for:
1. Percentage of confidence intervals including 1
2. Percentage of Bonferroni-corrected confidence intervals including 1
3. Average probability of stance preservation
4. Success rates (probability > threshold)

Results are aggregated across propositions (from propositions.json) for each initial stance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import argparse
from scipy import stats

# Configuration
plt.style.use("ggplot")

LETTERS = ["A", "B", "C", "D", "E"]

# Statistical analysis parameters
confidence = 0.99
epsilon = 1e-10
method = "clt"  # 'hoeffding', 'clt'
THRE_SUCCESS = 0.9  # Threshold for success rate (probability > 0.9)

THRE_SUCCESS_SQ = np.round(np.arange(0, 1.05, 0.05), 2)


def load_json(file_path):
    """Load JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_diagonal_probs(data, perm_key, letters=LETTERS, temp=1):
    """
    Extract the diagonal (i,i) probabilities for each initial letter from all experiments.

    Args:
        data: Experimental data containing probability matrices
        perm_key: Key for accessing specific permutation data
        letters: List of response letters ['A', 'B', 'C', 'D', 'E']

    Returns:
        dict: {letter: [list of 100 diagonal probabilities]}
    """
    diag_probs = {}
    for i, l in enumerate(letters):
        # 100 experiments, 5 initials, dim-5 probability vectors
        arr = np.asarray(data[perm_key])

        if temp == 0:
            concentrated = np.zeros_like(arr)
            concentrated[
                np.arange(arr.shape[0])[:, None, None],
                np.arange(arr.shape[1])[None, :, None],
                np.argmax(arr, axis=2)[:, :, None],
            ] = 1
            arr = concentrated

        else:
            # Apply concentration with sharpening parameter
            arr = arr**temp
            arr = arr / arr.sum(axis=2, keepdims=True)  # Normalize to sum to 1

        # Extract diagonal probabilities for letter i
        diag_vec = arr[:, i, i]
        diag_probs[l] = diag_vec  # take the i-th column for i-th initial

    return diag_probs


def compute_confidence_interval(samples, confidence=0.95, method="clt"):
    """
    Compute confidence interval for the mean of samples.
    One sided test:
        Null hypothesis: the mean is equal to 1
        Alternative hypothesis: the mean is less than 1

    Parameters:
        samples (array-like): Data samples
        confidence (float): Confidence level (default 0.95)
        method (str): 'clt' for central limit theorem (default), 'hoeffding' for Hoeffding's inequality

    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """

    # Check for NaN values
    if np.any(np.isnan(samples)):
        print("N.A. exists!")
        return 0, 0, 0

    n = len(samples)
    mean = np.mean(samples)
    upper = mean
    lower = mean

    if method == "clt":
        std = np.std(samples, ddof=1)
        se = std / np.sqrt(n)
        h = stats.t.ppf(confidence, n - 1) * se
        upper = mean + h
        lower = 0

    elif method == "hoeffding":
        # Hoeffding's inequality for bounded variables in [a, b]
        # For probabilities, a=0, b=1
        eps = np.sqrt((1 / (2 * n)) * np.log(1 / (1 - confidence)))
        upper = mean + eps
        lower = 0

    return mean, lower, upper


def letter_to_option(letter):
    """Convert letter to response option."""
    letter = letter.strip().upper()
    if letter == "A":
        return "Agree strongly"
    elif letter == "B":
        return "Agree"
    elif letter == "C":
        return "Neutral"
    elif letter == "D":
        return "Disagree"
    elif letter == "E":
        return "Disagree strongly"


def create_barplot(data, value_column, title, filename, model_name, color="steelblue"):
    """
    Create and save a bar plot visualization.

    Args:
        data: DataFrame with results
        value_column: Column to use for bar values
        title: Plot title
        filename: Output filename (without path)
        model_name: Model name for directory structure
        color: Bar color
    """
    # Create output directory
    output_dir = f"postprocess_results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Create full file path
    full_filename = os.path.join(output_dir, filename)

    # Compute mean and SE for each initial stance (letter)
    grouped = data.groupby("Letter")[value_column]
    means = grouped.mean()
    ses = grouped.apply(lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))

    # Ensure ordering follows LETTERS
    labels = [letter_to_option(l) for l in LETTERS]
    mean_values = [means[l] for l in LETTERS]
    se_values = [ses[l] for l in LETTERS]

    # Convert to percentages for display
    mean_pct = [m * 100 for m in mean_values]
    se_pct = [s * 100 for s in se_values]

    fig, ax = plt.subplots(figsize=(15, 10))
    x_pos = np.arange(len(LETTERS))
    bars = ax.bar(
                x_pos,
                mean_pct,
                yerr=se_pct,
                capsize=30,
                width=0.85,
                color=color,
                edgecolor="#333333",
                alpha=0.5,
                error_kw={"linewidth": 2.2, "ecolor": "#2B2B2B"},
            )

    # Add value labels in the middle of each bar: "mean (SE)"
    for i, (bar, m, s) in enumerate(zip(bars, mean_pct, se_pct)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, 
            max(bar.get_height()-10.0, 0.0),
            f"{m:.1f} ({s:.1f})",
            ha="center", 
            va="bottom", 
            fontsize=25,            # slightly larger
            fontweight="semibold",  # lighter than bold
            color="black",       
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=24, fontweight="bold")
    ax.set_xlabel("Initial Stance", fontsize=30, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=30, fontweight="bold", labelpad=12)
    title_with_annot = title + "\n(Values in %; Annot: Mean (SE))"
    ax.set_title(title_with_annot, fontsize=34, fontweight="bold", pad=18)
    ax.set_ylim(0, min(115, max(mean_pct) + max(se_pct) + 15))
    ax.tick_params(axis="y", labelsize=24)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(full_filename, bbox_inches="tight", dpi=500)
    plt.close()
    print(f"Saved bar plot: {full_filename}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Postprocess LLM probability outputs and generate statistical analysis"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt_4o_mini",
        help="Model name (default: gpt_4o_mini)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Threshold for success rate calculation (default: 0.95)",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Additional suffix for model directory (e.g., '_reversed')",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.99,
        help="Confidence level for statistical analysis (default: 0.99)",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-10,
        help="Epsilon parameter for statistical calculations (default: 1e-10)",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=1,
        help="Temperature parameter for temperature scaling (default: 1)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="clt",
        choices=["clt", "hoeffding"],
        help="Statistical method for confidence intervals (default: clt)",
    )
    parser.add_argument(
        "--prompt_id",
        type=int,
        default=1,
        help="Prompt ID to use for file naming (default: 1)",
    )

    return parser.parse_args()


def main():
    """Main processing function."""
    args = parse_arguments()

    # Set parameters from command line arguments
    MODEL = args.model
    SUFFIX = args.suffix
    THRE_SUCCESS = args.threshold
    CONFIDENCE = args.confidence
    EPSILON = args.epsilon
    METHOD = args.method
    TEMP = args.temp
    PROMPT_ID = args.prompt_id

    print("Starting postprocessing analysis...")
    print(f"Model: {MODEL}{SUFFIX}")
    print(f"Success threshold: {THRE_SUCCESS}")
    print(f"Confidence level: {CONFIDENCE}")
    print(f"Method: {METHOD}")
    print(f"Epsilon: {EPSILON}")
    print("-" * 50)

    # Load propositions
    propositions = load_json("propositions.json")
    num_propositions = len(propositions)
    print(f"Loaded {num_propositions} propositions from propositions.json")

    results = []
    success_rates_results = []
    files_processed = 0

    for prop in propositions:
        topic_id = prop["topic_id"]
        topic = prop["topic"]

        file_path = (
            f"results/{MODEL}{SUFFIX}/debate_speech_{topic_id}"
            f"_{MODEL}_prompt_{PROMPT_ID}_raw.json"
        )

        if not os.path.exists(file_path):
            continue  # skip missing files

        try:
            data = load_json(file_path)
            files_processed += 1

            for perm_key in data.keys():
                diag_probs = extract_diagonal_probs(
                    data, perm_key=perm_key, letters=LETTERS, temp=TEMP
                )

                for i, l in enumerate(LETTERS):
                    samples = diag_probs[l]
                    mean, ci_low, ci_high = compute_confidence_interval(
                        samples, confidence=CONFIDENCE, method=METHOD
                    )
                    mean, ci_low_bonferroni, ci_high_bonferroni = (
                        compute_confidence_interval(
                            samples,
                            confidence=1 - (1 - CONFIDENCE) / num_propositions,
                            method=METHOD,
                        )
                    )

                    includes_1 = ci_low <= 1 <= ci_high
                    includes_1_bonferroni = (
                        ci_low_bonferroni <= 1 <= ci_high_bonferroni
                    )
                    success_rate = np.mean(samples > THRE_SUCCESS)

                    results.append(
                        {
                            "Proposition_ID": topic_id,
                            "Proposition": topic,
                            "Perm": perm_key,
                            "Letter": l,
                            "Mean": mean,
                            "CI_low": ci_low,
                            "CI_high": ci_high,
                            "Includes_1": includes_1,
                            "Includes_1_bonferroni": includes_1_bonferroni,
                            "Success_rate": success_rate,
                        }
                    )

                    success_rates_result = {
                        "Proposition_ID": topic_id,
                        "Proposition": topic,
                        "Perm": perm_key,
                        "Letter": l,
                    }
                    for thre_success in THRE_SUCCESS_SQ:
                        success_rates_result[f"Success_rate_{thre_success}"] = (
                            np.mean(samples > thre_success)
                        )
                    success_rates_results.append(success_rates_result)

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue

    print(f"Processed {files_processed} / {num_propositions} proposition files")

    if not results:
        print("No results found! Check your file paths and configuration.")
        return

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print(f"\nTotal results: {len(results_df)} rows")

    # Summary by initial stance (Letter)
    def standard_error(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))

    summary_by_stance = (
        results_df.groupby("Letter")[
            ["Includes_1", "Includes_1_bonferroni", "Mean", "Success_rate"]
        ]
        .agg(["mean", standard_error])
    )
    # Flatten multi-level columns
    summary_by_stance.columns = [
        f"{col}_{stat}" for col, stat in summary_by_stance.columns
    ]
    summary_by_stance = summary_by_stance.reset_index()
    # Add readable stance labels
    summary_by_stance.insert(
        1, "Stance", summary_by_stance["Letter"].apply(letter_to_option)
    )

    print("\nSummary by initial stance:")
    print(summary_by_stance.to_string(index=False))

    # Save summary to text file
    if TEMP == 0:
        SUFFIX = SUFFIX + "_temp0"
    output_dir = f"postprocess_results/{MODEL}{SUFFIX}"
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "summary.txt")

    with open(summary_file, "w") as f:
        f.write("Summary by Initial Stance - Statistical Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {MODEL}{SUFFIX}\n")
        f.write(f"Propositions processed: {files_processed} / {num_propositions}\n")
        f.write(f"Success threshold: {THRE_SUCCESS}\n")
        f.write(f"Confidence level: {CONFIDENCE}\n")
        f.write(f"Method: {METHOD}\n")
        f.write(f"Epsilon: {EPSILON}\n")
        f.write(f"Temperature: {TEMP}\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary_by_stance.to_string(index=False))
        f.write("\n\n")
        f.write("Column Descriptions:\n")
        f.write(
            "- Includes_1_mean: Percentage of confidence intervals that include 1.0\n"
        )
        f.write(
            "- Includes_1_bonferroni_mean: Same with Bonferroni correction "
            f"(÷{num_propositions} propositions)\n"
        )
        f.write("- Mean_mean: Average diagonal probability (stance preservation)\n")
        f.write(
            "- Success_rate_mean: Fraction of cases with probability above threshold\n"
        )
        f.write("- *_standard_error: Standard error of the corresponding metric\n")

    print(f"Saved summary to: {summary_file}")

    # Create and save bar plots
    print("\nGenerating bar plots...")

    # Bar plot 1: Percentage of CIs including 1
    create_barplot(
        results_df,
        "Includes_1",
        f"Percentage of Confidence Intervals Including 1\n"
        f"({int(CONFIDENCE * 100)}% confidence interval, 1-sided)",
        "barplot_includes_1.pdf",
        f"{MODEL}{SUFFIX}",
        "#7A2E2E",
    )

    # Bar plot 2: Percentage of CIs including 1 (Bonferroni corrected)
    create_barplot(
        results_df,
        "Includes_1_bonferroni",
        f"Percentage of Confidence Intervals Including 1\n"
        f"(Bonferroni-corrected)",
        "barplot_includes_1_bonferroni.pdf",
        f"{MODEL}{SUFFIX}",
        "#8F4E2A",
    )

    # Bar plot 3: Average probability of stance preservation
    create_barplot(
        results_df,
        "Mean",
        "Average Probability of Stance Preservation",
        "barplot_mean_probabilities.pdf",
        f"{MODEL}{SUFFIX}",
        "#2F4B7C",
    )

    # Bar plot 4: Success rates
    create_barplot(
        results_df,
        "Success_rate",
        f"Success Rates\n(Fraction with probability > {THRE_SUCCESS})",
        "barplot_success_rates.pdf",
        f"{MODEL}{SUFFIX}",
        "#2E6F62",
    )

    # Save success rates results to CSV
    success_rates_results_df = pd.DataFrame(success_rates_results)
    success_rates_results_df.to_csv(
        f"postprocess_results/{MODEL}{SUFFIX}/postprocess_success_rates_results.csv",
        index=False,
    )

    # Save detailed results to CSV
    results_df.to_csv(
        f"postprocess_results/{MODEL}{SUFFIX}/postprocess_detailed_results.csv",
        index=False,
    )

    # Save / update cross-model comparison CSV
    model_label = f"{MODEL}{SUFFIX}"
    model_summary = summary_by_stance.copy()
    model_summary.insert(0, "Model", model_label)

    comparison_file = "postprocess_results/model_comparison.csv"
    if os.path.exists(comparison_file):
        existing = pd.read_csv(comparison_file)
        # Remove old rows for this model (allows re-runs to update)
        existing = existing[existing["Model"] != model_label]
        combined = pd.concat([existing, model_summary], ignore_index=True)
    else:
        combined = model_summary

    combined.to_csv(comparison_file, index=False)
    print(f"Updated cross-model comparison: {comparison_file}")

    print("\nAnalysis complete!")
    print("Generated files:")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/summary.txt")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/barplot_includes_1.pdf")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/barplot_includes_1_bonferroni.pdf")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/barplot_mean_probabilities.pdf")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/barplot_success_rates.pdf")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/postprocess_detailed_results.csv")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/postprocess_success_rates_results.csv")
    print(f"- {comparison_file}")


if __name__ == "__main__":
    main()
