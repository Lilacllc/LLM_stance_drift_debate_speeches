"""
Postprocess: Statistical Analysis of LLM Probability Outputs

This script performs postprocessing and statistical analysis on LLM output probability matrices.
It generates three heatmap visualizations for:
1. Percentage of confidence intervals including 1
2. Mean diagonal probabilities
3. Success rates (probability > 0.5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from pathlib import Path
from scipy import stats

# Configuration
plt.style.use("ggplot")

LETTERS = ["A", "B", "C", "D", "E"]

TOPICS = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_gender",
    "Race_x_SES",
    "Religion",
    "SES",
    "Sexual_orientation",
]

# Statistical analysis parameters
confidence = 0.99
epsilon = 1e-10
method = "clt"  # 'hoeffding', 'clt'
THRE_SUCCESS = 0.9  # Threshold for success rate (probability > 0.5)

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


def create_heatmap(data, value_column, title, filename, model_name, cmap="YlOrRd"):
    """
    Create and save a heatmap visualization.

    Args:
        data: DataFrame with results
        value_column: Column to use for heatmap values
        title: Plot title
        filename: Output filename (without path)
        model_name: Model name for directory structure
        cmap: Colormap for heatmap
    """
    # Create output directory
    output_dir = f"postprocess_results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Create full file path
    full_filename = os.path.join(output_dir, filename)

    # Compute mean and se pivot tables (as percentages)
    def standard_error(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))
    heatmap_mean = data.pivot_table(
        values=value_column, index="Letter", columns="Category", aggfunc="mean"
    )
    heatmap_se = data.pivot_table(
        values=value_column, index="Letter", columns="Category", aggfunc= standard_error
    )

    # Convert to percent and round
    mean_percent = (heatmap_mean * 100).round().astype(int)
    se_percent = (heatmap_se * 100).round().astype(int)

    # Create annotation matrix: "mean (sd)"
    annot = mean_percent.astype(str) + " (" + se_percent.astype(str) + ")"

    # Update title to mention percent
    title_with_percent = title + "\n(Values in %; Annot: Mean (SE))"

    # Create heatmap
    fig, ax = plt.subplots(figsize=(32, 10))
    sns.heatmap(
        heatmap_mean,
        cmap=cmap,
        annot=annot.values,
        annot_kws={"fontsize": 30},
        fmt="",
        vmin=0,
        vmax=1,
    )
    ax.figure.axes[-1].yaxis.set_tick_params(labelsize=30)
    ax.set_title(title_with_percent, fontsize=40, fontweight="bold")
    ax.set_xticklabels([tick.get_text().replace("_", "_\n") for tick in ax.get_xticklabels()], rotation=30, fontsize=30, fontweight="bold")
    ax.set_xlabel("Topic", fontsize=30, fontweight="bold")
    yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    ax.set_yticklabels([letter_to_option(str(l)).replace(" ", "\n") for l in yticklabels], rotation=0, fontsize=30, fontweight="bold")
    ax.set_ylabel("Initial Stance", fontsize=30, fontweight="bold")
    plt.tight_layout()
    plt.savefig(full_filename, bbox_inches="tight", dpi=500)
    plt.close()
    print(f"Saved heatmap: {full_filename}")


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
        "--batch",
        action="store_true",
        help="Use batch results (adds '_batch' to file paths)",
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
    BATCH = "_batch" if args.batch else ""
    THRE_SUCCESS = args.threshold
    CONFIDENCE = args.confidence
    EPSILON = args.epsilon
    METHOD = args.method
    TEMP = args.temp
    PROMPT_ID = args.prompt_id

    print("Starting postprocessing analysis...")
    print(f"Model: {MODEL}{SUFFIX}")
    print(f"Batch mode: {args.batch}")
    print(f"Success threshold: {THRE_SUCCESS}")
    print(f"Confidence level: {CONFIDENCE}")
    print(f"Method: {METHOD}")
    print(f"Epsilon: {EPSILON}")
    print("-" * 50)

    results = []
    success_rates_results = []

    for topic in TOPICS:
        print(f"Processing topic: {topic}")

        # Get all example IDs for this topic
        examples_id = []
        examples_file = f"bbq_data/examples/modified_{topic}_examples.jsonl"

        if not os.path.exists(examples_file):
            print(f"  Warning: Examples file not found: {examples_file}")
            continue

        with open(examples_file, "r") as file:
            for line in file:
                json_object = json.loads(line)
                examples_id.append(json_object["example_id"])

        # Process each domain file for this topic
        files_processed = 0
        for example_id in examples_id:
            domain = f"{topic}_{example_id}"
            file_path = f"experiment_results/{MODEL}{SUFFIX}/{topic}/{domain}_{MODEL}_prompt_{PROMPT_ID}{BATCH}_raw.json"

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
                                confidence=1 - (1 - CONFIDENCE) / 20,
                                method=METHOD,
                            )
                        )

                        includes_1 = ci_low <= 1 <= ci_high
                        includes_1_bonferroni = (
                            ci_low_bonferroni <= 1 <= ci_high_bonferroni
                        )
                        success_rate = np.mean(
                            samples > THRE_SUCCESS
                        )  # Adjusted threshold for success rate

                        results.append(
                            {
                                "Category": topic,
                                "Example": domain,
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

                        success_rates_result = {}
                        success_rates_result["Category"] = topic
                        success_rates_result["Example"] = domain
                        success_rates_result["Perm"] = perm_key
                        success_rates_result["Letter"] = l
                        for thre_success in THRE_SUCCESS_SQ:
                            success_rates_result[f"Success_rate_{thre_success}"] = (
                                np.mean(samples > thre_success)
                            )
                        success_rates_results.append(success_rates_result)

            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                continue

        print(f"  Processed {files_processed} files")

    if not results:
        print("No results found! Check your file paths and configuration.")
        return

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print(f"\nTotal results: {len(results_df)} rows")

    # Summary by topic
    summary_by_topic = (
        results_df.groupby("Category")[
            ["Includes_1", "Includes_1_bonferroni", "Mean", "Success_rate"]
        ]
        .mean()
        .reset_index()
    )
    summary_by_topic.columns = [
        "Category",
        "Percentage_CI_includes_1",
        "Percentage_CI_includes_1_bonferroni",
        "Mean",
        "Success_Rate",
    ]

    print("\nSummary by topic:")
    print(summary_by_topic)

    # Save summary to text file
    if TEMP == 0:
        SUFFIX = SUFFIX + "_temp0"
    output_dir = f"postprocess_results/{MODEL}{SUFFIX}"
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "summary_by_topic.txt")

    with open(summary_file, "w") as f:
        f.write("Summary by Category - Statistical Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {MODEL}{SUFFIX}\n")
        f.write(f"Batch mode: {args.batch}\n")
        f.write(f"Success threshold: {THRE_SUCCESS}\n")
        f.write(f"Confidence level: {CONFIDENCE}\n")
        f.write(f"Method: {METHOD}\n")
        f.write(f"Epsilon: {EPSILON}\n")
        f.write("=" * 50 + "\n\n")
        f.write(summary_by_topic.to_string(index=False))
        f.write("\n\n")
        f.write("Column Descriptions:\n")
        f.write(
            "- Percentage_CI_includes_1: Percentage of confidence intervals that include 1.0\n"
        )
        f.write("- Mean: Average diagonal probability (consistency measure)\n")
        f.write("- Success_Rate: Fraction of cases with probability above threshold\n")

    print(f"Saved summary to: {summary_file}")

    # Create and save heatmaps
    print("\nGenerating heatmaps...")

    # Heatmap 1: Percentage of CIs including 1
    create_heatmap(
        results_df,
        "Includes_1",
        f"Percentage of Confidence Intervals Including 1 \n({int(CONFIDENCE*100)}% confidence interval, 1-sided)",
        "heatmap_includes_1.pdf",
        f"{MODEL}{SUFFIX}",
        "YlOrRd",
    )

    # Heatmap 2: Percentage of CIs including 1 (Bonferroni corrected)
    create_heatmap(
        results_df,
        "Includes_1_bonferroni",
        f"Percentage of Bonferroni-Corrected Confidence Intervals Including 1", #\n({int(CONFIDENCE*100)}% confidence interval, 1-sided)
        f"heatmap_includes_1_bonferroni.pdf",
        f"{MODEL}{SUFFIX}",
        "YlOrRd",
    )

    # Heatmap 3: Mean diagonal probabilities
    create_heatmap(
        results_df,
        "Mean",
        "Mean Diagonal Probabilities",
        "heatmap_mean_probabilities.pdf",
        f"{MODEL}{SUFFIX}",
        "Blues",
    )

    # Heatmap 4: Success rates
    create_heatmap(
        results_df,
        "Success_rate",
        f"Success Rates \n(Fraction with probability > {THRE_SUCCESS})",
        "heatmap_success_rates.pdf",
        f"{MODEL}{SUFFIX}",
        "Greens",
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

    print("\nAnalysis complete!")
    print("Generated files:")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/summary_by_topic.txt")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/heatmap_includes_1.pdf")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/heatmap_mean_probabilities.pdf")
    print(f"- postprocess_results/{MODEL}{SUFFIX}/heatmap_success_rates.pdf")


if __name__ == "__main__":
    main()
