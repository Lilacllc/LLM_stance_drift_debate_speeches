import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from name_maps import get_latex_name, get_plot_name

# Directory containing all model/suffix result folders
RESULTS_DIR = "postprocess_results"
CSV_NAME = "postprocess_detailed_results.csv"
MODEL_COMPARISON_CSV = "model_comparison.csv"

# Metric definitions: (mean_column, se_column, display_title)
LATEX_METRICS = [
    ("Mean_mean", "Mean_standard_error",
     "Average Probability of Stance Preservation"),
    ("Includes_1_bonferroni_mean", "Includes_1_bonferroni_standard_error",
     "Bonferroni-Corrected CI Inclusion Rate"),
    ("Includes_1_mean", "Includes_1_standard_error",
     "CI Inclusion Rate"),
    ("Success_rate_mean", "Success_rate_standard_error",
     "Success Rate"),
]

STANCE_ORDER = ["A", "B", "C", "D", "E"]
STANCE_LABELS = {
    "A": "Agree strongly",
    "B": "Agree",
    "C": "Neutral",
    "D": "Disagree",
    "E": "Disagree strongly",
}


def latex_escape(s):
    """Escape underscores for LaTeX (fallback for unknown names)."""
    return s.replace("_", "\\_")


def generate_latex_tables(model_dirs, comprehensive=False):
    """
    Generate LaTeX table snippets from model_comparison.csv.

    Produces 4 tables (one per metric). Each table has:
    - Rows: initial stances (Agree strongly ... Disagree strongly)
    - Columns: model/variant names
    - Cells: mean (SE) in percentage with 1 decimal place

    Args:
        model_dirs: list of model directory names to include
        comprehensive: if True, label output as comprehensive

    Returns:
        path to the saved .tex file, or None on error
    """
    comparison_file = os.path.join(RESULTS_DIR, MODEL_COMPARISON_CSV)
    if not os.path.exists(comparison_file):
        print(f"Error: {comparison_file} not found. "
              "Run postprocess.py for all models first.")
        return None

    df = pd.read_csv(comparison_file)

    # Filter to requested models, preserving order
    df = df[df["Model"].isin(model_dirs)]
    available_models = [m for m in model_dirs if m in df["Model"].values]

    if not available_models:
        print("Error: No matching models found in model_comparison.csv.")
        return None

    n_stances = len(STANCE_ORDER)
    col_spec = "l" + "c" * n_stances

    output_lines = []

    for mean_col, se_col, title in LATEX_METRICS:
        # Comment header
        output_lines.append(f"% {title}")
        output_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        output_lines.append("\t\t\\\\")
        output_lines.append("\t\t\\hline")

        # Header row: stance names as columns
        header = "Initial Stance"
        for letter in STANCE_ORDER:
            header += f" & {STANCE_LABELS[letter]}"
        header += " \\\\"
        output_lines.append(f"\t\t{header}")

        # Unit row
        unit_row = ""
        for _ in STANCE_ORDER:
            unit_row += " & (\\%)"
        unit_row += " \\\\"
        output_lines.append(f"\t\t{unit_row}")
        output_lines.append("\t\t\\hline")

        # Data rows: one row per model
        for model in available_models:
            row = get_latex_name(model)
            for letter in STANCE_ORDER:
                cell = df[(df["Model"] == model) & (df["Letter"] == letter)]
                if len(cell) == 0:
                    row += " & N/A"
                else:
                    mean_val = cell[mean_col].values[0] * 100
                    se_val = cell[se_col].values[0] * 100
                    row += f" & {mean_val:.1f} ({se_val:.1f})"
            row += " \\\\"
            output_lines.append(f"\t\t{row}")

        output_lines.append("\t\t\\hline")
        output_lines.append("\\end{tabular}")
        output_lines.append("")
        output_lines.append("")

    # Save to file
    mode_label = "comprehensive" if comprehensive else "subset"
    output_file = f"figures/latex_tables_{mode_label}.tex"
    os.makedirs("figures", exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))

    print(f"LaTeX tables saved to: {output_file}")
    print(f"  Models included: {len(available_models)}")
    print(f"  Tables generated: {len(LATEX_METRICS)}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Plot average diagonal probability for each model."
    )
    parser.add_argument(
        "--errorbar", action="store_true", help="Plot error bars (std) for each bar"
    )
    parser.add_argument(
        "--neutral-only",
        action="store_true",
        help="Use only neutral letter (C) data instead of all letters",
    )
    parser.add_argument(
        "--metric",
        choices=["raw", "negative_log", "baseline"],
        default="raw",
        help="Metric type: raw (default), negative_log, or baseline (difference from gpt_3_5_turbo)",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Use comprehensive model list (13 models) instead of subset (9 models)",
    )
    parser.add_argument(
        "--latex-tables",
        action="store_true",
        help="Generate LaTeX table snippets from model_comparison.csv "
             "(subset: Table 1A,B; comprehensive: Table S1-S4)",
    )
    args = parser.parse_args()

    # Find all model/suffix subdirectories
    # Variant models (gpt_4o_mini ablations/configurations) — colored brown
    VARIANT_MODELS = {
        "gpt_4o_mini_reversed",
        "gpt_4o_mini_temp0",
        "gpt_4o_mini_multiple_summarization",
        "gpt_4o_mini_in_context",
        "gpt_4o_mini_assert",
    }

    if args.comprehensive:
        # Comprehensive list (13 models) for Figure S8
        model_dirs = [
            "gpt_4o_mini",
            "gpt_4_1",
            "gpt_3_5_turbo",
            "gemma_3n_e4b",
            "llama3_3_70b",
            "llama3_1_8b",
            "llama4_maverick",
            "qwen3_a3b",
            "gpt_4o_mini_reversed",
            "gpt_4o_mini_temp0",
            "gpt_4o_mini_multiple_summarization",
            "gpt_4o_mini_in_context",
            "gpt_4o_mini_assert",
        ]
    else:
        # Subset list (9 models) for Figure 5
        model_dirs = [
            "gpt_4o_mini",
            "gpt_4_1",
            "gpt_3_5_turbo",
            "gemma_3n_e4b",
            "llama3_3_70b",
            "llama3_1_8b",
            "llama4_maverick",
            "qwen3_a3b",
        ]

    # If --latex-tables, generate tables and exit
    if args.latex_tables:
        generate_latex_tables(model_dirs, comprehensive=args.comprehensive)
        return

    # Store average diagonal probabilities for each model
    model_averages = {}
    model_stds = {}

    # For baseline comparison, we need to load all data first
    all_model_data = {}

    for model_dir in model_dirs:
        csv_path = os.path.join(RESULTS_DIR, model_dir, CSV_NAME)
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # Filter data based on letter selection
        if args.neutral_only:
            df = df[df["Letter"] == "C"]

        # Store the data for baseline comparison
        if args.metric == "baseline":
            all_model_data[model_dir] = df["Mean"].values
        else:
            # Compute average diagonal probability across selected rows
            if args.metric == "negative_log":
                # Use negative log of mean values
                avg_diagonal_prob = -np.log(df["Mean"] + 1e-6).mean()
                std_diagonal_prob = np.log(df["Mean"] + 1e-6).std()
            else:
                # Use raw mean values
                avg_diagonal_prob = (df["Mean"]).mean()
                std_diagonal_prob = (df["Mean"]).std()

            model_averages[model_dir] = avg_diagonal_prob
            model_stds[model_dir] = std_diagonal_prob

    # Handle baseline comparison
    if args.metric == "baseline":
        baseline_model = "gpt_3_5_turbo"  #'gpt_3_5_turbo'
        if baseline_model not in all_model_data:
            print(f"Error: Baseline model '{baseline_model}' not found in results.")
            return

        baseline_data = all_model_data[baseline_model]

        # Compute differences from baseline for each model
        for model_dir, model_data in all_model_data.items():
            if model_dir == baseline_model:
                continue

            # Ensure same length by truncating to minimum length
            min_length = min(len(baseline_data), len(model_data))
            baseline_subset = baseline_data[:min_length]
            model_subset = model_data[:min_length]

            # Compute differences from baseline
            differences = model_subset - baseline_subset

            # Calculate mean and std of differences
            avg_difference = differences.mean()
            std_difference = differences.std()

            model_averages[model_dir] = avg_difference
            model_stds[model_dir] = std_difference

    # Create bar plot
    if args.comprehensive:
        plt.figure(figsize=(32.5, 14))  # Larger size for comprehensive comparison
    else:
        plt.figure(figsize=(22.5, 14))  # Standard size for subset
    models = list(model_averages.keys())
    averages = list(model_averages.values())
    stds = list(model_stds.values())

    # Create bar plot with optional error bars
    if args.errorbar:
        bars = plt.bar(range(len(models)), averages, yerr=stds, capsize=5, alpha=0.5)
    else:
        bars = plt.bar(range(len(models)), averages, capsize=5, alpha=0.5)

    # Color specific bars: variants in brown, base models in dark blue
    for i, (bar, model) in enumerate(zip(bars, models)):
        if model in VARIANT_MODELS:
            bar.set_color("brown")
        else:
            bar.set_color("darkblue")

    # Customize the plot
    if args.metric == "negative_log":
        plt.ylabel("-log(SPR)", fontsize=40, fontweight="bold")
        metric_type = "Negative Log"
    elif args.metric == "baseline":
        plt.ylabel(f"SPR Difference from {baseline_model}", fontsize=40, fontweight="bold")
        metric_type = "Baseline"
    else:
        plt.ylabel("SPR", fontsize=40, fontweight="bold")
        metric_type = "Raw"
        plt.ylim(0, 1.05*max(averages))
    plt.yticks(fontsize=30, fontweight="bold")

    title = f"Stance Preservation Rate (SPR) Across Models"
    if args.neutral_only:
        title += " (Neutral Only)"
    elif args.metric == "baseline":
        title += f"\n (Value of Baseline Model: {baseline_subset.mean():.3f})"
    plt.title(title, fontsize=45, fontweight="bold")

    # Set x-axis labels
    plt.xlabel("Models", fontsize=40, fontweight="bold")
    models_labels = [get_plot_name(model) for model in models]
    plt.xticks(range(len(models)), models_labels, rotation=30, ha="center", fontsize=22, fontweight="bold")

    # Add value labels inside bars
    for i, (bar, avg) in enumerate(zip(bars, averages)):
        # Position text inside the bar
        if avg >= 0:
            # For positive values, place text in the middle of the bar
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{avg:.3f}",
                ha="center",
                va="center",
                fontsize=32,
                color="black",
                fontweight="semibold",
            )
        else:
            # For negative values, place text in the middle of the bar
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{avg:.3f}",
                ha="center",
                va="center",
                fontsize=32,
                color="black",
                fontweight="semibold",
            )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    # Set output filename based on mode
    if args.comprehensive:
        output_filename = "figures/average_diagonal_probability_by_model_comprehensive.pdf"
    else:
        output_filename = "figures/average_diagonal_probability_by_model.pdf"
    
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    plt.savefig(output_filename, dpi=500, bbox_inches="tight")
    plt.close()

    # Print summary statistics
    data_type = "Neutral Only" if args.neutral_only else "All Letters"
    if args.metric == "negative_log":
        metric_type = "Negative Log"
    elif args.metric == "baseline":
        metric_type = f"Baseline (vs {baseline_model})"
    else:
        metric_type = "Raw"

    mode_type = "Comprehensive (13 models, Figure S8)" if args.comprehensive else "Subset (9 models, Figure 5)"
    print(f"Average Diagonal Probability by Model ({metric_type}, {data_type}, {mode_type}):")
    print("=" * 70)
    for model, avg in model_averages.items():
        std = model_stds[model]
        print(f"{model}: {avg:.4f} ± {std:.4f}")

    print(f"\nPlot saved: {output_filename}")
    print(f"Total models analyzed: {len(model_averages)}")
    if args.comprehensive:
        print("Using comprehensive model list (13 models).")
    else:
        print("Using subset model list (9 models).")
    if args.errorbar:
        print("Error bars (std) are included.")
    if args.neutral_only:
        print("Using only neutral letter (C) data.")
    else:
        print("Using all letters data.")
    if args.metric == "negative_log":
        print("Using negative log scale.")
    elif args.metric == "baseline":
        print(f"Using baseline comparison ({baseline_model}).")
    else:
        print("Using raw scale.")


if __name__ == "__main__":
    main()
