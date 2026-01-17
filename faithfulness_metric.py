import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Directory containing all model/suffix result folders
RESULTS_DIR = "postprocess_results"
CSV_NAME = "postprocess_detailed_results.csv"


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
        help="Use comprehensive model list (11 models for Figure S8) instead of subset (8 models for Figure 5)",
    )
    args = parser.parse_args()

    # Find all model/suffix subdirectories
    if args.comprehensive:
        # Comprehensive list (11 models) for Figure S8
        model_dirs = [
            "gemini_2_5_flash_lite",
            "gemma_3n_e4b",
            "gpt_3_5_turbo",
            "gpt_4o_mini",
            "llama3_3_70b",
            "llama3_8b",
            "gpt_4o_mini_reversed",
            "gpt_4o_mini_temp0",
            "gpt_4o_mini_double_summarization",
            "gpt_4o_mini_in_context",
            "gpt_4o_mini_assert",
        ]
    else:
        # Subset list (8 models) for Figure 5
        model_dirs = [
            "gemini_2_5_flash_lite",
            "gemma_3n_e4b",
            "gpt_3_5_turbo",
            "gpt_4o_mini",
            "llama3_3_70b",
            "gpt_4o_mini_double_summarization",
            "gpt_4o_mini_in_context",
            "gpt_4o_mini_assert",
        ]

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
        plt.figure(figsize=(26, 14))  # Larger size for comprehensive comparison
    else:
        plt.figure(figsize=(20, 14))  # Standard size for subset
    models = list(model_averages.keys())
    averages = list(model_averages.values())
    stds = list(model_stds.values())

    # Create bar plot with optional error bars
    if args.errorbar:
        bars = plt.bar(range(len(models)), averages, yerr=stds, capsize=5, alpha=0.7)
    else:
        bars = plt.bar(range(len(models)), averages, capsize=5, alpha=0.7)

    # Color specific bars
    for i, (bar, model) in enumerate(zip(bars, models)):
        if (
            model == "gpt_4o_mini_double_summarization"
            or model == "gpt_4o_mini_in_context"
            or model == "gpt_4o_mini_assert"
            or model == "gpt_4o_mini_reversed"
            or model == "gpt_4o_mini_temp0"
        ):
            bar.set_color("brown")
        else:
            bar.set_color("darkblue")

    # Customize the plot
    if args.metric == "negative_log":
        plt.ylabel("-log(SPR)", fontsize=30, fontweight="bold")
        metric_type = "Negative Log"
    elif args.metric == "baseline":
        plt.ylabel(f"SPR Difference from {baseline_model}", fontsize=30, fontweight="bold")
        metric_type = "Baseline"
    else:
        plt.ylabel("SPR", fontsize=30, fontweight="bold")
        metric_type = "Raw"
        plt.ylim(0.95*min(averages), 1.05*max(averages))
    plt.yticks(fontsize=22, fontweight="bold")

    title = f"Stance Preservation Rate (SPR) Across Models"
    if args.neutral_only:
        title += " (Neutral Only)"
    elif args.metric == "baseline":
        title += f"\n (Value of Baseline Model: {baseline_subset.mean():.3f})"
    plt.title(title, fontsize=40, fontweight="bold")

    # Set x-axis labels
    plt.xlabel("Models", fontsize=30, fontweight="bold")
    models_labels = [model.replace("gpt_4o_mini_", "gpt_4o_mini\n_").replace("_summarization", "\n_summarization")  for model in models]
    plt.xticks(range(len(models)), models_labels, rotation=30, ha="center", fontsize=22, fontweight="bold")

    # Add value labels inside bars
    for i, (bar, avg) in enumerate(zip(bars, averages)):
        # Position text inside the bar
        if avg >= 0:
            # For positive values, place text in the middle of the bar
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.01,
                f"{avg:.3f}",
                ha="center",
                va="center",
                fontsize=30,
                color="white",
                fontweight="bold",
            )
        else:
            # For negative values, place text in the middle of the bar
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{avg:.3f}",
                ha="center",
                va="center",
                fontsize=30,
                color="white",
                fontweight="bold",
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

    mode_type = "Comprehensive (Figure S8)" if args.comprehensive else "Subset (Figure 5)"
    print(f"Average Diagonal Probability by Model ({metric_type}, {data_type}, {mode_type}):")
    print("=" * 70)
    for model, avg in model_averages.items():
        std = model_stds[model]
        print(f"{model}: {avg:.4f} ± {std:.4f}")

    print(f"\nPlot saved: {output_filename}")
    print(f"Total models analyzed: {len(model_averages)}")
    if args.comprehensive:
        print("Using comprehensive model list (11 models).")
    else:
        print("Using subset model list (8 models).")
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
