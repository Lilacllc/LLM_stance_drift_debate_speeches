import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from postprocess import letter_to_option
# Model name mapping from main.py
MODEL_NAME_MAP = {
    "gpt-4o-mini": "gpt_4o_mini",
    "gpt-3.5-turbo": "gpt_3_5_turbo",
    "gpt-4o": "gpt_4o",
    "gpt-4.1": "gpt_4_1",
    "meta-llama/Llama-3-8b-chat-hf": "llama3_8b",
    "google/gemma-3n-E4B-it": "gemma_3n_e4b",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "llama3_3_70b",
    "gemini-2.5-pro": "gemini_2_5_pro",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "gemini-2.5-flash-lite": "gemini_2_5_flash_lite",
}
letter_list = ["A", "B", "C", "D", "E"]

def get_data_dir(topic_file_name, example_id, model_name, prompt_choice, output_dir="experiment_results"):
    """
    Reconstruct plot based on given parameters.
    
    Args:
        topic_file_name (str): Topic name (e.g., "Teaser", "Age")
        example_id (int): Example ID from the dataset
        model_name (str): Model name (e.g., "gpt-4o-mini")
        prompt_choice (int): Prompt choice number
        output_dir (str): Base output directory
    """
    
    
    # Load proposition from data file
    topic_file_path = f"bbq_data/modified_{topic_file_name}.jsonl"
    with open(topic_file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["example_id"] == example_id:
                proposition = data["proposition"]
                break
    
    # Construct file paths
    EXAMPLE = f"{topic_file_name}_{example_id}"
    file_name = f"{EXAMPLE}_{MODEL_NAME_MAP[model_name]}_prompt_{prompt_choice}_raw"
    
    # Load results matrix
    topic_output_dir = os.path.join(output_dir, topic_file_name)
    json_file_path = os.path.join(topic_output_dir, file_name + ".json")
    
    return json_file_path, f"{EXAMPLE}_{MODEL_NAME_MAP[model_name]}_prompt_{prompt_choice}"

def visualize_transition_matrices(results_tranmat, letter_list, title, plot_file_path, present_se = False, panel='left'):
    if panel=='left': 
        cbar = False
        width = 14.724
    elif panel=='right':
        cbar = True
        width = 14.308
    elif panel=='center':
        cbar = False
        width = 11.557 
    else:
        cbar = True
        width = 15
    
    fig, ax = plt.subplots(
        1, 1, figsize=(width, 15)
        )

    if present_se:
        mean = pd.DataFrame((results_tranmat["mean"])* 100).round().astype(int)
        se = pd.DataFrame((results_tranmat["se"])* 100).round().astype(int)
        # Create annotation matrix: "mean (se)"
        annot = mean.astype(str) + " (" + se.astype(str) + ")"
        # Update title to mention percent
        title = title + "\n(Values in %; Annot: Mean (SE))"
        sns.heatmap(
            mean,
            ax=ax,
            cmap="Reds",
            cbar=cbar,
            annot=annot.values,
            annot_kws={"fontsize": 35},
            fmt="",
            vmin=0,
            vmax=100
            )
    else:
        sns.heatmap(np.round(results_tranmat, 2), ax=ax, annot=True, cmap="Reds", cbar=cbar, vmin=0, vmax=1, annot_kws={"fontsize": 35})

    xticks = np.arange(0, len(letter_list), 1)
    letter_xticks = [letter_to_option(letter_list[tick]) for tick in list(xticks)]
    yticks = np.arange(len(letter_list) - 1, -1, -1)
    letter_yticks = [letter_to_option(letter_list[tick]) for tick in list(yticks)]

    ax.set_title(title, fontsize=40, fontweight="bold")
    ax.set_xlabel("To State", fontsize=35, fontweight="bold")
    ax.set_xticks(xticks + 0.5, [tick.replace(" ", "\n") for tick in letter_xticks], fontsize=35, rotation=30, fontweight="bold", ha="center")

    if panel=='right' or panel=='center':
        ax.set_ylabel(None)
        ax.set_yticks([],[])
    if panel=='right' or panel=='single':
        ax.figure.axes[-1].yaxis.set_tick_params(labelsize=35)
    if panel=='left' or panel=='single':
        ax.set_ylabel("From State", fontsize=35, fontweight="bold")
        ax.set_yticks(yticks + 0.5, [tick.replace(" ", "\n") for tick in letter_yticks], fontsize=35, rotation=30, fontweight="bold", ha="right")


    plt.tight_layout()
    plt.savefig(f"{plot_file_path}.pdf", bbox_inches="tight", dpi=500)
    #plt.show()
    
    print(f"Plot reconstructed and saved to: {plot_file_path}.pdf")
    return

# Example usage:
def main():
    # Modify these parameters as needed
    json_file_path, file_name = get_data_dir(
        topic_file_name="Teaser",
        example_id=22,
        model_name="gpt-4.1",  
        prompt_choice=1
    )
    plot_file_path = os.path.join("figures", file_name)
    with open(json_file_path, "r") as file:
        data = json.load(file)
    perm_key = [perm_key for perm_key in data.keys()][0]
    arr = np.asarray(data[perm_key])
    mean = np.mean(arr, axis=0)
    se = np.std(arr, axis=0, ddof=1)/np.sqrt(arr.shape[0])

    visualize_transition_matrices({'mean': mean, 'se':se}, letter_list, "Empirical Transition Matrix\n(Mixed Pattern)", plot_file_path+"_se", present_se = True, panel='right')

    visualize_transition_matrices(mean, letter_list, "Empirical Transition Matrix\n(Mixed Pattern)", plot_file_path, present_se = False, panel='center')

    visualize_transition_matrices(np.eye(5), letter_list, "Identity Matrix\n(Perfect Stance Preservation)", "figures/ideal", present_se=False, panel='right')

    json_file_path, file_name = get_data_dir(
        topic_file_name="Teaser",
        example_id=1,
        model_name="gpt-4.1",  
        prompt_choice=1
    )
    plot_file_path = os.path.join("figures", file_name)
    with open(json_file_path, "r") as file:
        data = json.load(file)
    perm_key = [perm_key for perm_key in data.keys()][0]
    arr = np.asarray(data[perm_key])
    mean = np.mean(arr, axis=0)
    se = np.std(arr, axis=0, ddof=1)/np.sqrt(arr.shape[0])

    visualize_transition_matrices({'mean': mean, 'se':se}, letter_list, "Empirical Transition Matrix\n(Polarization)", plot_file_path+"_se", present_se = True, panel='left')

    visualize_transition_matrices(mean, letter_list, "Empirical Transition Matrix\n(Polarization)", plot_file_path, present_se = False, panel='left')

if __name__ == "__main__":
    main()