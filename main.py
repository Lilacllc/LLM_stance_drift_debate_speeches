import matplotlib.pyplot as plt
from openai import OpenAI
from together import Together
import os
import json
import argparse
import sys

from utils import estimate_tran_mat, visualize_transition_matrices, NumpyArrayEncoder
from chat_client import UnifiedChatClient
import pdb

# Available models (for reference)
# models_available = [
#     "gpt-3.5-turbo",
#     "gpt-4o-mini",
#     "gpt-4o",
#     "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     "mistralai/Mistral-7B-Instruct-v0.3",
# ]

MODEL_NAME_MAP = {
    "gpt-4o-mini": "gpt_4o_mini",
    "gpt-3.5-turbo": "gpt_3_5_turbo",
    "gpt-4o": "gpt_4o",
    "meta-llama/Llama-3-8b-chat-hf": "llama3_8b",
    "google/gemma-3n-E4B-it": "gemma_3n_e4b",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "llama3_3_70b",
    "gemini-2.5-pro": "gemini_2_5_pro",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "gemini-2.5-flash-lite": "gemini_2_5_flash_lite",
}


REPITITION_EST_MAT = 100  # Number of times to repeat the experiment of "Est" setting
ITERATION = 6
SHUFFLE_REP = 5
letter_list = ["A", "B", "C", "D", "E"]
SEP = "="
PROMPT_CHOICE = 1


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run LLM predisposition analysis for a specific topic"
    )
    parser.add_argument(
        "topic_file_name",
        type=str,
        help="Topic file name (e.g., Age, Disability_status, Gender_identity, etc.)",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name to evaluate (e.g., gpt-4o-mini, gpt-3.5-turbo, etc.)",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to save results",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch API for faster processing (default: False)",
    )
    parser.add_argument(
        "--multiple_summarization",
        action="store_true",
        help="Use multiple summarization (encode multiple times with randomly shuffled option orders, average probabilities)",
    )
    parser.add_argument(
        "--summarization_count",
        type=int,
        default=5,
        help="Number of encodings with different shuffled option orders when using multiple summarization (default: 5)",
    )

    args = parser.parse_args()
    topic_file_name = args.topic_file_name
    model_name = args.model_name
    output_dir = args.output_dir
    use_batch = args.batch
    multiple_summarization = args.multiple_summarization
    summarization_count = args.summarization_count

    # Validate topic file exists
    topic_file_path = f"bbq_data/modified_{topic_file_name}.jsonl"
    examples_file_path = f"bbq_data/examples/modified_{topic_file_name}_examples.jsonl"

    if not os.path.exists(topic_file_path):
        print(f"Error: Topic file '{topic_file_path}' not found.")
        sys.exit(1)

    if not os.path.exists(examples_file_path):
        print(f"Error: Examples file '{examples_file_path}' not found.")
        sys.exit(1)

    # Validate batch API is only used with OpenAI models
    if use_batch and not model_name.startswith("gpt"):
        print("Error: Batch API is only supported with OpenAI models (gpt-*).")
        print("Either use an OpenAI model or remove the --batch flag.")
        sys.exit(1)

    # Create output directory structure
    topic_output_dir = os.path.join(output_dir, topic_file_name)
    print("Creating output directory:", topic_output_dir)
    os.makedirs(topic_output_dir, exist_ok=True)

    print(f"Processing topic: {topic_file_name}")
    print(f"Using model: {model_name}")
    print(f"Using batch API: {use_batch}")
    print(f"Using multiple summarization: {multiple_summarization}")
    if multiple_summarization:
        print(f"Summarization count: {summarization_count}")
    print(f"Output directory: {topic_output_dir}")

    examples_id = []
    with open(examples_file_path, "r") as file:
        for line in file:
            json_object = json.loads(line)
            examples_id.append(json_object["example_id"])

    data = []
    with open(topic_file_path, "r") as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)

    for example_id in examples_id:
        proposition = data[example_id]["proposition"]
        TOPIC = topic_file_name + f"_{data[example_id]['example_id']}_"

        # Evaluate the model
        print(f"Evaluating {model_name}")

        # Create unified client for regular API calls
        unified_client = UnifiedChatClient(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            together_api_key=os.environ.get("TOGETHER_API_KEY"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            timeout=30,
            max_retries=2,
        )

        # For batch API, we still need the original OpenAI client
        if use_batch:
            if "gpt" in model_name:
                batch_client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"], timeout=30, max_retries=2
                )
            else:
                print("Error: Batch API is only supported with OpenAI models (gpt-*).")
                sys.exit(1)
            client = batch_client
        else:
            client = unified_client
        # Create filename with batch indicator
        batch_suffix = "_batch" if use_batch else ""
        file_name = (
            TOPIC
            + MODEL_NAME_MAP[model_name]
            + "_prompt_"
            + str(PROMPT_CHOICE)
            #   + batch_suffix # no need to distinguish between batch and non-batch runs
        )

        # Create full file paths with the topic output directory
        log_file_path = os.path.join(topic_output_dir, file_name + ".log")
        json_file_path = os.path.join(topic_output_dir, file_name + ".json")
        raw_json_file_path = os.path.join(topic_output_dir, file_name + "_raw.json")

        if use_batch:
            print("Note: Batch processing may take several minutes to complete.")

        try:
            results_tranmat, results_probs = estimate_tran_mat(
                model=model_name,
                client=client,
                proposition=proposition,
                letters=letter_list,
                sep=SEP,
                repitition=REPITITION_EST_MAT,
                log_filename=log_file_path,
                max_tokens=200,
                max_argument_words=100,
                prompt_choice=PROMPT_CHOICE,
                use_batch=use_batch,
                multiple_summarization=multiple_summarization,
                summarization_count=summarization_count,
            )
        except:
            print(f"Error processing example {example_id} with model {model_name}.")
            print("Please check the log file for details:", log_file_path)
            continue

        # Save visualization to the topic output directory
        plot_file_path = os.path.join(topic_output_dir, file_name)
        visualize_transition_matrices(
            results_tranmat, letter_list, proposition, plot_file_path
        )

        try:
            with open(json_file_path, "w") as file_path:
                json.dump(results_tranmat, file_path, cls=NumpyArrayEncoder, indent=4)
            with open(raw_json_file_path, "w") as file_path:
                json.dump(results_probs, file_path, cls=NumpyArrayEncoder, indent=4)
        except:
            pdb.set_trace()

        print(f"Completed processing for example {example_id}")
        print(f"Results saved to: {json_file_path}")
        print(f"Visualization saved to: {plot_file_path}.png")
        print("-" * 50)


if __name__ == "__main__":
    main()
