from datetime import datetime
import os
import json
import csv
import time
import gc
from llama_cpp import Llama

# Config

# Path to the folder containing all your .gguf model files
MODEL_DIR = "D:/llm_models_to_evaluate"

# Path to standardized test dataset
DATASET_PATH = "./test_dataset.json"

# Timestamp string in the format YYYY_MM_DD_HH_MM
TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M")

# Final CSV results
RESULTS_PATH = f"./experiment_results_{TIMESTAMP}.csv"

# Parameters for loading and running the models
MODEL_PARAMS = {
    "n_gpu_layers": -1,  # GPU Offload
    "n_ctx": 4096,  # Context size
    "verbose": False
}

# Parameters for the generation step
GENERATION_PARAMS = {
    "temperature": 0.2,  # Low temp for more deterministic output
    "max_tokens": 512,  # Max tokens to generate for the summary
    "stream": True  # True to measure Time To First Token
}


def get_model_files(directory):
    """
    Finds all .gguf files in the specified directory.
    """
    gguf_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".gguf"):
            gguf_files.append(os.path.join(directory, filename))
    return gguf_files


def load_dataset(json_path):
    """
    Loads the test dataset from a JSON file.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Dataset file at {json_path} is not valid JSON.")
        return None


def format_prompt(test_case):
    """
    Formats the prompt using data from a test case.
    """
    person_name = test_case.get("person_name", "a participant")
    timeline_emotions = test_case.get("timeline_emotions", "[]")
    engagement_info = test_case.get("engagement_info", "")

    return (
        f"Write a concise, human-readable summary of {person_name}'s emotional and attentional (engagement) development "
        f"over time based on the following emotional sequence: {timeline_emotions}.{engagement_info}\n"
        f"You must not quote the raw sequence itself, but summarize the overall trend.\n"
        f"Summary:"
    )


def main():

    print("Starting experiment...")

    # Load dataset
    dataset = load_dataset(DATASET_PATH)
    if not dataset:
        return

    # Find models
    model_files = get_model_files(MODEL_DIR)
    if not model_files:
        print(f"ERROR: No .gguf models found in {MODEL_DIR}")
        return

    total_models = len(model_files)
    total_prompts = len(dataset)
    print(f"Found {total_models} models and {total_prompts} prompts.")

    # Prepare CSV output file
    csv_header = [
        "model_name", "sequence_id", "prompt_tokens", "completion_tokens",
        "ttft_seconds", "total_time_seconds", "tokens_per_second",
        "prompt_text", "generated_text", "rubric_score (manual)"
    ]

    try:
        with open(RESULTS_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
    except IOError as e:
        print(f"ERROR: Could not write to CSV file at {RESULTS_PATH}. {e}")
        return

    # Experiment
    for i, model_path in enumerate(model_files):
        model_name = os.path.basename(model_path)
        print(f"\n--- Loading Model {i + 1}/{total_models}: {model_name} ---")

        try:
            llm = Llama(model_path=model_path, **MODEL_PARAMS)
        except Exception as e:
            print(f"ERROR: Failed to load model {model_name}. Skipping. Error: {e}")
            continue

        for j, test_case in enumerate(dataset):
            print(f"  > Testing Model: {i + 1}/{total_models} | Prompt: {j + 1}/{total_prompts}", end='\r')

            prompt = format_prompt(test_case)
            sequence_id = test_case.get("sequence_id", f"seq_{j + 1}")

            # Reset metrics for each run
            ttft = 0.0
            first_token_time = None
            start_time = 0.0
            end_time = 0.0
            output_text = ""

            # Manual token count
            prompt_tokens = 0
            completion_tokens = 0

            try:
                prompt_tokens = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True)) # Tokenize prompt

                start_time = time.time() # Start tie before inference

                # Run the model with streaming
                stream = llm.create_completion(
                    prompt,
                    **GENERATION_PARAMS
                )

                # Streaming loop for each model
                for chunk in stream:

                    # ttft calculation
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = first_token_time - start_time

                    # Safely get the text content
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        output_text += chunk['choices'][0].get('text', '')

                end_time = time.time() # end time after inference

                if output_text:  # Only tokenize if text was generated
                    completion_tokens = len(llm.tokenize(output_text.encode("utf-8"), add_bos=False))

                total_time = end_time - start_time

                # Calculate generation time (= total time - ttft)
                generation_time = end_time - first_token_time if first_token_time else total_time

                # Edge case handling
                if generation_time <= 0 or completion_tokens <= 1:
                    tokens_per_sec = 0.0
                else:
                    # completion_tokens - 1 because the first token is measured by ttft, not generation speed
                    tokens_per_sec = ((completion_tokens - 1) / generation_time)
                    if tokens_per_sec < 0:
                        tokens_per_sec = 0.0  # Handle case of 1 token

                # CSV data
                result_data = [
                    model_name, sequence_id, prompt_tokens, completion_tokens,
                    ttft, total_time, tokens_per_sec,
                    prompt, output_text.strip(), ""  # Empty string for rubric score to be added
                ]

            # Log generation errors
            except Exception as e:
                print(f"\nERROR during generation for {model_name} on {sequence_id}. Error: {e}")
                result_data = [
                    model_name, sequence_id, prompt_tokens, 0, 0, 0, 0,
                    prompt, f"GENERATION_ERROR: {e}", ""
                ]

            # Write to the CSV file
            with open(RESULTS_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(result_data)

        # Clear model from VRAM
        del llm
        gc.collect()

    print(f"\n\n--- Experiment Complete! ---")
    print(f"Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    # Check for dependencies
    try:
        import llama_cpp.llama
    except ImportError:
        print("ERROR: 'llama-cpp-python' library not found.")
        exit(1)

    # CUDA Availability Check
    import llama_cpp

    llama_cpp.llama_backend_init() # Backend initialization

    print("\n--- CUDA Availability Check ---")

    if llama_cpp.llama_supports_gpu_offload():
        print("SUCCESS: llama.cpp reports that GPU offload is available.")
    else:
        print("WARNING: llama.cpp reports GPU offload is NOT available. \n The experiment will run on the CPU, "
              "which will be MUCH slower.")

    print("-------------------------------------------\n")

    main()