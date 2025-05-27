import argparse
import json
import os
from glob import glob
import random
from verl.utils.reward_score.ptp import format_reward

from datasets import Dataset

random.seed(1234)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reflections for incorrect traces")
    parser.add_argument(
        "--data_files",
        type=str,
        nargs='+',
        default=["data/final_data.json"],
        help="Files containing filtered data with incorrect traces (accepts multiple files or glob patterns)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Hugging Face repository ID",
    )
    return parser.parse_args()


def load_multiple_files(file_paths):
    all_data = []
    expanded_paths = []

    for path in file_paths:
        if '*' in path:
            expanded_paths.extend(glob(path))
        else:
            expanded_paths.append(path)

    expanded_paths = list(dict.fromkeys(expanded_paths))

    file_data = []
    file_names = []
    for file_path in expanded_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping.")
            continue
        print(f"Loading data from {file_path}...")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            file_data.append(data)
            file_names.append(file_path)
            print(f"Loaded {len(data)} samples from {file_path}")
        except json.JSONDecodeError:
            print(f"Error: {file_path} is not a valid JSON file, skipping.")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    if not file_data:
        print("No valid data found in any of the files.")
        return []

    baseline_size = len(file_data[-1])
    print(f"Using last file as baseline with {baseline_size} samples")

    for i, (data, file_path) in enumerate(zip(file_data, file_names)):
        if i == len(file_data) - 1:
            all_data.extend(data)
            print(f"Added all {len(data)} samples from baseline file: {file_path}")
            continue

        # For "bbh" files, sample 7 times the baseline
        if "bbh" in file_path.lower():
            sample_size = 7 * baseline_size
        elif "gsm" in file_path.lower() or "math" in file_path.lower():
            sample_size = int(0.5 * baseline_size)
        else:
            sample_size = baseline_size

        if sample_size > len(data):
            print(f"Warning: Requested {sample_size} samples from {file_path} but it only has {len(data)} samples.")
            sampled_data = data
        else:
            sampled_data = random.sample(data, sample_size)

        all_data.extend(sampled_data)

        if "bbh" in file_path.lower():
            print(f"Added {len(sampled_data)} samples (7x baseline) from BBH file: {file_path}")
        else:
            print(f"Added {len(sampled_data)} samples (0.5x baseline) from file: {file_path}")

    return all_data


def format_filter(sample):
    completion = sample["response"][-1]["content"]
    score = format_reward(completion, sample['metadata'])
    if score == 0:
        return False
    else:
        return True


if __name__ == "__main__":
    args = parse_args()

    hf_data = load_multiple_files(args.data_files)
    print(f"Loaded a total of {len(hf_data)} samples from {len(args.data_files)} file specifications")

    seen_instructions = set()
    hf_data = [
        sample for sample in hf_data if sample["instruction"][-1]["content"] not in seen_instructions and
        not seen_instructions.add(sample["instruction"][-1]["content"])
    ]
    print('=' * 50)
    print(f"Final dataset has {len(hf_data)} samples after removing duplicates by instruction")
    print('=' * 50)
    hf_data = [sample for sample in hf_data if format_filter(sample)]
    print(f"Final dataset has {len(hf_data)} samples after filtering by reward score")

    print("Pushing dataset to Hugging Face Hub...")
    dataset = Dataset.from_list(hf_data)
    dataset.push_to_hub(repo_id=args.repo_id, private=False)
    print("Done!")