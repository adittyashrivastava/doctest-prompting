import argparse
from collections import defaultdict
import random
import json
import os
import re

from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm

from utils import (
    check_answer,
    format_hf_data_baseline,
    clean_trace,
)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


VALID_TASKS_FOR_BBH = [
    "boolean_expressions",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Process data for machine learning datasets")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="../afs_data/doctest-prompting-data/logs2/anthropic-claude-3-sonnet-20240229",
        help="Directory containing the input data logs",
    )
    parser.add_argument(
        "--partial_programs_dir",
        type=str,
        default="../afs_data/doctest-prompting/bbh/mocks/partialprograms",
        help="Directory containing partial program files",
    )
    parser.add_argument("--save_file", type=str, default="data.json")
    return parser.parse_args()


def parse_log_file(filename, task):
    with open(filename, "r") as f:
        first_line = f.readline().strip()
        variant_match = re.search(r"variant='([^']*)'", first_line)
        model_match = re.search(r"model='([^']*)'", first_line)
        cot_match = re.search(r"CoT=([^,)\s]+)", first_line)
        baseline_match = re.search(r"baseline_template_format=([^,)\s]+)", first_line)
        if not (variant_match and model_match and cot_match and baseline_match):
            return []
        variant = variant_match.group(1)
        model = model_match.group(1)
        cot = cot_match.group(1)
        baseline = baseline_match.group(1)
        # only process files with CoT=True and baseline_template_format=True
        if cot != "True" or baseline != "True":
            return []
        data = f.read()
    # Group 1: Input block.
    # Group 2: Output block (trace).
    # Group 3: Prediction from the summary (to be used as final_answer).
    # Group 4: is_correct flag.
    pattern = (
        r"^------------------------------ input ------------------------------\s*\n"
        r"(.*?)\s*\n"  # group 1: input block
        r"^------------------------------ output ------------------------------\s*\n"
        r"(.*?)(?=\n^------------------------------\s*correct=)"  # group 2: output block (trace)
        r".*?prediction='([^']*)'.*?y='([^']*)'.*?is_correct=(True|False).*?------------------------------")
    matches = re.findall(pattern, data, re.MULTILINE | re.DOTALL)
    results = []
    correct = 0
    for input_text, trace, prediction, ground_truth, is_correct_str in tqdm(matches):
        is_correct, final_prediction = check_answer(trace, ground_truth.strip())
        if prediction == "**parse failed**":
            continue
        if final_prediction is None:
            continue
        is_correct = is_correct_str == "True"

        trace = clean_trace(trace)
        if not is_correct:
            continue
        correct += 1
        results.append({
            'task': f'bbh/{task}',
            'input': input_text.strip(),
            'trace': trace,
            'final_answer': prediction.strip(),
            'ground_truth': ground_truth.strip(),
        })
    summary_pattern = r"------------------------------\s*total=(\d+)"
    summary_matches = re.findall(summary_pattern, data)
    if summary_matches:
        expected_num = int(summary_matches[-1])
        if expected_num != len(results):
            print(f"[Warnings] Expected {expected_num} total traces but found {len(results)} from {filename}")
            return None
    # print updated accuracy
    total = len(matches)
    # save accuracy to a file in the same directory
    with open(f"{filename.replace('.log', '.txt')}", "w") as f:
        f.write(
            f"Task: {task}, Variant: {variant}, Model: {model}, Correct: {correct}, Total: {total}, Accuracy: {correct / total}"
        )
    return results


def main(args):
    data = []
    data_folders = args.data_folder.split(",")
    for data_folder in data_folders:
        for task in os.listdir(data_folder):
            if "bbh" in args.partial_programs_dir and task not in VALID_TASKS_FOR_BBH:
                print(f"Skipping task {task} because it is not valid for BBH")
                continue
            task_folder = os.path.join(data_folder, task)
            if not os.path.isdir(task_folder):
                continue
            json_files = [f for f in os.listdir(task_folder) if f.endswith(".json")]
            if json_files:
                raise NotImplementedError("JSON files are not supported yet. Please use log files instead.")
            else:
                # always use the base version
                log_file = "baseline-cot-test-000-000.log"
                file_path = os.path.join(task_folder, log_file)
                samples = parse_log_file(file_path, task)
                if samples:
                    data.extend(samples)
    print('=' * 50)
    print(f"We have {len(data)} before deduplication and filtering")
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, "w") as f:
        json.dump(data, f, indent=4)

    bbh_data = data
    data = load_dataset('openai/gsm8k', 'main', split='train')
    gsm8k_data = []
    for sample in data:
        answer = extract_answer(sample['answer'])
        response = sample['answer'].split('####')[0].strip()
        gsm8k_data.append({
            'task': 'gsm8k',
            'input': sample['question'],
            'trace': response,
            'final_answer': answer,
            'ground_truth': answer,
        })
    data = load_dataset('simplescaling/openaimath', split='train')
    math_data = []
    for sample in data:
        math_data.append({
            'task': 'math500',
            'input': sample['problem'],
            'trace': sample['solution'],
            'final_answer': sample['answer'],
            'ground_truth': sample['answer'],
        })
    data = load_dataset('PTPReasoning/MedCalc-Bench-v1.0', split='train')
    med_data = []
    for sample in data:
        task = 'medcalc_formulas' if sample['Category'] in ['lab test', 'physical', 'date', 'dosage conversion'
                                                           ] else 'medcalc_rules'
        med_data.append({
            'task': task,
            'input': sample['input'],
            'trace': sample['Ground Truth Explanation'],
            'final_answer': sample['Ground Truth Answer'],
            'ground_truth': sample['Ground Truth Answer'],
        })

    combined_data = concatenate_datasets([
        Dataset.from_list(bbh_data),
        Dataset.from_list(gsm8k_data),
        Dataset.from_list(math_data),
        Dataset.from_list(med_data),
    ])
    reference_data = load_dataset('PTPReasoning/PTP-SFT-ITL-Final-Clean', split='train')
    ref_inputs_by_task = defaultdict(list)
    for sample in reference_data:
        inp = sample['instruction'][-1]['content']
        ref_inputs_by_task[sample['task']].append(inp)
    combined_by_task = defaultdict(list)
    for sample in combined_data:
        combined_by_task[sample['task']].append(sample)

    filtered_data = []
    for task, ref_inputs in ref_inputs_by_task.items():
        pool = combined_by_task.get(task, [])
        hits = [s for s in pool if s['input'] in ref_inputs]

        needed = len(ref_inputs)
        have = len(hits)
        if have < needed:
            used_inputs = {s['input'] for s in hits}
            remain = [s for s in pool if s['input'] not in used_inputs]
            want = needed - have

            if len(remain) < want:
                print(f"Warnings! Task {task}: need {want} more unique samples, "
                      f"but only {len(remain)} remain.")
                want = len(remain)

            hits += random.sample(remain, want)

        elif have > needed:
            hits = random.sample(hits, needed)

        filtered_data.extend(hits)

    # dedupe
    seen = set()
    filtered_data = [s for s in filtered_data if s['input'] not in seen and not seen.add(s['input'])]

    if len(filtered_data) != len(reference_data):
        print(f"Warnings! Found {len(filtered_data)} samples, but expected {len(reference_data)}")

    hf_data = format_hf_data_baseline(filtered_data)
    hf_data = Dataset.from_list(hf_data)
    hf_data.push_to_hub(
        repo_id="PTPReasoning/PTP-SFT-ITL-Final-Baseline",
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
