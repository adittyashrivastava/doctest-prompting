import argparse
import json
import os
import re

from tqdm import tqdm

from utils import (
    check_answer,
    clean_trace,
    construct_partial_programs,
    extract_functions_from_partial_programs,
    extract_step_outputs,
    format_hf_data,
    parse_partial_programs,
    format_clean_hf_data,
)

# VALID_TASKS_FOR_BBH = [
#     "boolean_expressions",
#     "date_understanding",
#     #"disambiguation_qa",
#     "formal_fallacies",
#     "geometric_shapes",
#     "logical_deduction_three_objects",
#     # "movie_recommendation",
#     "multistep_arithmetic_two",
#     "navigate",
#     "object_counting",
#     "penguins_in_a_table",
#     # "reasoning_about_colored_objects",
#     "ruin_names",
#     # "salient_translation_error_detection",
#     # "sports_understanding",
#     "temporal_sequences",
#     "tracking_shuffled_objects_three_objects",
#     # "web_of_lies",
# ]

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


def parse_log_file(filename, partial_program_dir, task):
    with open(filename, "r") as f:
        # read and parse info from the namespace line.
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
        # only process files with CoT=False and baseline_template_format=False
        if cot != "False" or baseline != "False":
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
    partial_program_funcs = parse_partial_programs(partial_program_dir, task, variant)
    for input_text, trace, prediction, ground_truth, is_correct_str in tqdm(matches):
        is_correct, final_prediction = check_answer(trace, ground_truth.strip())
        if prediction == "**parse failed**":
            continue
        if final_prediction is None:
            continue
        is_correct = is_correct_str == "True" if "bbh" in partial_program_dir else is_correct

        if not is_correct:
            continue

        trace = clean_trace(trace.strip())
        step_outputs = extract_step_outputs(None, trace.split("\n"))
        if step_outputs is None or step_outputs == []:
            continue
        partial_program_funcs = parse_partial_programs(partial_program_dir, task, variant)
        partial_program = construct_partial_programs(partial_program_funcs, step_outputs)
        results.append({
            'task':
                f'bbh/{task}' if "bbh" in partial_program_dir else task,
            'variant':
                variant,
            'model':
                model,
            'input':
                input_text.strip(),
            'trace':
                trace,
            'final_answer':
                final_prediction
                if final_prediction is not None and "bbh" not in partial_program_dir else prediction.strip(),
            'ground_truth':
                ground_truth.strip(),
            'partial_program':
                partial_program,
            'is_correct':
                float(is_correct),
            'step_outputs': [step.fn_name for step in step_outputs],
            'partial_program_funcs':
                extract_functions_from_partial_programs(partial_program),
            'total_functions_defined':
                len(partial_program_funcs) - 1,  # exclude the extra
            'actual_functions_used':
                len(extract_functions_from_partial_programs(partial_program)),
        })
    summary_pattern = r"------------------------------\s*total=(\d+)"
    summary_matches = re.findall(summary_pattern, data)
    if summary_matches:
        expected_num = int(summary_matches[-1])
        if expected_num != len(results):
            print(f"[Warnings] Expected {expected_num} total traces but found {len(results)} from {filename}")
            return None
    # print updated accuracy
    correct = sum(1 for result in results if result["is_correct"])
    total = len(matches)
    # save accuracy to a file in the same directory
    with open(f"{filename.replace('.log', '.txt')}", "w") as f:
        f.write(
            f"Task: {task}, Variant: {variant}, Model: {model}, Correct: {correct}, Total: {total}, Accuracy: {correct / total}"
        )
    return results


# TODO: update the parse_json_file
# but we might not need it for now
def parse_json_file(filename, partial_program_dir):
    with open(filename, "r") as f:
        data = json.load(f)
    task = data["task"]
    results_dict = data["results"]
    variant = data["metadata"]["command_args"]["variant"]
    model = data["metadata"]["model"]
    partial_program = parse_partial_programs(partial_program_dir, task, variant)
    final_results = []
    for result_id, result in tqdm(results_dict.items()):
        extra_info = result.get("extra_info", None)
        is_correct, final_prediction = check_answer(result["output"], result["target"], extra_info)
        if result['parse_failed']:
            continue
        if final_prediction is None:
            continue

        is_correct = result["is_correct"] if "bbh" in partial_program_dir else is_correct
        prediction = result["prediction"]

        if not is_correct:
            continue

        trace = clean_trace(result['output'].strip())
        step_outputs = extract_step_outputs(None, trace.split("\n"))
        if step_outputs is None or step_outputs == []:
            continue
        partial_program_funcs = parse_partial_programs(partial_program_dir, task, variant)
        partial_program = construct_partial_programs(partial_program_funcs, step_outputs)
        final_results.append({
            'task':
                f'bbh/{task}' if "bbh" in partial_program_dir else task,
            'variant':
                variant,
            'model':
                model,
            'input':
                result["input"].strip(),
            'trace':
                trace,
            'final_answer':
                final_prediction
                if final_prediction is not None and "bbh" not in partial_program_dir else prediction.strip(),
            'ground_truth':
                result["target"].strip(),
            'partial_program':
                partial_program,
            'is_correct':
                float(is_correct),
            'step_outputs': [step.fn_name for step in step_outputs],
            'partial_program_funcs':
                extract_functions_from_partial_programs(partial_program),
            'total_functions_defined':
                len(partial_program_funcs) - 1,  # exclude the extra
            'actual_functions_used':
                len(extract_functions_from_partial_programs(partial_program)),
            'extra_info':
                extra_info,
        })
    # print updated accuracy
    correct = sum(1 for result in final_results if result["is_correct"])
    total = len(results_dict)
    # print(f"Task: {task}, Variant: {variant}, Model: {model}, Correct: {correct}, Total: {total}, Accuracy: {correct/total}")
    with open(f"{filename.replace('.json', '.txt')}", "w") as f:
        f.write(
            f"Task: {task}, Variant: {variant}, Model: {model}, Correct: {correct}, Total: {total}, Updated Accuracy: {correct / total}"
        )
    return final_results


def group_by_funcs(data):
    groups = {}
    for sample in data:
        task = sample.get("task")
        funcs = sample.get("partial_program_funcs", {})
        func_keys = frozenset(key for key in funcs if key != "extra")
        key = (task, func_keys)
        groups.setdefault(key, []).append(sample)
    return groups


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
                for json_file in json_files:
                    file_path = os.path.join(task_folder, json_file)
                    samples = parse_json_file(file_path, args.partial_programs_dir)
                    if samples:
                        data.extend(samples)
            else:
                # always use the base version
                log_file = "baseline-dpt-test-000-000.log"
                file_path = os.path.join(task_folder, log_file)
                samples = parse_log_file(file_path, args.partial_programs_dir, task)
                if samples:
                    data.extend(samples)

    print('=' * 50)
    print(f"We have {len(data)} before deduplication and filtering")
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, "w") as f:
        json.dump(data, f, indent=4)

    groups = group_by_funcs(data)
    task_to_groups = {}
    for (task, func_keys), samples in groups.items():
        task_to_groups.setdefault(task, []).append((func_keys, samples))

    for task, group_list in task_to_groups.items():
        print(f"Task: {task}")
        for func_keys, samples in group_list:
            print(f" Number of functions: {len(func_keys)} | Samples: {len(samples)}")

    hf_data = format_hf_data(groups)
    print('=' * 50)
    print(f"We have {len(hf_data)} after grouping")
    with open(args.save_file.replace(".json", "_grouped.json"), "w") as f:
        json.dump(hf_data, f, indent=4)

    clean_hf_data = format_clean_hf_data(data)
    print('=' * 50)
    print(f"We have {len(clean_hf_data)} after cleaning")
    with open(args.save_file.replace(".json", "_clean.json"), "w") as f:
        json.dump(clean_hf_data, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
