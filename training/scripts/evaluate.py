import argparse
import ast
import itertools
import random
import json
import logging
import os
import re
import types
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Optional

import colorlog
from latex2sympy2_extended import NormalizationConfig
from lighteval.logging.evaluation_tracker import (EnhancedJSONEncoder,
                                                  EvaluationTracker)
from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_loader import load_model
from lighteval.models.model_output import GenerativeResponse
from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
from lighteval.pipeline import (EnvConfig, ParallelismManager, Pipeline,
                                PipelineParameters)
from lighteval.utils.imports import is_accelerate_available, is_vllm_available
from tqdm import tqdm

try:
    from typicality import SkeletonLogProb, Hmm
except ImportError:
    print('To use auditors, please git-clone the docstring submodule and set up the pythonpath to include the submodule folder.')

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None
if is_vllm_available():
    import ray
    from more_itertools import distribute
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.tokenizer import get_tokenizer

    logging.getLogger("vllm").propagate = True
    logging.getLogger("vllm").handlers.clear()

    logging.getLogger("ray").propagate = True
    logging.getLogger("ray").handlers.clear()
else:
    LLM = None
    SamplingParams = None
    get_tokenizer = None
    ray = None
    distribute = None


class EquationDeprecationFilter(logging.Filter):

    def filter(self, record):
        # Return False to filter out the message, True to keep it
        return "equations is deprecated, as it handled by the parser now" not in record.getMessage()


handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter("%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                      datefmt=None,
                                      reset=True,
                                      log_colors={
                                          'DEBUG': 'cyan',
                                          'INFO': 'green',
                                          'WARNING': 'yellow',
                                          'ERROR': 'red',
                                          'CRITICAL': 'red,bg_white',
                                      },
                                      secondary_log_colors={},
                                      style='%')
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[handler])

for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)  # suppress all logs by default

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# https://github.com/huggingface/latex2sympy2_extended/blob/ef36bfe02d7c2ec62f9a3c558afb0f1dedff6f79/src/latex2sympy2_extended/math_normalization.py#L466C11-L466C12
# this is annoy, so we filter it out
math_normalization_logger = logging.getLogger('latex2sympy2_extended.math_normalization')
math_normalization_logger.addFilter(EquationDeprecationFilter())

TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR: str = "tmp/"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

PREFIX_1 = "Use the existing partial program without modification to answer the question."
PREFIX_2 = "Augment the current partial program to answer the question."


rules_weight = '/home/jixuanl/verl/auditors/hmm/save-medcalc_bench_rules-tune190-hmm.pkl'
formulas_weight = '/home/jixuanl/verl/auditors/hmm/save-medcalc_bench_formulas-tune329-hmm.pkl'
pubmed_weight = '/home/jixuanl/verl/auditors/hmm/save-pubmedqa-tune500-hmm.pkl'



RULES_MODEL = Hmm.load(rules_weight)
FORMULAS_MODEL = Hmm.load(formulas_weight)
PUBMED_MODEL = Hmm.load(pubmed_weight)


def evaluate_single_trace(trace, task_type=None):
    if task_type == "medcalc_bench_formulas_dedup":
        model = FORMULAS_MODEL
    elif task_type == "medcalc_bench_rules_dedup":
        model = RULES_MODEL
    elif task_type == "pubmedqa_dedup":
        model = PUBMED_MODEL
    else:
        raise ValueError(f"Unknown task type: {task_type}. Supported types are 'medcalc_bench_formulas_dedup', 'medcalc_bench_rules_dedup', and 'pubmedqa_dedup'.")
    if isinstance(trace, str):
        trace = trace.split('\n')
    quantile = model.quantile(trace, 3)
    return quantile


def extract_from_boxed(text: str) -> str:
    match = re.search(r"\$?\\boxed{(.*?)}\$?", text)
    if match:
        return match.group(1)
    return text


def extract_answer_from_response(response):
    answer_pattern = re.compile(r"<answer>\n(.*?)\n</answer>", re.DOTALL)
    answer_match = answer_pattern.search(response)
    if answer_match is None:
        return extract_from_boxed(response)
    answer = answer_match.group(1).strip()
    answer = extract_from_boxed(answer)
    return answer


def extract_partial_programs_from_response(response):
    pattern = r"<partial_program>\n(.*?)\n</partial_program>"
    match = re.search(pattern, response, re.DOTALL)
    partial_program_content = match.group(1) if match else ""
    base_partial_program = ""
    if partial_program_content:
        base_partial_program = f'\n\n<partial_program>\n{partial_program_content}\n</partial_program>'
    else:
        logger.warning(f"Partial program not found in response: {response}")

    return base_partial_program


# if base_partial_program is empty, we will use the original input ids
# not sure if this is the best way to do it,
# since pass the same input_ids again in batch, can result in different outputs
def _augment_prompt(original_input_ids, tokenizer, base_partial_program, assistant_start_token, prefix=PREFIX_2):
    if not base_partial_program:
        return original_input_ids
    decoded_input = tokenizer.decode(original_input_ids)
    parts = decoded_input.split(assistant_start_token)
    input_before_assistant = parts[0]
    pattern = r"\n\n<partial_program>\n(.*?)\n</partial_program>"
    if re.search(pattern, input_before_assistant, re.DOTALL):
        # replace existing partial program with the new one
        new_input = re.sub(pattern,
                           lambda m: f"\n\n<partial_program>\n{base_partial_program}\n</partial_program>",
                           input_before_assistant,
                           flags=re.DOTALL)
    else:
        # append the new one at the end of input_before_assistant
        new_input = f'{input_before_assistant}{base_partial_program}\n\n{prefix}'
    new_input += assistant_start_token
    input_ids = tokenizer(new_input, add_special_tokens=False)['input_ids']
    return input_ids


# we patchfify the vllm model generate function here to allow for in-task learning and test-time scaling
def _generate(
    self,
    inputs: list[list[int]],
    max_new_tokens: Optional[int] = None,
    stop_tokens: Optional[list[str]] = None,
    returns_logits: Optional[bool] = False,
    num_samples: int = 1,
    generate: bool = True,
) -> list[GenerativeResponse]:
    """Contains the actual logic of the generation."""
    sampling_params = SamplingParams(**self._config.generation_parameters.to_vllm_dict())
    
    # special parameters for our models
    in_task_learning_examples = getattr(self, "in_task_learning_examples", 0)
    in_task_learning_mode = getattr(self, "in_task_learning_mode", 0)
    test_time_scaling_repeats = getattr(self, "test_time_scaling_repeats", 0)
    # verifier mode?
    audit_self_consistency = getattr(self, "audit_self_consistency", 0)
    baseline_self_consistency = getattr(self, "baseline_self_consistency", 0)

    # audit mode
    task_type = getattr(self, "task_type", None)

    # this is not elegant, but lighteval process the chat template in prompt manager
    # it is quite hard to modify the prompt there, so we need to pass the assistant start token here
    # because we need to add user prompt
    assistant_start_token = getattr(self, "assistant_start_token", "<|im_end|>\n<|im_start|>assistant\n")

    if self.data_parallel_size > 1 and (in_task_learning_examples > 0 or test_time_scaling_repeats > 0 or audit_self_consistency > 0 or baseline_self_consistency > 0):
        raise ValueError(
            "Baseline or Verifier Best of N In-task learning or test-time scaling is not supported with multiple data parallel workers. Please set data_parallel_size to 1. Current value: {self.data_parallel_size}"
        )
    if num_samples > 1 and (in_task_learning_examples > 0 or test_time_scaling_repeats > 0 or audit_self_consistency > 0 or baseline_self_consistency > 0):
        raise ValueError(
            "Baseline or Verifier Best of N In-task learning or test-time scaling is not supported with multiple samples. Please set num_samples to 1. Current value: {num_samples}"
        )
    if not generate and (in_task_learning_examples > 0 or test_time_scaling_repeats > 0 or audit_self_consistency > 0 or baseline_self_consistency > 0):
        raise ValueError(
            "Baseline or Verifier Best of N or In-task learning or test-time scaling only supports generative tasks. Please set generate to True.")

    if audit_self_consistency > 0:
        num_samples = audit_self_consistency
    
    if baseline_self_consistency > 0:
        num_samples = baseline_self_consistency

    if generate:
        sampling_params.n = num_samples
        sampling_params.max_tokens = max_new_tokens
        sampling_params.stop = stop_tokens
        sampling_params.logprobs = 1 if returns_logits else 0

    else:
        sampling_params.temperature = 0
        sampling_params.prompt_logprobs = 1
        sampling_params.max_tokens = 1
        sampling_params.detokenize = False

    if self.data_parallel_size > 1:
        # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
        # also seems to only work with decorator and not with ray.remote() fn
        # see https://github.com/vllm-project/vllm/issues/973
        # note: this has changed on 0.3.3, and it only works now if num_gpus are set.
        # but then tensor_parallel breaks
        # Hynek: With the newest vllm, it actually breaks when tensor_parallel_size == 1 and num_gpus not set,
        # as VLLM complains about no GPUs available.
        @ray.remote(num_gpus=1 if self.tensor_parallel_size == 1 else None)
        def run_inference_one_model(model_args: dict, sampling_params: SamplingParams, requests):
            llm = LLM(**model_args)
            return llm.generate(prompt_token_ids=requests, sampling_params=sampling_params)

        # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
        # interleaved important to balance context lengths across workers
        requests = [list(x) for x in distribute(self.data_parallel_size, inputs)]
        inputs = ((self.model_args, sampling_params, req) for req in requests)
        object_refs = [run_inference_one_model.remote(*x) for x in inputs]
        results = ray.get(object_refs)
        # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
        ray.shutdown()
        # flatten results
        outputs = [
            x for x in itertools.chain.from_iterable(itertools.zip_longest(*[list(x) for x in results]))
            if x is not None
        ]

    else:
        # if we are using in task learning, we will use the first few samples sequentially
        # to get a base partial program, and then run the rest in parallel
        if in_task_learning_examples > 0:
            if in_task_learning_examples > len(inputs):
                logger.warning(
                    f"In-task learning examples {in_task_learning_examples} is greater than the number of inputs {len(inputs)}. Setting in-task learning examples to {len(inputs)}."
                )
                in_task_learning_examples = len(inputs)
            outputs = []
            itl_inputs = inputs[:in_task_learning_examples]
            base_partial_program = ""
            for itl_input in tqdm(itl_inputs, desc="In-task learning", total=in_task_learning_examples):
                # always use prefix 2 for here
                input_ids = _augment_prompt(itl_input, self.tokenizer, base_partial_program, assistant_start_token,
                                            PREFIX_2)
                vllm_outputs = self.model.generate(
                    prompt_token_ids=input_ids,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                outputs.extend(vllm_outputs)
                for vllm_output in vllm_outputs:
                    response = [output.text for output in vllm_output.outputs][0]
                    base_partial_program = extract_partial_programs_from_response(response)
            # run the rest of the inputs with the base partial program added to the prompt
            # for efficiency, we should modify the inputs, then run the rest in parallel
            rest_inputs = inputs[in_task_learning_examples:]
            if len(rest_inputs) == 0:
                return outputs
            rest_inputs_ids = []
            for rest_input in rest_inputs:
                prefix = PREFIX_1 if in_task_learning_mode == 0 else PREFIX_2
                input_ids = _augment_prompt(rest_input, self.tokenizer, base_partial_program, assistant_start_token,
                                            prefix)
                rest_inputs_ids.append(input_ids)
            # run the rest in parallel
            rest_outputs = self.model.generate(
                prompt_token_ids=rest_inputs_ids,
                sampling_params=sampling_params,
                use_tqdm=True,
            )
            outputs.extend(rest_outputs)

        # for test-time scaling, we will run each input multiple times (iteratively)
        # we should find a way to run the inputs in parallel
        elif test_time_scaling_repeats > 0:
            outputs = self.model.generate(
                prompt_token_ids=inputs,
                sampling_params=sampling_params,
                use_tqdm=True,
            )
            for _ in range(test_time_scaling_repeats):
                input_ids = []
                for vllm_output in outputs:
                    response = [output.text for output in vllm_output.outputs][0]
                    base_partial_program = extract_partial_programs_from_response(response)
                    # augment the input with the base partial program
                    # use vllm_output.prompt_token_ids to get the corresponding input ids
                    augment_input_id = _augment_prompt(vllm_output.prompt_token_ids, self.tokenizer,
                                                       base_partial_program, assistant_start_token, PREFIX_2)
                    input_ids.append(augment_input_id)
                outputs = self.model.generate(
                    prompt_token_ids=input_ids,
                    sampling_params=sampling_params,
                    use_tqdm=True,
                )

        elif baseline_self_consistency > 0:
            temp_outputs = self.model.generate(
                prompt_token_ids=inputs,
                sampling_params=sampling_params,
                use_tqdm=True,
            )
            outputs = []
            for vllm_output in temp_outputs:
                response = [output.text for output in vllm_output.outputs]
                response_answers = [extract_answer_from_response(r) for r in response]
                # do majority vote
                vote_counts = Counter(response_answers)
                majority_answer = vote_counts.most_common(1)[0][0]
                for i, answer in enumerate(response_answers):
                    if answer == majority_answer:
                        new_vllm_output = deepcopy(vllm_output)
                        # checked, this should be fine
                        new_vllm_output.outputs = [vllm_output.outputs[i]] 
                        outputs.append(new_vllm_output)
                        break

            assert len(outputs) == len(inputs), f"Outputs length {len(outputs)} does not match inputs length {len(inputs)}"

        elif audit_self_consistency:
            outputs = [None] * len(inputs)  
            resample_0_indices = []  
            resample_1_indices = []  
            resample_0_inputs = []   
            resample_1_inputs = []   
            total_possible_responses = audit_self_consistency * len(inputs)
            total_responses_generated = len(inputs)  
            # initial sampling
            sampling_params.n = 1
            temp_outputs = self.model.generate(
                prompt_token_ids=inputs,
                sampling_params=sampling_params,
                use_tqdm=True,
            )
            for i, vllm_output in enumerate(temp_outputs):
                response = [output.text for output in vllm_output.outputs][0]
                audit_result = evaluate_single_trace(response, task_type)
                if audit_result == 2:
                    # Result is valid, no resampling needed
                    outputs[i] = vllm_output
                elif audit_result == 1:
                    # Need to resample (audit_self_consistency-3) times
                    resample_1_indices.append(i)
                    resample_1_inputs.append(inputs[i])
                else:  # audit_result == 0
                    # Need to resample (audit_self_consistency-1) times
                    resample_0_indices.append(i)
                    resample_0_inputs.append(inputs[i])

            if resample_0_inputs:
                resampling_params = deepcopy(sampling_params)
                resample_0_count = max(1, audit_self_consistency - 1)
                resampling_params.n = resample_0_count

                total_responses_generated += len(resample_0_inputs) * resample_0_count

                resampled_0_outputs = self.model.generate(
                    prompt_token_ids=resample_0_inputs,
                    sampling_params=resampling_params,
                    use_tqdm=True,
                )
                for j, (original_idx, resampled_result) in enumerate(zip(resample_0_indices, resampled_0_outputs)):
                    # Get all responses (including original)
                    original_response = temp_outputs[original_idx].outputs[0].text
                    new_responses = [output.text for output in resampled_result.outputs]
                    all_responses = [original_response] + new_responses
                    
                    # Extract answers and use majority voting
                    all_answers = [extract_answer_from_response(r) for r in all_responses]
                    vote_counts = Counter(all_answers)
                    majority_answer = vote_counts.most_common(1)[0][0]
                    
                    # Find first response with majority answer
                    for idx, answer in enumerate(all_answers):
                        if answer == majority_answer:
                            if idx == 0:
                                # It's the original response
                                outputs[original_idx] = temp_outputs[original_idx]
                            else:
                                # It's one of the new responses
                                new_output = deepcopy(resampled_result)
                                new_output.outputs = [resampled_result.outputs[idx-1]]
                                outputs[original_idx] = new_output
                            break
                        
            if resample_1_inputs:
                resampling_params = deepcopy(sampling_params)
                resample_1_count = max(1, audit_self_consistency - 3)
                resampling_params.n = resample_1_count
                
                total_responses_generated += len(resample_1_inputs) * resample_1_count

            
                resampled_1_outputs = self.model.generate(
                    prompt_token_ids=resample_1_inputs,
                    sampling_params=resampling_params,
                    use_tqdm=True,
                )
                
                for j, (original_idx, resampled_result) in enumerate(zip(resample_1_indices, resampled_1_outputs)):
                    original_response = temp_outputs[original_idx].outputs[0].text
                    new_responses = [output.text for output in resampled_result.outputs]
                    all_responses = [original_response] + new_responses

                    all_answers = [extract_answer_from_response(r) for r in all_responses]
                    vote_counts = Counter(all_answers)
                    majority_answer = vote_counts.most_common(1)[0][0]
                    
                    for idx, answer in enumerate(all_answers):
                        if answer == majority_answer:
                            if idx == 0:
                                outputs[original_idx] = temp_outputs[original_idx]
                            else:
                                new_output = deepcopy(resampled_result)
                                new_output.outputs = [resampled_result.outputs[idx-1]]
                                outputs[original_idx] = new_output
                            break
            
            for i, output in enumerate(outputs):
                if output is None:
                    outputs[i] = temp_outputs[i]
            
            efficiency_percentage = (total_responses_generated / total_possible_responses) * 100
            with open("audit_self_consistency.txt", "a") as f:
                f.write(f"Audit_self_consistency: {audit_self_consistency}, Task: {task_type}, Total responses generated: {total_responses_generated}, Total possible responses: {total_possible_responses}, Efficiency: {efficiency_percentage:.2f}%\n")

            print(f"Generated {total_responses_generated} out of {total_possible_responses} possible responses ({efficiency_percentage:.2f}%)")

            assert None not in outputs, "Some prompts could not be processed"
            assert len(outputs) == len(inputs), f"Outputs length {len(outputs)} does not match inputs length {len(inputs)}"

        else:
            outputs = self.model.generate(
                prompt_token_ids=inputs,
                sampling_params=sampling_params,
                use_tqdm=True,
            )

    return outputs


# TODO: not elegant
VLLMModel._generate = _generate
VLLMModel.final_cleanup = VLLMModel.cleanup
VLLMModel.cleanup = lambda self: None


# monkey patching to save details in json format
# https://github.com/huggingface/lighteval/blob/main/src/lighteval/logging/info_loggers.py#L159
# https://github.com/huggingface/lighteval/blob/main/src/lighteval/logging/evaluation_tracker.py#L227
def save_json_details(self, date_id: str, details_datasets):
    # output_dir_details_sub_folder = self._get_details_sub_folder(date_id)
    output_dir_details_sub_folder = Path(self.output_dir) / "details"  # / self.general_config_logger.model_name
    self.fs.mkdirs(output_dir_details_sub_folder, exist_ok=True)
    logger.info(f"Saving details to {output_dir_details_sub_folder}")

    for task_name, dataset in details_datasets.items():
        formatted_data = []
        for detail in dataset:
            try:
                predictions = ast.literal_eval(detail['predictions'])
            except ValueError:
                predictions = [[[detail['predictions']]]]
            full_prompt = detail['full_prompt']
            target = detail['gold']
            actual_input = self.tokenizer.decode(detail['input_tokens'][0])
            formatted_data.append({
                "example": detail['example'],
                "instruction": detail['instruction'],
                "base_prompt": full_prompt,
                "input": actual_input,
                "predictions": predictions[0][0][0],
                "target": target,
                "num_effective_few_shots": detail['num_effective_few_shots'],
                "specifics": detail['specifics'],
            })
        json_task_name = task_name.replace("|", "_")
        save_file_name = getattr(self, "save_file_name", f"details_{json_task_name}_{date_id}")
        save_file_name = f"{save_file_name}.json"

        output_file_details = output_dir_details_sub_folder / save_file_name
        with self.fs.open(str(output_file_details), "w") as f:
            json.dump(formatted_data, f, indent=4)
        # uncomment for default parquet format
        # output_file_details = output_dir_details_sub_folder / f"details_{task_name}_{date_id}.parquet"
        # with self.fs.open(str(output_file_details), "wb") as f:
        #     dataset.to_parquet(f)


# patchify to save the results file with different names
def save_results(self, date_id: str, results_dict: dict):
    output_dir_results = Path(self.output_dir) / "results"  # / self.general_config_logger.model_name
    self.fs.mkdirs(output_dir_results, exist_ok=True)

    save_file_name = getattr(self, "save_file_name", f"results_{date_id}")
    save_file_name = f"{save_file_name}.json"

    # metrics_value (dict[str, dict[str, list[float]]]): Maps each task to its dictionary of metrics to scores for all the example of the task.
    metrics_values = self.metrics_logger.metrics_values
    results_dict['metrics_values'] = metrics_values

    output_results_file = output_dir_results / save_file_name
    logger.info(f"Saving results to {output_results_file}")
    with self.fs.open(output_results_file, "w") as f:
        f.write(json.dumps(results_dict, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/user_data/jixuanl/PTP/Qwen2.5-7B-Instruct-PTP-SFT-v1-trl")
    parser.add_argument("--suite", type=str, default="custom")
    parser.add_argument("--task",
                        type=str,
                        nargs="+",
                        default=["gsm8k"],
                        help="Specify one or more tasks (e.g., --task gsm8k mathqa).")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--job_id", type=int, default=0, help="Optional job id for future reference.")
    parser.add_argument("--dataset_loading_processes",
                        type=int,
                        default=1,
                        help="Number of processes to use for dataset loading.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    parser.add_argument("--num_fewshot_seeds",
                        type=int,
                        default=1,
                        help="Number of seeds to use for few-shot evaluation.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate on.")
    parser.add_argument("--use_chat_template", action="store_true", help="Use chat template for generation.")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to use.")
    parser.add_argument("--load_responses_from_details_date_id",
                        type=str,
                        default=None,
                        help="Load responses from details directory.")
    parser.add_argument('--few_shot', type=int, default=0)
    # geneartion config
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=float, default=20)
    # output folder
    parser.add_argument("--output_dir", type=str, default="evals", help="Directory to save evaluation results.")

    # in-task learning
    parser.add_argument("--in_task_learning_examples",
                        type=int,
                        default=0,
                        help="Number of examples to use for in-task learning.")
    parser.add_argument(
        "--in_task_learning_mode",
        type=int,
        default=0,
        help="Mode for in-task learning. 0: use existing partial program, 2: augment the current partial program.")
    # test-time scaling
    parser.add_argument("--test_time_scaling_repeats",
                        type=int,
                        default=0,
                        help="Number of repeats for test-time scaling.")
    # best of n
    parser.add_argument("--audit_self_consistency",
                        type=int,
                        default=0,
                        help="Use self-consistency for generation.")
    parser.add_argument("--baseline_self_consistency",
                        type=int,
                        default=0,
                        help="Use baseline self-consistency for generation.")
    
    return parser.parse_args()


def main(args):
    env_config = EnvConfig(token=TOKEN, cache_dir=CACHE_DIR)

    if len(args.task) > 1:
        logger.warning("LightEval supports run all tasks togther, but we will run them sequentially.")
    if args.in_task_learning_examples > 0 and args.test_time_scaling_repeats > 0:
        raise ValueError(
            "In-task learning and test-time scaling are mutually exclusive for NOW. Using in-task learning only.")

    if "DeepSeek-R1-Distill" in args.model or "Bio-Medical-Llama-3-2-1B-CoT-012025" in args.model or 'OpenR1-Qwen-7B' in args.model:
        logger.warning(f"Using {args.model}, setting temperature to 0.6 and top_p to 0.95.")
        logger.warning("reference: https://arxiv.org/pdf/2501.12948")
        args.temperature = 0.6
        args.top_p = 0.95
        args.top_k = -1

    if args.model == 'Qwen/Qwen2.5-7B':
        logger.warning("Not using chat template for Qwen2.5-7B model following https://arxiv.org/pdf/2503.20783")
        args.use_chat_template = False

    if args.model in ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen-2.5-7B', 'ContactDoctor/Bio-Medical-Llama-3-8B', 
                      'meta-llama/Llama-3.1-8B-Instruct', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 
                      'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025', 
                      'open-r1/OpenR1-Qwen-7B']:
        logger.warning(f"Running evaluation with system prompt for {args.model}")
        args.system_prompt = SYSTEM_PROMPT

    generation_parameters = GenerationParameters(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    model_config = VLLMModelConfig(
        pretrained=args.model,
        dtype="bfloat16",
        use_chat_template=args.use_chat_template,
        max_model_length=args.max_new_tokens,
        max_num_batched_tokens=args.max_new_tokens,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        generation_parameters=generation_parameters,
        trust_remote_code=True,
        seed=args.seed,
    )
    model = load_model(model_config, env_config=env_config)

    system_prompt = None
    for task in args.task:
        if args.system_prompt is not None:
            if 'gpqa_diamond' in task or 'aime' in task or 'mmlu' in task or 'medqa' in task or 'pubmedqa' in task:
                logger.warning(f"System prompt temporarily disabled for {task}.")
                system_prompt = None
            else:
                system_prompt = args.system_prompt
                logger.warning(f"Using system prompt: {args.system_prompt}")

        if 'aime' in task and args.temperature < 1e-5:
            logger.warning("Temperature temporarily set to 0.7 for AIME because of Pass@K.")
            model._config.generation_parameters.temperature = 0.7

        elif (args.audit_self_consistency > 0 or args.baseline_self_consistency > 0) and args.temperature < 1e-5:
            logger.warning("Temperature temporarily set to 0.7 for audit best of n or baseline majority vote or audit mode.")
            logger.warning("Please be aware that these test-time-scaling only works for SSRT models!")
            model._config.generation_parameters.temperature = 1.0

        else:
            model._config.generation_parameters.temperature = args.temperature

        if model._config.generation_parameters.temperature < 1e-5:
            logger.warning("Temperature is set to 0, using greedy decoding.")
            args.top_p = 1.0
            args.top_k = -1

        evaluation_tracker = EvaluationTracker(
            output_dir=args.output_dir,
            save_details=True,
            push_to_hub=False,
            hub_results_org="lighteval",
        )

        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.VLLM,
            env_config=env_config,
            job_id=args.job_id,
            dataset_loading_processes=args.dataset_loading_processes,
            custom_tasks_directory="scripts/custom_dataset.py" ,
            override_batch_size=-1,  # Cannot override batch size when using VLLM
            num_fewshot_seeds=args.num_fewshot_seeds,
            max_samples=args.max_samples,
            use_chat_template=args.use_chat_template,
            system_prompt=system_prompt,
            load_responses_from_details_date_id=args.load_responses_from_details_date_id,
        )

        pipe_task = f"{args.suite}|{task}|{args.few_shot}|0"
        pipeline = Pipeline(
            tasks=pipe_task,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            # model_config=model_config,
            model=model,
            metric_options={},
        )
        pipeline.evaluation_tracker.save_details = types.MethodType(save_json_details, pipeline.evaluation_tracker)
        pipeline.evaluation_tracker.tokenizer = pipeline.model.tokenizer
        pipeline.evaluation_tracker.save_results = types.MethodType(save_results, pipeline.evaluation_tracker)

        pipeline.evaluation_tracker.save_file_name = f'{args.suite}_{task}_{args.few_shot}_{model._config.generation_parameters.temperature}_{args.in_task_learning_examples}_{args.in_task_learning_mode}_{args.test_time_scaling_repeats}'

        logger.info(f'Arguments: {vars(args)}')

        # in-task learning
        pipeline.model.in_task_learning_examples = args.in_task_learning_examples
        pipeline.model.in_task_learning_mode = args.in_task_learning_mode
        # test-time scaling
        pipeline.model.test_time_scaling_repeats = args.test_time_scaling_repeats
        pipeline.model.assistant_start_token = "<|im_end|>\n<|im_start|>assistant\n"
        # verifier best of n
        pipeline.model.audit_self_consistency = args.audit_self_consistency
        # baseline best of n
        pipeline.model.baseline_self_consistency = args.baseline_self_consistency
        # audit mode
        pipeline.model.task_type = task

        pipeline.evaluate()
        pipeline.save_and_push_results()
        pipeline.show_results()

    # we do manual cleanup here
    pipeline.model.final_cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
