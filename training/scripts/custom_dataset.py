"""Custom evaluation tasks for LightEval."""
import logging
import random
import re
from datetime import datetime
from typing import Callable, Union

import lighteval.tasks.default_prompts as prompt
import litellm
import numpy as np
from latex2sympy2_extended import NormalizationConfig
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    compare_gold_target,
    extract_target_from_pred,
    get_extraction_regexes,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import PassAtK
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from math_verify import LatexExtractionConfig as verify_LatexExtractionConfig
from math_verify import parse, verify

from verl.utils.reward_score.ptp import is_boxed

logger = logging.getLogger(__name__)

LETTER_INDICES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
    "X", "Y", "Z"
]
INTEGER_INDICES = list(map(str, list(range(1, 27))))

DATE_TIME_EVAL_PROMPT = """You are given a model response and a ground truth answer concerning date values.
Your goal is to determine if the model response matches the ground truth answer.
Note that the model response may contain extra information or be expressed in a different format, but as long as it conveys the same date information, it is considered correct.

Provide a brief explanation of your reasoning process.
Then, on a new line, output only a single word: "Correct" or "Wrong" (no additional text or punctuation).

Model response: {answer}
Ground truth answer: {ground_truth}"""


##### Helper functions
class SelfConsistency:

    def __init__(
        self,
        n: int = None,
        normalize_gold: Callable = None,
        normalize_pred: Callable = None,
        strip_strings: bool = False,
        sample_scoring_function: Union[Callable[[str, str], float], str] = None,
    ):
        """Computing self-consistency

        Args:
            n (int): Number of samples to generate
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
            strip_strings (bool, optional): Whether to strip both reference and predictions. Defaults to False.
            sample_scoring_function (callable or str, optional): Function to use to score each sample.
                Either pass the full function (should take a string prediction and a string gold, and return a score between 0 and 1)
                a string (any of `prefix`, `suffix` or `full`) to define the type of exact match that you want, or nothing to defaults to "full".
                    `prefix` checks if the prediction starts with the gold,
                    `suffix` if the prediction ends with the gold,
                    `full` if the prediction and gold are equal
        """
        self.n = n
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
        self.strip_strings = strip_strings

        # Managed the logic of the per prediction of sample scoring
        if callable(sample_scoring_function):
            self.score_sample = sample_scoring_function
            self.type_exact_match = None
        else:
            if isinstance(sample_scoring_function, str):
                if sample_scoring_function not in ["prefix", "suffix", "full"]:
                    raise ValueError(
                        f"type_exact_match (used in parametrized_exact_match) must be one of prefix, suffix, or full. Was {sample_scoring_function} instead."
                    )
                self.type_exact_match = sample_scoring_function
            else:
                self.type_exact_match = "full"
            self.score_sample = self.default_sample_scoring

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> dict[str, float]:
        if len(golds) > 1:
            raise Exception("Cannot compute self-consistency with several golds")

        if self.n is None:
            self.n = len(predictions)
            logger.warning(
                "n undefined in the self-consistency. We assume it's the same as the sample's number of predictions.")
        elif len(predictions) < self.n:
            logger.warning(f"Number of predictions is less than {self.n} for self-consistency.")

        gold = self.get_processed_gold(golds[0])

        all_scores = []
        for pred in predictions[:self.n]:
            cur_pred = self.get_processed_pred(pred=pred)
            all_scores.append(self.score_sample(cur_pred, gold))

        return self.self_consistency(all_scores)

    def get_processed_gold(self, gold: str) -> float:
        if self.strip_strings:
            gold = gold.strip()

        if self.normalize_gold:
            gold = self.normalize_gold(gold)

        return gold

    def get_processed_pred(self, pred: str) -> float:
        if not pred:
            return ""

        if self.strip_strings:
            pred = pred.strip()

        if self.normalize_pred:
            pred = self.normalize_pred(pred)

        return pred

    def default_sample_scoring(self, pred: str, gold: str) -> int:
        if self.type_exact_match == "prefix":
            return 1 if pred.startswith(gold) else 0
        if self.type_exact_match == "suffix":
            return 1 if pred.endswith(gold) else 0
        return 1 if gold == pred else 0

    def self_consistency(self, all_scores: list[int]) -> float:
        if not all_scores:
            return 0.0
        count_correct = all_scores.count(1)
        return 1.0 if count_correct > len(all_scores) / 2 else 0.0


def llm_as_a_judge(model, prompt):
    response = litellm.completion(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=512,
        temperature=0,
        timeout=30,
    )
    if response is None:
        return False
    output = response.choices[0].message.content.strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return False
    final_judgment = lines[-1].lower()
    if final_judgment == "correct":
        return True
    elif final_judgment == "wrong":
        return False
    else:
        return "correct" in final_judgment


def convert_sympy_to_float(expr):
    """
    Convert a Sympy expression to a standard Python float.
    This function uses evalf() to numerically evaluate the expression.
    """
    if isinstance(expr, list):
        expr = expr[0]
    try:
        # Evaluate numerically and cast to float.
        return float(expr.evalf())
    except Exception as e:
        print("Error converting sympy expression to float:", e)
        raise e


def extract_from_boxed(text: str) -> str:
    match = re.search(r"\$?\\boxed{(.*?)}\$?", text)
    if match:
        return match.group(1)
    return text


def ptp_decorator(metric_fn, check_boxed=True, include_formatted_doc=True):

    def ptp_normalized(golds, predictions, formatted_doc):

        def extract_answer(text: str) -> str:
            match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                if check_boxed and not is_boxed(answer):
                    answer = f"$\\boxed{{{answer}}}$"
                if not check_boxed and is_boxed(answer):
                    answer = extract_from_boxed(answer)
            else:
                answer = text.strip()
            # normalize by latex
            # convert \sqrt(a) to \sqrt{a}
            answer = re.sub(r'\\sqrt\((.*?)\)', r'\\sqrt{\1}', answer)
            return answer

        processed_predictions = [extract_answer(pred) for pred in predictions]
        if include_formatted_doc:
            return metric_fn(golds, processed_predictions, formatted_doc)
        else:
            return metric_fn(golds, processed_predictions)

    return ptp_normalized


def wrap_metric_with_ptp(metric: SampleLevelMetric, check_boxed=True, include_formatted_doc=True) -> SampleLevelMetric:
    return SampleLevelMetric(
        metric_name=f'ptp_{metric.metric_name}',
        sample_level_fn=ptp_decorator(metric.sample_level_fn, check_boxed, include_formatted_doc),
        category=metric.category,
        use_case=metric.use_case,
        corpus_level_fn=metric.corpus_level_fn,
        higher_is_better=metric.higher_is_better,
    )


##### METRICS
self_consistency_16n = SampleLevelMetric(
    metric_name="self_consistency@16_samples",
    sample_level_fn=SelfConsistency(
        n=16,
        strip_strings=True,
        # Extracting mathematical expressions and latex expressions
        normalize_gold=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
                language=Language.ENGLISH,
            ),
        ),
        # Extracting mathematical expressions and latex expressions
        normalize_pred=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
                language=Language.ENGLISH,
            ),
        ),
        # Uses sympy for comparision
        sample_scoring_function=compare_gold_target,
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

math_pass_at_1_32n = SampleLevelMetric(
    metric_name="math_pass@1:32_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=32,
        strip_strings=True,
        # Extracting mathematical expressions and latex expressions
        normalize_gold=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
                language=Language.ENGLISH,
            ),
        ),
        # Extracting mathematical expressions and latex expressions
        normalize_pred=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
                language=Language.ENGLISH,
            ),
        ),
        # Uses sympy for comparision
        sample_scoring_function=compare_gold_target,
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)
latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)
gpqa_instruct_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=6,
)
expr_gold_metric = wrap_metric_with_ptp(expr_gold_metric, check_boxed=True)
latex_gold_metric = wrap_metric_with_ptp(latex_gold_metric, check_boxed=True)
gpqa_instruct_metric = wrap_metric_with_ptp(gpqa_instruct_metric, check_boxed=False)
math_pass_at_1_32n = wrap_metric_with_ptp(math_pass_at_1_32n, check_boxed=True, include_formatted_doc=False)
math_consistency_16n = wrap_metric_with_ptp(self_consistency_16n, check_boxed=True, include_formatted_doc=False)


# reference: https://github.com/ncbi-nlp/MedCalc-Bench/blob/main/evaluation/evaluate.py
# modified to use llm as a judge for date time and more robust math comparison
# TODO: use llm as a judge here
def medcalc_bench_eval(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    calid = formatted_doc.specific['calid']
    # output_type = formatted_doc.specific['output_type']
    lower_limit = formatted_doc.specific['lower_limit']
    upper_limit = formatted_doc.specific['upper_limit']
    answer = predictions[0]
    ground_truth = formatted_doc.choices[formatted_doc.gold_index]
    calid = int(calid)
    answer_parsed = None
    gold_parsed = None

    match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = answer.strip()

    try:
        if calid in [13, 68]:
            answer = extract_from_boxed(answer)
            match = re.search(r"^(0?[1-9]|1[0-2])\/(0?[1-9]|[12][0-9]|3[01])\/(\d{4})", answer)
            if match:
                month = int(match.group(1))
                day = int(match.group(2))
                year = match.group(3)
                answer = f"{month:02}/{day:02}/{year}"
            else:
                answer = "N/A"

            if datetime.strptime(answer, "%m/%d/%Y").strftime("%-m/%-d/%Y") == datetime.strptime(
                    ground_truth, "%m/%d/%Y").strftime("%-m/%-d/%Y"):
                correctness = 1
            else:
                correctness = 0
            # correctness = float(
            #     llm_as_a_judge("anthropic/claude-3-5-haiku-20241022",
            #                    DATE_TIME_EVAL_PROMPT.format(answer=answer, ground_truth=ground_truth)))

        elif calid in [69]:
            answer = extract_from_boxed(answer)
            # Output Type: integer (A, B)
            match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?",
                              ground_truth)
            ground_truth = f"({match.group(1)}, {match.group(3)})"
            answer = answer.replace("[", "(").replace("]", ")").replace("'", "").replace('"', "")
            match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", answer)
            if match:
                weeks = match.group(1)
                days = match.group(3)
                answer = f"({weeks}, {days})"
                if eval(answer) == eval(ground_truth):
                    correctness = 1
                else:
                    correctness = 0
            else:
                correctness = 0
            # correctness = float(
            #     llm_as_a_judge("anthropic/claude-3-5-haiku-20241022",
            #                    DATE_TIME_EVAL_PROMPT.format(answer=answer, ground_truth=ground_truth)))

        elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:
            # Output Type: integer A
            # answer = round(eval(answer))
            # if answer == eval(ground_truth):
            #     correctness = 1
            # else:
            #     correctness = 0
            gold_parsed = parse(
                ground_truth,
                extraction_mode='first_match',
            )
            if not is_boxed(answer):
                answer = f"$\\boxed{{{answer}}}$"
            answer_parsed = parse(
                answer,
                extraction_config=[
                    verify_LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            correctness = float(verify(gold_parsed, answer_parsed))
        elif calid in [
                2, 3, 5, 6, 7, 8, 9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61,
                62, 63, 64, 65, 66, 67
        ]:
            # answer = eval(answer)
            # if answer >= eval(lower_limit) and answer <= eval(upper_limit):
            #     correctness = 1
            # else:
            #     correctness = 0
            gold_parsed = parse(
                ground_truth,
                extraction_mode='first_match',
            )
            gold_parsed = convert_sympy_to_float(gold_parsed)
            if not is_boxed(answer):
                answer = f"$\\boxed{{{answer}}}$"
            answer_parsed = parse(
                answer,
                extraction_config=[
                    verify_LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            answer_parsed = convert_sympy_to_float(answer_parsed)
            if answer_parsed >= eval(lower_limit) and answer_parsed <= eval(upper_limit):
                correctness = 1
            else:
                correctness = 0
        else:
            raise ValueError(f"Unknown calculator ID: {calid}")
    except Exception as e:
        print(
            f"Error in calculator ID {calid}: {e}, answer: {answer}, ground_truth: {ground_truth}, answer_parsed: {answer_parsed}, gold_parsed: {gold_parsed}"
        )
        correctness = 0
    return correctness


medcalc_bench_metric = SampleLevelMetric(
    metric_name="medcalc_bench_metric",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=medcalc_bench_eval,
    corpus_level_fn=np.mean,
)


def mmlu_pro_eval(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    ground_truth = formatted_doc.choices[formatted_doc.gold_index]
    answer = predictions[0]
    match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = answer.strip()
    answer = extract_from_boxed(answer)

    # reference: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py
    def extract_answer(text):
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            # print("1st answer extract failed\n" + text)
            return extract_again(text)

    def extract_again(text):
        match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
        if match:
            return match.group(1)
        else:
            return extract_final(text)

    def extract_final(text):
        pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return None

    pred = extract_answer(answer)
    if pred is None:
        print(f"Could not extract answer from: {answer}")
        pred = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    if pred.lower() == ground_truth.lower():
        correctness = 1
    else:
        correctness = 0
    return correctness


mmlu_pro_metric = SampleLevelMetric(
    metric_name="mmlu_pro_metric",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=mmlu_pro_eval,
    corpus_level_fn=np.mean,
)


def medqa_eval(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    ground_truth = formatted_doc.choices[formatted_doc.gold_index]
    answer = predictions[0]
    match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = answer.strip()
    answer = extract_from_boxed(answer)

    # reference: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py
    def extract_answer(text):
        pattern = r"answer is \(?([A-D])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            # print("1st answer extract failed\n" + text)
            return extract_again(text)

    def extract_again(text):
        match = re.search(r'.*[aA]nswer:\s*([A-D])', text)
        if match:
            return match.group(1)
        else:
            return extract_final(text)

    def extract_final(text):
        pattern = r"\b[A-J]\b(?!.*\b[A-D]\b)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return None

    pred = extract_answer(answer)
    if pred is None:
        print(f"Could not extract answer from: {answer}")
        pred = random.choice(["A", "B", "C", "D"])
    if pred.lower() == ground_truth.lower():
        correctness = 1
    else:
        correctness = 0
    return correctness


medqa_metric = SampleLevelMetric(
    metric_name="medqa_metric",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=medqa_eval,
    corpus_level_fn=np.mean,
)


def pubmedqa_eval(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    ground_truth = formatted_doc.choices[formatted_doc.gold_index]
    answer = predictions[0]
    match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = answer.strip()
    answer = extract_from_boxed(answer)

    # reference: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py
    def extract_answer(text):
        pattern = r"answer is \(?([A-C])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            # print("1st answer extract failed\n" + text)
            return extract_again(text)

    def extract_again(text):
        match = re.search(r'.*[aA]nswer:\s*([A-C])', text)
        if match:
            return match.group(1)
        else:
            return extract_final(text)

    def extract_final(text):
        pattern = r"\b[A-J]\b(?!.*\b[A-C]\b)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return None

    pred = extract_answer(answer)
    if pred is None:
        print(f"Could not extract answer from: {answer}")
        reverse_options = {"yes": 'A', "no": 'B', "maybe": 'C'}
        pred = reverse_options.get(answer.lower(), None)
        if pred is None:
            pred = random.choice(["A", "B", "C"])
    if pred.lower() == ground_truth.lower():
        correctness = 1
    else:
        correctness = 0
    return correctness


pubmedqa_metric = SampleLevelMetric(
    metric_name="pubmedqa_metric",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=pubmedqa_eval,
    corpus_level_fn=np.mean,
)

##### PROMPTS
# original reference: https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/default_prompts.py
# but here we use a general prompt template for all math problems
def math_prompt_fn(line, task_name: str = None):
    solution = line.get("solution", None)
    if solution is None:
        solution = line.get("answer", None)

    problem = line.get("problem", None)
    if problem is None:
        problem = line.get("question", None)
    return Doc(
        task_name=task_name,
        query=problem,
        gold_index=0,
        choices=[solution],
    )


def medcalc_bench_prompt_fn(line, task_name: str = None):
    #     MEDCAL_ZERO_SHOT_PROMPT = """Here is the patient node:
    # {note}

    # Here is the task:
    # {question}

    # Please think step-by-step to solve the question and then generate the required score.
    # You should output the final answer in the format of 'The answer is: $\\boxed{{ANSWER}}$' (without quotes) where ANSWER is the short and direct answer of the question.
    # """.strip()
    return Doc(
        task_name=task_name,
        # query=MEDCAL_ZERO_SHOT_PROMPT.format(
        #     note=line["Patient Note"],
        #     question=line["Question"],
        # ),
        # query=MEDCAL_ZERO_SHOT_PROMPT.format(
        #     question=line["input"],
        # ),
        query=line['input'],
        choices=[line["corrected_answer"]] if line["corrected_answer"] else [line['Ground Truth Answer']],
        gold_index=0,
        specific={
            'calid': line["Calculator ID"],
            'output_type': line["Output Type"],
            'relevant_entities': line["Relevant Entities"],
            'lower_limit': line["Lower Limit"],
            'upper_limit': line["Upper Limit"],
        })


def mmlu_pro_prompt_fn(line, task_name: str = None):
    topic = line['category']
    MMLU_PROMPT = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}. Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"
    query = MMLU_PROMPT + 'Question: ' + line['question'] + '\nOptions: '
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["options"])])
    # query += "Answer:"
    gold_ix = line['answer_index']
    return Doc(
        task_name=task_name,
        query=query.strip(),
        choices=[f"{key}" for key, _ in zip(LETTER_INDICES, line["options"])],
        gold_index=gold_ix,
    )


def medqa_prompt_fn(line, task_name: str = None):
    MEDQA_PROMPT = "The following are multiple choice questions (with answers). Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"
    query = MEDQA_PROMPT + line['question'] + '\n'
    query += "".join([f"{key}. {choice}\n" for key, choice in line['options'].items()])
    gold_idx = LETTER_INDICES.index(line["answer_idx"])
    return Doc(
        task_name=task_name,
        query=query.strip(),
        choices=["A", "B", "C", "D"],
        gold_index=gold_idx,
    )


# converted to multiple-choice format
# reference:  https://raw.githubusercontent.com/FreedomIntelligence/HuatuoGPT-o1/refs/heads/main/evaluation/data/eval_data.json
def pubmedqa_prompt_fn(line, task_name: str = None):
    PUBMEDQA_PROMPT = "The following are multiple choice questions (with answers). Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"
    query = PUBMEDQA_PROMPT + line['question'] + '\n'
    query += "".join([f"{key}. {choice}\n" for key, choice in line['options'].items()])
    gold_idx = LETTER_INDICES.index(line["answer_idx"])
    return Doc(
        task_name=task_name,
        query=query.strip(),
        choices=["A", "B", "C"],
        gold_index=gold_idx,
    )


##### TASKS
gsm8k = LightevalTaskConfig(
    name="gsm8k",
    suite=["custom"],
    prompt_function=math_prompt_fn,
    hf_repo="gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=32768,
    metric=[
        expr_gold_metric,
        Metrics.quasi_exact_match_gsm8k,
    ],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=32768,
    metric=[
        Metrics.expr_gold_metric,
        expr_gold_metric,
        math_pass_at_1_32n,
    ],
    version=1,
)
aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["custom"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=32768,
    metric=[
        Metrics.expr_gold_metric,
        expr_gold_metric,
        math_pass_at_1_32n,
    ],
    version=1,
)
math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=math_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[Metrics.latex_gold_metric, latex_gold_metric],
    version=1,
)
# this already included subdomains of genetics and molecular biology
gpqa_diamond = LightevalTaskConfig(
    name="gpqa_diamond",
    suite=["custom"],
    prompt_function=prompt.gpqa_instruct,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metric=[Metrics.gpqa_instruct_metric, gpqa_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=0,
)
medcalc_bench = LightevalTaskConfig(
    name="medcalc_bench",
    suite=["custom"],
    prompt_function=medcalc_bench_prompt_fn,
    hf_repo="PTPReasoning/MedCalc-Bench-v1.0",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[medcalc_bench_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
medcalc_bench_rules = LightevalTaskConfig(
    name="medcalc_bench_rules",
    suite=["custom"],
    prompt_function=medcalc_bench_prompt_fn,
    hf_repo="PTPReasoning/MedCalc-Bench-v1.0",
    hf_subset="default",
    hf_avail_splits=["test_rules"],
    evaluation_splits=["test_rules"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[medcalc_bench_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
medcalc_bench_formulas = LightevalTaskConfig(
    name="medcalc_bench_formulas",
    suite=["custom"],
    prompt_function=medcalc_bench_prompt_fn,
    hf_repo="PTPReasoning/MedCalc-Bench-v1.0",
    hf_subset="default",
    hf_avail_splits=["test_formulas"],
    evaluation_splits=["test_formulas"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[medcalc_bench_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
medcalc_bench_rules_train = LightevalTaskConfig(
    name="medcalc_bench_rules_train",
    suite=["custom"],
    prompt_function=medcalc_bench_prompt_fn,
    hf_repo="PTPReasoning/MedCalc-Bench-v1.0",
    hf_subset="default",
    hf_avail_splits=["train_rules"],
    evaluation_splits=["train_rules"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[medcalc_bench_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
medcalc_bench_formulas_train = LightevalTaskConfig(
    name="medcalc_bench_formulas_train",
    suite=["custom"],
    prompt_function=medcalc_bench_prompt_fn,
    hf_repo="PTPReasoning/MedCalc-Bench-v1.0",
    hf_subset="default",
    hf_avail_splits=["train_formulas"],
    evaluation_splits=["train_formulas"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[medcalc_bench_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
medcalc_bench_rules_dedup = LightevalTaskConfig(
    name="medcalc_bench_rules_dedup",
    suite=["custom"],
    prompt_function=medcalc_bench_prompt_fn,
    hf_repo="PTPReasoning/MedCalc-Bench-v1.0",
    hf_subset="default",
    hf_avail_splits=["test_rules_dedup"],
    evaluation_splits=["test_rules_dedup"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[medcalc_bench_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
medcalc_bench_formulas_dedup = LightevalTaskConfig(
    name="medcalc_bench_formulas_dedup",
    suite=["custom"],
    prompt_function=medcalc_bench_prompt_fn,
    hf_repo="PTPReasoning/MedCalc-Bench-v1.0",
    hf_subset="default",
    hf_avail_splits=["test_formulas_dedup"],
    evaluation_splits=["test_formulas_dedup"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[medcalc_bench_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
# reference performance: https://www.a1-labs.co/assets/mmlupro/
# we can get similar performance
mmlu_pro_health = LightevalTaskConfig(
    name="mmlu_pro_health",
    suite=["custom"],
    prompt_function=mmlu_pro_prompt_fn,
    hf_repo="PTPReasoning/mmlu_pro_health",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[mmlu_pro_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
mmlu_pro_biology = LightevalTaskConfig(
    name="mmlu_pro_biology",
    suite=["custom"],
    prompt_function=mmlu_pro_prompt_fn,
    hf_repo="PTPReasoning/mmlu_pro_biology",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[mmlu_pro_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
# sanity check
# https://qwenlm.github.io/blog/qwen2.5-llm/
# qwen2.5-7b-instruct reported 56.3 which is close to what we got
mmlu_pro = LightevalTaskConfig(
    name="mmlu_pro",
    suite=["custom"],
    prompt_function=mmlu_pro_prompt_fn,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_avail_splits=["validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[mmlu_pro_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
medqa = LightevalTaskConfig(
    name="medqa",
    suite=["custom"],
    prompt_function=medqa_prompt_fn,
    hf_repo="GBaker/MedQA-USMLE-4-options",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[medqa_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
pubmedqa = LightevalTaskConfig(
    name="pubmedqa",
    suite=["custom"],
    prompt_function=pubmedqa_prompt_fn,
    hf_repo="PTPReasoning/PubMedQA",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[pubmedqa_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)
pubmedqa_dedup = LightevalTaskConfig(
    name="pubmedqa_dedup",
    suite=["custom"],
    prompt_function=pubmedqa_prompt_fn,
    hf_repo="PTPReasoning/PubMedQA",
    hf_subset="default",
    hf_avail_splits=["test_dedup"],
    evaluation_splits=["test_dedup"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[pubmedqa_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=0,
)

TASKS_TABLE = []
TASKS_TABLE.append(gsm8k)
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime25)
TASKS_TABLE.append(medcalc_bench)
TASKS_TABLE.append(medcalc_bench_rules)
TASKS_TABLE.append(medcalc_bench_formulas)
TASKS_TABLE.append(medcalc_bench_rules_train)
TASKS_TABLE.append(medcalc_bench_formulas_train)
TASKS_TABLE.append(medcalc_bench_rules_dedup)
TASKS_TABLE.append(medcalc_bench_formulas_dedup)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(gpqa_diamond)
TASKS_TABLE.append(mmlu_pro_health)
TASKS_TABLE.append(mmlu_pro_biology)
TASKS_TABLE.append(mmlu_pro)
TASKS_TABLE.append(medqa)
TASKS_TABLE.append(pubmedqa)
TASKS_TABLE.append(pubmedqa_dedup)

##### MODULE LOGIC
if __name__ == "__main__":
    print([t.name for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
