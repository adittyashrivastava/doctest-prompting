import datetime
import collections
import logging
import re

from latex2sympy2_extended import NormalizationConfig
from sympy.printing import sstr


class EquationDeprecationFilter(logging.Filter):

    def filter(self, record):
        # Return False to filter out the message, True to keep it
        return "equations is deprecated, as it handled by the parser now" not in record.getMessage()


# https://github.com/huggingface/latex2sympy2_extended/blob/ef36bfe02d7c2ec62f9a3c558afb0f1dedff6f79/src/latex2sympy2_extended/math_normalization.py#L466C11-L466C12
# this is spammy, so we filter it out
math_normalization_logger = logging.getLogger('latex2sympy2_extended.math_normalization')
math_normalization_logger.addFilter(EquationDeprecationFilter())

try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

StepOutput = collections.namedtuple(
    'StepOutput',
    [
        'value',  # return value as a string
        'fn_name',  # function name
        'args',  # arguments to the function, as a string like '(1, 2)'
        'start_line',  # n where trace_lines[n] contains 'Calling {fn_name}{args}...'
        'end_line',  # n where trace_lines[n] contains '...{fn_name} returns {value}'
    ])


def is_boxed(expression):
    return bool(re.fullmatch(r'\s*\\boxed\s*{\s*[^{}]*\s*}\s*', expression))


def extract_from_boxed(text: str) -> str:
    match = re.search(r"\$?\\boxed{(.*?)}\$?", text)
    if match:
        return match.group(1)
    return text


def extract_step_outputs(fn_name, trace_lines):
    """Generate StepOutput's for all steps with the given function name.
    fn_name can be None, in which case all steps will match.
    trace_lines is a list of strings.
    """
    enter_fn_regex = r'Calling (\w+)(\(.*\))\.\.\.'
    exit_fn_regex = r'\.\.\.(\w+) returned (.+)'
    stack = []
    outputs = []
    for line_num, line in enumerate(trace_lines):
        line = line.strip()
        m_enter = re.match(enter_fn_regex, line)
        m_exit = re.match(exit_fn_regex, line)
        if m_enter and (not fn_name or m_enter.group(1) == fn_name):
            step_fn = m_enter.group(1)
            args = m_enter.group(2)
            stack.append((step_fn, args, line_num))
        if m_exit and (not fn_name or m_exit.group(1) == fn_name):
            returned_value = m_exit.group(2)
            end_line_num = line_num
            if not stack:
                return None
            step_fn, args, start_line_num = stack.pop()
            outputs.append(
                StepOutput(value=returned_value,
                           fn_name=step_fn,
                           args=args,
                           start_line=start_line_num,
                           end_line=end_line_num))
    return outputs


def extract_functions_from_partial_programs(partial_program):
    """Extracts all valid function names from the partial program.
    
    Returns:
        A list of function names (strings) found in the partial program.
    """
    pattern = r'^\s*def\s+([a-zA-Z_]\w*)\s*\('
    fn_names = re.findall(pattern, partial_program, flags=re.MULTILINE)
    return fn_names


def convert_sympy_to_float(expr):
    """Convert a Sympy expression to a standard Python float.
    This function uses evalf() to numerically evaluate the expression.
    """
    if isinstance(expr, list):
        expr = expr[0]
    try:
        # Evaluate numerically and cast to float.
        return float(expr.evalf())
    except Exception as e:
        print(f"Error converting sympy expression to float: {expr}, {e}")
        raise e


def medcalc_bench_eval(answer, ground_truth, extra_info=None):
    calid = extra_info.get('calid', 0)
    lower_limit = extra_info.get('lower_limit', 0)
    upper_limit = extra_info.get('upper_limit', 0)
    calid = int(calid)
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

        elif calid in [69]:
            # Output Type: integer (A, B)
            answer = extract_from_boxed(answer)
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

        elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:
            gold_parsed = parse(
                ground_truth,
                extraction_mode='first_match',
            )
            if not is_boxed(answer):
                answer = f"$\\boxed{{{answer}}}$"
            answer_parsed = parse(
                answer,
                extraction_config=[
                    LatexExtractionConfig(
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
                    LatexExtractionConfig(
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
        # print(f"Error in calculator ID {calid}: {e}, answer: {answer}, ground_truth: {ground_truth}, answer_parsed: {answer_parsed}, gold_parsed: {gold_parsed}")
        correctness = 0
    return correctness, str(answer)


def accuracy_reward(completion, solution, extra_info=None):
    answer_pattern = re.compile(r"<answer>\n(.*?)\n</answer>", re.DOTALL)
    answer_match = answer_pattern.search(completion)
    if answer_match is None:
        return 0.0, str(completion)
    answer = answer_match.group(1).strip()
    if answer == solution:
        return 1.0, str(answer)

    if extra_info is not None:
        task = extra_info.get('task', None)
        if task == 'medcalc':
            return medcalc_bench_eval(answer, solution, extra_info)

    gold_parsed = parse(
        solution,
        extraction_mode="first_match",
    )
    if len(gold_parsed) != 0:
        if not is_boxed(answer):
            answer = f"$\\boxed{{{answer}}}$"
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            answer,
            extraction_config=[
                LatexExtractionConfig(
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
        try:
            reward = float(verify(gold_parsed, answer_parsed))
            try:
                parsed_str = sstr(answer_parsed[0]) if isinstance(answer_parsed, list) and len(answer_parsed) > 0 else (
                    sstr(answer_parsed) if answer_parsed is not None else str(answer))
            except Exception:
                parsed_str = str(answer)
            return reward, parsed_str
        except Exception as e:
            print(f"Error in accuracy_reward: {e}")
            try:
                parsed_str = sstr(answer_parsed) if answer_parsed is not None else str(answer)
            except Exception as e:
                parsed_str = str(answer)
            return 0.0, parsed_str
    else:
        return 0.0, str(answer)


# Note: Add this makes model separate its reasoning into multiple calls now
# but still it is using natural language reasoning
# before it just use one function call to do everything
# def trace_step_rewards(step_outputs):
#     num_steps = len(step_outputs)
#     return min(1.0, num_steps / 5)

# def trace_step_rewards(step_outputs, max_penalty=-1.0):
#     num_steps = len(step_outputs)
#     # magic number to encourage 5 steps or more
#     step_reward = min(1.0, num_steps / 5)

#     # unique_functions = {step.fn_name for step in step_outputs}
#     # # diversity reward
#     # diversity_reward = len(unique_functions) / num_steps

#     # penalty for repetition
#     all_calls = [(step.fn_name, step.args, step.value) for step in step_outputs]
#     unique_calls = set(all_calls)
#     total_calls = len(all_calls)
#     penalty = 1 - (len(unique_calls) / total_calls) * max_penalty
#     return max(max_penalty, step_reward - penalty)


# TODO: make this continuous?
def trace_step_rewards(step_outputs):
    all_calls = [(step.fn_name, step.args, step.value) for step in step_outputs]
    unique_calls = set(all_calls)
    total_calls = len(all_calls)
    step_reward = min(1.0, total_calls / 5)
    return step_reward * (len(unique_calls) / total_calls)


def format_reward(completion, extra_info=None):
    """
    <think>
    <partial_program>
    ...
    </partial_program>

    <program_trace>
    ...
    </program_trace>
    </think>
    <answer>
    ...
    </answer>

    prefix_type: 0 / 1 / 2
    1: No modifications
    2: Augment
    0: directly use the question without any base partial program
    """
    # ensure nothing precedes the <think> block (apart from whitespace).
    think_start = re.search(r"<think>\n", completion)
    if not think_start or completion[:think_start.start()].strip():
        return 0.0

    # there should be only one answer block
    answer_blocks = re.findall(r"<answer>\n.*?\n</answer>", completion, re.DOTALL)
    if len(answer_blocks) != 1 or not re.search(r"<answer>\n.*?\n</answer>\s*$", completion, re.DOTALL):
        # print("Invalid answer block")
        return 0.0

    # no tags in the answer block
    answer_match = re.search(r"<answer>\n(.*?)\n</answer>\s*$", completion, re.DOTALL)
    if answer_match:
        answer_inner = answer_match.group(1)
        disallowed_tags = [
            "<think>", "<partial_program>", "<program_trace>", "<reflection>", "</think>", "</partial_program>",
            "</program_trace>", "</reflection>"
        ]
        if any(tag in answer_inner for tag in disallowed_tags):
            return 0.0
    else:
        return 0.0

    # only one think block
    think_blocks = re.findall(r"<think>\n.*?\n</think>", completion, re.DOTALL)
    if len(think_blocks) != 1:
        # print("Invalid think block")
        return 0.0

    # ensure that the <think> block is entirely before the <answer> block.
    think_match = re.search(r"<think>\n(.*?)\n</think>", completion, re.DOTALL)
    answer_match = re.search(r"<answer>\n.*?\n</answer>\s*$", completion, re.DOTALL)
    if think_match is None or answer_match is None or think_match.end() > answer_match.start():
        # print("Think block should be before answer block")
        return 0.0

    # check if there is any content between </think> and <answer> tags
    think_end = re.search(r"</think>", completion)
    answer_start = re.search(r"<answer>", completion)
    if think_end and answer_start:
        between_content = completion[think_end.end():answer_start.start()].strip()
        if between_content:
            # print("Invalid content between </think> and <answer>")
            return 0.0

    # validate the structure inside the <think> block.
    # The think block must contain exactly a <partial_program> block followed (optionally separated by whitespace)
    # by a <program_trace> block, and nothing else.
    think_content = think_match.group(1)
    pattern = r"^<partial_program>\n(.*?)\n</partial_program>\s*<program_trace>\n(.*?)\n</program_trace>$"
    think_inner_match = re.match(pattern, think_content, re.DOTALL)
    if not think_inner_match:
        return 0.0

    partial_program_content = think_inner_match.group(1).strip()
    program_trace_content = think_inner_match.group(2).strip()
    trace_lines = program_trace_content.split('\n')
    step_outputs = extract_step_outputs(None, trace_lines)
    if step_outputs is None or len(step_outputs) == 0:
        return 0.0

    if extra_info is None:
        return trace_step_rewards(step_outputs)

    base_partial_program = extra_info.get('base_partial_program', None)
    prefix_type = extra_info.get('prefix_type', 0)
    if base_partial_program is None:
        return trace_step_rewards(step_outputs)

    base_partial_program_funcs = extract_functions_from_partial_programs(base_partial_program)
    defined_functions = extract_functions_from_partial_programs(partial_program_content)

    # check that trace uses and only uses the defined functions
    valid_usage = all(step.fn_name in defined_functions for step in step_outputs)
    used_functions = set(step.fn_name for step in step_outputs)
    all_used = all(fn in used_functions for fn in defined_functions) if defined_functions else False
    if not valid_usage or not all_used:
        return 0.0

    if len(set(defined_functions)) < 3:
        return 0.0

    if prefix_type == 0:
        return trace_step_rewards(step_outputs)
    elif prefix_type == 1:
        # no modifications
        # so partial program should use the same set of functions as the base partial program
        if set(defined_functions) != set(base_partial_program_funcs):
            return 0.0
    elif prefix_type == 2:
        # augment
        # so partial program should be a superset of the base partial program
        if not set(base_partial_program_funcs) < set(defined_functions):
            return 0.0
    else:
        # invalid prefix type
        return 0.0

    return trace_step_rewards(step_outputs)


def baseline_format_reward(completion, extra_info=None):
    """
    <think>
    </think>
    <answer>
    </answer>
    """
    think_start = re.search(r"<think>\n", completion)
    if not think_start or completion[:think_start.start()].strip():
        return 0.0
    # there should be only one answer block
    answer_blocks = re.findall(r"<answer>\n(.*?)\n</answer>", completion, re.DOTALL)
    if len(answer_blocks) != 1 or not re.search(r"<answer>\n.*?\n</answer>\s*$", completion, re.DOTALL):
        # print("Invalid answer block")
        return 0.0
    # no tags in the answer block
    answer_match = re.search(r"<answer>\n(.*?)\n</answer>\s*$", completion, re.DOTALL)
    if answer_match:
        answer_inner = answer_match.group(1)
        disallowed_tags = [
            "<think>", "<partial_program>", "<program_trace>", "<reflection>", "</think>", "</partial_program>",
            "</program_trace>", "</reflection>"
        ]
        if any(tag in answer_inner for tag in disallowed_tags):
            return 0.0
    else:
        return 0.0
    # only one think block
    think_blocks = re.findall(r"<think>\n.*?\n</think>", completion, re.DOTALL)
    if len(think_blocks) != 1:
        # print("Invalid think block")
        return 0.0
    # ensure that the <think> block is entirely before the <answer> block.
    think_match = re.search(r"<think>\n(.*?)\n</think>", completion, re.DOTALL)
    answer_match = re.search(r"<answer>\n.*?\n</answer>\s*$", completion, re.DOTALL)
    if think_match is None or answer_match is None or think_match.end() > answer_match.start():
        # print("Think block should be before answer block")
        return 0.0

    # check if there is any content between </think> and <answer> tags
    think_end = re.search(r"</think>", completion)
    answer_start = re.search(r"<answer>", completion)
    if think_end and answer_start:
        between_content = completion[think_end.end():answer_start.start()].strip()
        if between_content:
            # print("Invalid content between </think> and <answer>")
            return 0.0
    return 1.0


def repetition_penalty(max_penalty: float):
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def compute_trace_penalty(trace_block: str) -> float:
        """
        Repeat penalty for trace (same fn_name, same arguments, same return value)
        """
        trace_lines = trace_block.splitlines()
        step_outputs = extract_step_outputs(None, trace_lines)
        if not step_outputs:
            return 0.0
        all_calls = [(so.fn_name, so.args, so.value) for so in step_outputs]
        unique_calls = set(all_calls)
        total_calls = len(all_calls)
        scaling = 1 - (len(unique_calls) / total_calls)
        return scaling * max_penalty

    def compute_penalty(completion: str) -> float:
        think_match = re.search(r"<think>\n(.*?)\n</think>", completion, re.DOTALL)
        if not think_match:
            return 0.0
        think_block = think_match.group(1)
        pattern = r"^<partial_program>\n(.*?)\n</partial_program>\s*<program_trace>\n(.*?)\n</program_trace>$"
        think_inner_match = re.match(pattern, think_block, re.DOTALL)
        if not think_inner_match:
            return 0.0
        partial_program_block = think_inner_match.group(1)
        program_trace_block = think_inner_match.group(2)

        penalty = compute_trace_penalty(program_trace_block)
        return max(max_penalty, penalty)

    return compute_penalty


def extraction_output(completion):
    model_output = completion.split("<|im_start|>assistant\n", 1)[1]
    return model_output


def compute_score(solution_str, ground_truth, extra_info=None):
    if "<|im_start|>assistant\n" in solution_str:
        solution_str = extraction_output(solution_str)
    accuracy_score, pred = accuracy_reward(solution_str, ground_truth, extra_info)
    format_score = format_reward(solution_str, extra_info)
    total_score = accuracy_score * 1.0 + format_score * 1.0

    return {
        'score': total_score,
        "acc": accuracy_score == 1.0,
    }


def compute_baseline_score(solution_str, ground_truth, extra_info=None):
    if "<|im_start|>assistant\n" in solution_str:
        solution_str = extraction_output(solution_str)
    accuracy_score, pred = accuracy_reward(solution_str, ground_truth, extra_info)
    format_score = baseline_format_reward(solution_str, extra_info)
    total_score = accuracy_score * 1.0 + format_score * 1.0

    return {
        'score': total_score,
        "acc": accuracy_score == 1.0,
    }


if __name__ == "__main__":
    completion = "To calculate the estimated gestational age, we need to determine the difference between today's date and the last menstrual period date.\n\n1. **Identify the dates:**\n   - Last Menstrual Period (LMP): 04/15/2003\n   - Current Date: 06/16/2003\n\n2. **Calculate the difference in days:**\n   - From April 15 to April 30: 15 days (since April has 30 days)\n   - May has 31 days, so from May 1 to May 31: 31 days\n   - June 1 to June 16: 16 days\n\n3. **Sum up the days:**\n   - Days in April: 15\n   - Days in May: 31\n   - Days in June: 16\n   - Total days: 15 + 31 + 16 = 62 days\n\n4. **Convert days into weeks and days:**\n   - 62 days \u00f7 7 days/week = 8 weeks and 6 days (since 62 = 8 * 7 + 6)\n\nTherefore, the estimated gestational age is (8 weeks, 6 days).\n\n<answer>\n\\boxed{(8 weeks, 6 days)}\n</answer>"

    solution = "('8 weeks', '6 days')"
    extra_info = {
        'task': 'medcalc',
        'calid': 69,
    }
    print(accuracy_reward(completion, solution, extra_info))

    trace = """<think>
The formula for computing the albumin corrected delta ratio is albumin corrected delta gap (mEq/L)/(24 - bicarbonate mEq/L).
To compute the formula of albumin corrected delta gap, the formula is albumin corrected anion gap (in mEq/L) - 12.
The formula for computing a patient's albumin corrected anion gap is: anion_gap (in mEq/L) + 2.5 * (4 - albumin (in g/dL)).
The formula for computing a patient's anion gap is: sodium (mEq/L) - (chloride (mEq/L)+ bicarbonate (mEq/L)).
The concentration of sodium is 133.0 mEq/L. 
The concentration of chloride is 105.0 mEq/L. 
The concentration of bicarbonate is 22.0 mEq/L. 
Plugging in these values into the anion gap formula gives us 133.0 mEq/L - (105.0 mEq/L + 22.0 mEq/L) = 6.0 mEq/L. Hence, The patient's anion gap is 6.0 mEq/L.
The concentration of albumin is 3.2 g/dL. Plugging in these values into the albumin corrected anion gap formula, we get 6.0 (mEq/L) + 2.5 * (4 - 3.2 (in g/dL)) = 7.0 mEq/L. Hence, the patient's albumin corrected anion gap is 7.0 mEq/L.
Plugging in 7.0 mEq/L for the anion gap into the albumin corrected delta gap formula, we get 7.0 - 12 = -5.0 mEq/L. Hence, the patient's albumin corrected delta gap is -5.0 mEq/L.
Plugging in the albumin corrected delta gap and the bicarbonate concentration into the albumin corrected delta ratio formula, we get -5.0 mEq/L / 2.0 mEq/L = -2.5. The patient's albumin corrected delta ratio is -2.5.
</think>
<answer>
$\\boxed{-2.5}$
</answer>"""
    reward = compute_baseline_score(trace, solution, extra_info)
    print(reward)
