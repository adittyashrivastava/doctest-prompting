import collections
import re
import os
import json
import random
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from constants import PTP_SYSTEM_PROMPT
from verl.utils.reward_score.ptp import medcalc_bench_eval

random.seed(1234)


def parse_output(output):
    for line in output.split('\n'):
        line = line.strip()
        m = re.search(r'Final answer: (.+)', line)
        if m:
            return m.group(1)
    return '**parse failed**'


def normalize_target(target):
    target = target.replace('\'', '')
    # parens around a multiple-choice answer are optional
    m = re.search(r'\(([A-Z])\)', target)
    if m:
        return m.group(1)
    else:
        # .0 at the end of a numerical answer is also optional
        m = re.search(r'([0-9]+)\.0+$', target)
        if m:
            return m.group(1)
        else:
            return target


def is_boxed(expression):
    return bool(re.fullmatch(r'\$\s*\\boxed\s*{.*}\s*\$', expression))


def check_answer(output, target, extra_info=None):
    answer_pattern = re.compile(r"<answer>\n(.*?)\n</answer>", re.DOTALL)
    answer_match = answer_pattern.search(output)
    if answer_match:
        output = answer_match.group(1)
    output = parse_output(output)
    if normalize_target(output) == normalize_target(target):
        return True, normalize_target(output)
    else:
        if extra_info is not None:
            extra_info['calid'] = extra_info['calculator_id']
            return medcalc_bench_eval(output, target, extra_info)
        else:
            verify_func = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
            )
            try:
                score, _ = verify_func([target], [output])
            except Exception as e:
                score = 0.0
            if score == 1.0:
                return True, normalize_target(output)
            else:
                return False, normalize_target(output)


StepOutput = collections.namedtuple(
    'StepOutput',
    [
        'value',  # return value as a string
        'fn_name',  # function name
        'args',  # arguments to the function, as a string like '(1, 2)'
        'start_line',  # n where trace_lines[n] contains 'Calling {fn_name}{args}...'
        'end_line',  # n where trace_lines[n] contains '...{fn_name} returns {value}'
    ])


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


# sometimes the trace is wrapped in ```
# and sometimes it repeat the question >>>
def clean_trace(output):
    # remove code block markers regardless of content
    output = re.sub(r"^```[\w]*\n", "", output)
    output = re.sub(r"\n```$", "", output)
    output = output.replace("\n@traced\n", "")

    lines = output.split("\n")
    cleaned_lines = []

    for i, line in enumerate(lines):
        if re.match(r"^(>>>|>) \w+\(.*\).*$", line):
            continue
        cleaned_lines.append(line)
    output = "\n".join(cleaned_lines)

    pattern = r"Final answer:.*"
    result = re.sub(pattern, "", output, flags=re.IGNORECASE | re.DOTALL)
    result = result.rstrip()

    # notice these edge cases, especially with claude reflections
    # count how many # used in the trace (new line)
    # if there are more than 3, we want to remove them
    num_comment_lines = sum(1 for line in output.splitlines() if line.strip().startswith("#"))
    if num_comment_lines > 2:
        return None

    return result.strip()


# remove the last function in the partial program
# since it is usually task specific
def remove_non_traced_functions(code: str) -> str:
    new_code = ""
    pattern = r"(.*)(def [^\n]+)\n(.*$)"
    matches = re.search(pattern, code, re.DOTALL)
    if matches:
        new_code = matches.group(1).strip()
    return new_code


def get_partial_programs(partial_program_dir, task, variant):
    if variant and not variant.startswith("_"):
        variant = f"_{variant}"
    with open(os.path.join(partial_program_dir, f"{task + variant}.py")) as fp:
        content = fp.read()
    return remove_non_traced_functions(content)


def parse_partial_programs(partial_program_dir, task, variant):
    partial_program = get_partial_programs(partial_program_dir, task, variant)
    functions = {}
    # Pattern to match the start of a function definition with preceding decorators.
    # It captures the function name in a named group.
    header_pattern = re.compile(r'(?P<start>(?:^\s*@.*\n)*^\s*def\s+(?P<name>[a-zA-Z_]\w*)\s*\(.*)', re.MULTILINE)

    matches = list(header_pattern.finditer(partial_program))

    if matches:
        first_match_start = matches[0].start()
        functions['extra'] = partial_program[:first_match_start]

        for i, match in enumerate(matches):
            name = match.group('name')
            start_index = match.start()
            end_index = matches[i + 1].start() if i + 1 < len(matches) else len(partial_program)
            func_block = partial_program[start_index:end_index]
            functions[name] = func_block
    else:
        functions['extra'] = partial_program

    return functions


def construct_partial_programs(partial_program_funcs, step_outputs):
    step_fn_names = [step.fn_name for step in step_outputs]
    # reconstruct the partial programs using only used functions
    # and the extra code
    partial_program = partial_program_funcs['extra']
    for name, func in partial_program_funcs.items():
        if name in step_fn_names:
            partial_program += f"{func}\n"
    return partial_program


def extract_functions_from_partial_programs(partial_program):
    """
    Extracts all valid function names from the partial program.
    
    Returns:
        A list of function names (strings) found in the partial program.
    """
    pattern = r'^\s*def\s+([a-zA-Z_]\w*)\s*\('
    fn_names = re.findall(pattern, partial_program, flags=re.MULTILINE)
    return fn_names


def format_hf_sample(question, partial_program, trace, answer, base_partial_program=None, task=None, metadata=None):
    prefix_1 = "Use the existing partial program without modification to answer the question."
    prefix_2 = "Augment the current partial program to answer the question."
    if task == 'gsm8k' or task == 'math500' or task == 'multistep_arithmetic_two':
        answer = f"$\\boxed{{{answer}}}$"

    if base_partial_program is None:
        user_instruction = question
        prefix_type = 0
        # 20% we use its partial program in prompt and we add prefix
        # do not add functions for partial program
        if random.random() < 0.2:
            base_partial_program = partial_program
            user_instruction = f'{question}\n\n<partial_program>\n{base_partial_program}\n</partial_program>\n\n{prefix_1}'
            prefix_type = 1
    else:
        user_instruction = f'{question}\n\n<partial_program>\n{base_partial_program}\n</partial_program>\n\n{prefix_2}'
        prefix_type = 2

    assistant_response = f'<think>\n<partial_program>\n{partial_program}\n</partial_program>\n\n<program_trace>\n{trace}\n</program_trace>\n</think>\n<answer>\n{answer}\n</answer>'
    metadata['prefix_type'] = prefix_type
    metadata['base_partial_program'] = base_partial_program
    sample = {
        'system': PTP_SYSTEM_PROMPT,
        'instruction': [
            {
                'role': 'system',
                'content': PTP_SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': user_instruction
            },
        ],
        'response': [{
            'role': 'assistant',
            'content': assistant_response
        },],
        'messages': [
            {
                'role': 'system',
                'content': PTP_SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': user_instruction
            },
            {
                'role': 'assistant',
                'content': assistant_response
            },
        ],
        'metadata': metadata if metadata else {},
        'task': task if task else '',
    }
    return sample


def format_hf_clean_sample(question,
                           partial_program,
                           trace,
                           answer,
                           base_partial_program=None,
                           task=None,
                           metadata=None):
    metadata['prefix_type'] = 0
    metadata['base_partial_program'] = base_partial_program
    user_instruction = question
    assistant_response = f'<think>\n<partial_program>\n{partial_program}\n</partial_program>\n\n<program_trace>\n{trace}\n</program_trace>\n</think>\n<answer>\n{answer}\n</answer>'
    sample = {
        'system': PTP_SYSTEM_PROMPT,
        'instruction': [
            {
                'role': 'system',
                'content': PTP_SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': user_instruction
            },
        ],
        'response': [{
            'role': 'assistant',
            'content': assistant_response
        },],
        'messages': [
            {
                'role': 'system',
                'content': PTP_SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': user_instruction
            },
            {
                'role': 'assistant',
                'content': assistant_response
            },
        ],
        'metadata': metadata if metadata else {},
        'task': task if task else '',
    }
    return sample


# def format_hf_data(grouped_data):
#     """
#     This one is tricky.

#     For each task (question family), do the following:
#     1. Group samples by their actual function set (ignoring extra)
#        even if two groups have the same number of functions, they will be treated separately
#        if their function sets differ

#     2. Sort these groups in ascending order by number of functions

#     3. Start an incremental chain with the group that has the fewest functions
#        This base group's partial program will be used for formatting the next question

#     4. For each subsequent group, check if the current base function set is a subset
#        of the new group's function set. If so, then for every sample in that group,
#        format it as:
#            Q_current + PP_base -> PP_current + trace + answer
#        then update the base to this group, so the next chain step uses this new PP

#     5. If a group does not contain the base, then just format it as:
#            Q_current -> PP_current + trace + answer

#     prefix_type: 0 / 1 / 2
#     1: using its own partial program as base partial program, and we instruct the model using existing ones without modification
#     2: using the base partial program as base partial program, and we instruct the model to augment the current one
#     0: directly use the question without any base partial program
#     """
#     all_formatted_samples = []
#     # Regroup by task.
#     task_groups = {}
#     for (task, func_keys), samples in grouped_data.items():
#         task_groups.setdefault(task, []).append((func_keys, samples))

#     for task, groups in task_groups.items():
#         # Sort groups by (number of functions, lex order of function names) for stability.
#         sorted_groups = sorted(groups, key=lambda x: (len(x[0]), sorted(list(x[0]))))
#         if not sorted_groups:
#             continue

#         # Base group: format every sample with no chaining.
#         base_func_set, base_samples = sorted_groups[0]
#         for sample in base_samples:
#             formatted_sample = format_hf_sample(
#                 sample['input'],
#                 sample['partial_program'],
#                 sample['trace'],
#                 sample['final_answer'],
#                 base_partial_program=None,
#                 task=sample['task'],
#                 metadata={
#                     'ground_truth': sample['ground_truth'],
#                     'prediction': sample['final_answer'],
#                     'is_correct': sample['is_correct'],
#                     'total_functions_defined': sample['total_functions_defined'],
#                     'actual_functions_used': sample['actual_functions_used'],
#                     'extra_info': sample.get('extra_info', None),
#                 }
#             )
#             all_formatted_samples.append(formatted_sample)
#         # Set the current base using the first sample of the base group.
#         current_base_func_set = base_func_set
#         current_base_pp = base_samples[0]['partial_program']

#         # Process each subsequent group.
#         for func_set, group_samples in sorted_groups[1:]:
#             if current_base_func_set.issubset(func_set):
#                 # Chain: use the current base's partial program for every sample.
#                 for sample in group_samples:
#                     formatted_sample = format_hf_sample(
#                         sample['input'],
#                         sample['partial_program'],
#                         sample['trace'],
#                         sample['final_answer'],
#                         base_partial_program=current_base_pp,
#                         task=sample['task'],
#                         metadata={
#                             'ground_truth': sample['ground_truth'],
#                             'prediction': sample['final_answer'],
#                             'is_correct': sample['is_correct'],
#                             'total_functions_defined': sample['total_functions_defined'],
#                             'actual_functions_used': sample['actual_functions_used'],
#                         }
#                     )
#                     all_formatted_samples.append(formatted_sample)
#                 # Update the base with this group.
#                 current_base_func_set = func_set
#                 current_base_pp = group_samples[0]['partial_program']
#             else:
#                 # Not chainable: format every sample without chaining.
#                 for sample in group_samples:
#                     formatted_sample = format_hf_sample(
#                         sample['input'],
#                         sample['partial_program'],
#                         sample['trace'],
#                         sample['final_answer'],
#                         base_partial_program=None,
#                         task=sample['task'],
#                         metadata={
#                             'ground_truth': sample['ground_truth'],
#                             'prediction': sample['final_answer'],
#                             'is_correct': sample['is_correct'],
#                             'total_functions_defined': sample['total_functions_defined'],
#                             'actual_functions_used': sample['actual_functions_used'],
#                         }
#                     )
#                     all_formatted_samples.append(formatted_sample)
#     return all_formatted_samples


def format_hf_data(grouped_data):
    """
    This one is tricky.

    For each task (question family), do the following:
    1. Group samples by their actual function set (ignoring extra)
       even if two groups have the same number of functions, they will be treated separately
       if their function sets differ

    2. Sort these groups in ascending order by number of functions

    3. Start an incremental chain with the group that has the fewest functions
       This base group's partial program will be used for formatting the next question

    4. For each subsequent group, check if the current base function set is a subset
       of the new group's function set. If so, then for every sample in that group,
       format it as:
           Q_current + PP_base -> PP_current + trace + answer
       then update the base to this group, so the next chain step uses this new PP

    Note:       
        Modified to consider *all* combinations.
    
        For each task, the samples are grouped by function set and sorted in ascending order.
        
        For the base (smallest function set) group, we output samples without chaining.
        Then, we maintain a list of candidate bases (tuples of (func_set, partial_program)).
        
        For each subsequent group, for every sample, we iterate over all candidate bases and if a base's function set 
        is a subset of the sample's function set, we output a formatted sample using that candidate's partial program.
        If no candidate is valid, the sample is formatted without chaining.
        
        After processing a group, it is added as a candidate base for subsequent groups.
      
    5. If a group does not contain the base, then just format it as:
           Q_current -> PP_current + trace + answer

    prefix_type: 0 / 1 / 2
    1: using its own partial program as base partial program, and we instruct the model using existing ones without modification
    2: using the base partial program as base partial program, and we instruct the model to augment the current one
    0: directly use the question without any base partial program
    """
    all_formatted_samples = []
    # Regroup by task.
    task_groups = {}
    for (task, func_keys), samples in grouped_data.items():
        task_groups.setdefault(task, []).append((func_keys, samples))

    for task, groups in task_groups.items():
        # Sort groups by (number of functions, lex order of function names) for stability.
        sorted_groups = sorted(groups, key=lambda x: (len(x[0]), sorted(list(x[0]))))
        if not sorted_groups:
            continue

        # Candidate bases: list of (func_set, partial_program)
        candidate_bases = []

        # Process the base group: format every sample with no chaining.
        base_func_set, base_samples = sorted_groups[0]
        for sample in base_samples:
            formatted_sample = format_hf_sample(sample['input'],
                                                sample['partial_program'],
                                                sample['trace'],
                                                sample['final_answer'],
                                                base_partial_program=None,
                                                task=sample['task'],
                                                metadata={
                                                    'ground_truth': sample['ground_truth'],
                                                    'prediction': sample['final_answer'],
                                                    'is_correct': sample['is_correct'],
                                                    'total_functions_defined': sample['total_functions_defined'],
                                                    'actual_functions_used': sample['actual_functions_used'],
                                                    'extra_info': sample.get('extra_info', None),
                                                })
            all_formatted_samples.append(formatted_sample)
        candidate_bases.append((base_func_set, base_samples[0]['partial_program']))

        # Process subsequent groups.
        for func_set, group_samples in sorted_groups[1:]:
            for sample in group_samples:
                # Check all candidate bases to see which ones are valid for chaining.
                valid_candidate_found = False
                for cand_func_set, cand_partial_program in candidate_bases:
                    if cand_func_set.issubset(func_set):
                        valid_candidate_found = True
                        formatted_sample = format_hf_sample(
                            sample['input'],
                            sample['partial_program'],
                            sample['trace'],
                            sample['final_answer'],
                            base_partial_program=cand_partial_program,
                            task=sample['task'],
                            metadata={
                                'ground_truth': sample['ground_truth'],
                                'prediction': sample['final_answer'],
                                'is_correct': sample['is_correct'],
                                'total_functions_defined': sample['total_functions_defined'],
                                'actual_functions_used': sample['actual_functions_used'],
                                'extra_info': sample.get('extra_info', None),
                            })
                        all_formatted_samples.append(formatted_sample)
                # If no candidate base applies, fall back to no chaining.
                if not valid_candidate_found:
                    formatted_sample = format_hf_sample(
                        sample['input'],
                        sample['partial_program'],
                        sample['trace'],
                        sample['final_answer'],
                        base_partial_program=None,
                        task=sample['task'],
                        metadata={
                            'ground_truth': sample['ground_truth'],
                            'prediction': sample['final_answer'],
                            'is_correct': sample['is_correct'],
                            'total_functions_defined': sample['total_functions_defined'],
                            'actual_functions_used': sample['actual_functions_used'],
                            'extra_info': sample.get('extra_info', None),
                        })
                    all_formatted_samples.append(formatted_sample)
            # Add this group as a candidate base for future groups.
            candidate_bases.append((func_set, group_samples[0]['partial_program']))
    return all_formatted_samples


def format_clean_hf_data(data):
    """
    We are not doing any augmentation for clean data
    """
    clean_data = []
    for sample in data:
        question = sample['input']
        partial_program = sample['partial_program']
        trace = sample['trace']
        answer = sample['final_answer']
        base_partial_program = None
        task = sample['task']
        metadata = {
            'ground_truth': sample['ground_truth'],
            'prediction': sample['final_answer'],
            'is_correct': sample['is_correct'],
            'total_functions_defined': sample['total_functions_defined'],
            'actual_functions_used': sample['actual_functions_used'],
            'extra_info': sample.get('extra_info', None),
        }
        sample = format_hf_clean_sample(question,
                                        partial_program,
                                        trace,
                                        answer,
                                        base_partial_program=base_partial_program,
                                        task=task,
                                        metadata=metadata)
        clean_data.append(sample)
    return clean_data


def format_hf_data_baseline(data):
    clean_data = []
    for sample in data:
        task = sample['task']
        if 'gsm' in task or 'math500' in task or 'multistep_arithmetic_two' in task or 'medcalc' in task:
            sample['final_answer'] = f"$\\boxed{{{sample['final_answer']}}}$"
        assistant_response = f"<think>\n{sample['trace']}\n</think>\n<answer>\n{sample['final_answer']}\n</answer>"
        new_sample = {
            'system': PTP_SYSTEM_PROMPT,
            'instruction': [
                {
                    'role': 'system',
                    'content': PTP_SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': sample['input']
                },
            ],
            'response': [{
                'role': 'assistant',
                'content': assistant_response
            },],
            'messages': [
                {
                    'role': 'system',
                    'content': PTP_SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': sample['input']
                },
                {
                    'role': 'assistant',
                    'content': assistant_response
                },
            ],
            'task': sample['task'],
        }
        clean_data.append(new_sample)
    return clean_data


if __name__ == "__main__":
    # partial_program_dir = '/project/flame/jixuanl/verl/afs_data/doctest-prompting/bbh/mocks/partialprograms'
    # task = 'boolean_expressions'
    # variant = ""
    # functions = parse_partial_programs(partial_program_dir, task, variant)
    # # for name, func in functions.items():
    # #     print(f"Function Name: {name}")
    # #     print(f"Function Definition:\n{func}\n")
    # step_outputs = [
    #     StepOutput(value="True", fn_name="solve_or", args="(2)", start_line=1, end_line=2),
    # ]
    # partial_program = construct_partial_programs(functions, step_outputs)
    # print("Constructed Partial Program:")
    a = "<program_trace>\nCalling extract_options('Today is 3/5, and it is Jane's second time in the year 1973 to see a meteor shower. What is the date today in MM/DD/YYYY?\\nOptions:\\n(A) 04/05/1973\\n(B) 03/02/1973\\n(C) 01/22/2007\\n(D) 01/02/1973\\n(E) 03/05/1973\\n(F) 03/08/1983\\n')...\n...extract_options returned [('A', '04/05/1973'), ('B', '03/02/1973'), ('C', '01/22/2007'), ('D', '01/02/1973'), ('E', '03/05/1973'), ('F', '03/08/1983')]\nCalling extract_date_facts('Today is 3/5, and it is Jane's second time in the year 1973 to see a meteor shower. What is the date today in MM/DD/YYYY?\\nOptions:\\n(A) 04/05/1973\\n(B) 03/02/1973\\n(C) 01/22/2007\\n(D) 01/02/1973\\n(E) 03/05/1973\\n(F) 03/08/1983\\n')...\n...extract_date_facts returned ['Today is 3/5,', 'and it is Jane's second time in the year 1973 to see a meteor shower.']\nCalling make_inference('Today is 3/5,', [])...\n...make_inference returned \"If today is 3/5, then today's date is 03/05/YYYY where YYYY represents the current year.\"\nCalling make_inference('and it is Jane's second time in the year 1973 to see a meteor shower.', [\"If today is 3/5, then today's date is 03/05/YYYY where YYYY represents the current year.\"])...\n...make_inference returned 'Since it is Jane\\'s second time in the year 1973 to see a meteor shower, the YYYY part of the date must be 1973.'\nCalling extract_question('Today is 3/5, and it is Jane's second time in the year 1973 to see a meteor shower. What is the date today in MM/DD/YYYY?\\nOptions:\\n(A) 04/05/1973\\n(B) 03/02/1973\\n(C) 01/22/2007\\n(D) 01/02/1973\\n(E) 03/05/1973\\n(F) 03/08/1983\\n')...\n...extract_question returned 'What is the date today in MM/DD/YYYY?'\nCalling answer_question('What is the date today in MM/DD/YYYY?', [\"If today is 3/5, then today's date is 03/05/YYYY where YYYY represents the current year.\", 'Since it is Jane\\'s second time in the year 1973 to see a meteor shower, the YYYY part of the date must be 1973.'])...\n...answer_question returned 'So today\\'s date in MM/DD/YYYY format must be 03/05/1973.'\nCalling match_option('So today\\'s date in MM/DD/YYYY format must be 03/05/1973.', ('A', '04/05/1973'))...\n...match_option returned False\nCalling match_option('So today\\'s date in MM/DD/YYYY format must be 03/05/1973.', ('B', '03/02/1973'))...\n...match_option returned False\nCalling match_option('So today\\'s date in MM/DD/YYYY format must be 03/05/1973.', ('C', '01/22/2007'))...\n...match_option returned False\nCalling match_option('So today\\'s date in MM/DD/YYYY format must be 03/05/1973.', ('D', '01/02/1973'))...\n...match_option returned False\nCalling match_option('So today\\'s date in MM/DD/YYYY format must be 03/05/1973.', ('E', '03/05/1973'))...\n...match_option returned True\nCalling match_option('So today\\'s date in MM/DD/YYYY format must be 03/05/1973.', ('F', '03/08/1983'))...\n...match_option returned False\n</program_trace>"
    step_outputs = extract_step_outputs(None, a.split('\n'))
    defined_functions = [
        'extract_options', 'extract_date_facts', 'extract_question', 'make_inference', 'answer_question', 'match_option'
    ]
    valid_usage = all(step.fn_name in defined_functions for step in step_outputs)
    used_functions = set(step.fn_name for step in step_outputs)
    all_used = all(fn in used_functions for fn in defined_functions) if defined_functions else False
    print("Valid usage:", valid_usage)
    print("Used functions:", used_functions)
    print("All used:", all_used)
