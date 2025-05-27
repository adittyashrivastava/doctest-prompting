PTP_SYSTEM_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The assistant first reasons through the problem by generating high-level partial programs with key parts hidden using "..." markers. It then simulates programs trace based on the incomplete partial programs. The partial program must be general enough to solve all instances of the problem type, not just specific examples. The partial programs and traces are enclosed within <partial_program> </partial_program> and <program_trace> </program_trace> tags, while the overall reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. You should also wrap your final answer in $\\boxed{{ANSWER}}$ if it is a mathematical expression.

Format:
<think>
<partial_program>
[Partial Program here]
</partial_program>
<program_trace>
[Program Trace here]
</program_trace>
</think>
<answer>
[Final Answer here]
</answer>"""
############################################################################################################
PTP_SYSTEM_PROMPT_WITH_REFLECTION = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The assistant first produces a high-level partial program with key parts hidden using "..." markers, along with a corresponding program trace. The partial program must be general enough to solve all instances of the problem type, not just specific examples. If the assistant determines that the partial program and trace are insufficient or could be improved, it then adds a reflection before generating a new iteration. Every subsequent iteration (after the first) starts with a <reflection> block, followed by an updated partial program and program trace. The partial programs, traces, and reflections are enclosed within <partial_program> </partial_program>, <program_trace> </program_trace>, and <reflection> </reflection> tags, respectively. The overall reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags. You should also wrap your final answer in $\\boxed{{ANSWER}}$ if it is a mathematical expression.

Format:
<think>
<partial_program>
[Partial Program here]
</partial_program>
<program_trace>
[Program Trace here]
</program_trace>
For each subsequent iteration (if needed):
<reflection>
[Reflection here]
</reflection>
<partial_program>
[Revised Partial Program here]
</partial_program>
<program_trace>
[Updated Program Trace here]
</program_trace>
After all reflections (if any):
</think>
<answer>
[Final Answer here]
</answer>"""
############################################################################################################
REFLECTION = """Now, carefully review your previous response.

1. Analyze the Partial Program:
- Determine if the partial program includes all necessary functions to solve the problem.
- If the current functions are insufficient, modify existing ones or add new functions as needed.
- Use high-level placeholders ("...") for implementation details.
- Ensure the revised program is general enough for similar problems; if a function is too broad, consider splitting it into more specific functions.

2. Analyze the Program Trace:
- If the partial program is sufficient, step through every function call and return in the trace.
- Identify any incorrect computations or branching that leads to the wrong final answer.
- Correct the trace based on these issues.
- Note: Since the program only defines functions without implementations, errors likely come from the trace; however, if the functions are inadequate, update the partial program as well.

3. Document your reflection:
- Clearly state whether the error lies in the partial program, the program trace, or both.
- If you modified the program, explain why extra functions were necessary.
- If the error was in the trace, describe the mistake and how you corrected it.
- Keep the tone neutral, as if you do not know the final correct answer.

4. Output Requirements:
- Your reflection should be detailed, structured, and coherent.
- After the reflection, provide: a. A completely revised partial program (or the unchanged one if correct). This program should only define functions with "..." for implementation and follow the original format. b. A complete, unabridged updated program trace showing every step from the beginning to the end, with no omissions or inline comments. c. The final correct answer.
- You must use the following format:

<reflection>
[Your reflection on what was incorrect and how you improved it]
</reflection>
<partial_program>
[Revised partial program]
</partial_program>
<program_trace>
[FULL updated program trace showing EVERY step from the VERY BEGINNING to the end]
</program_trace>
<answer>
[Final derived correct answer]
</answer>

# Important Notes:
- Do not reveal the final correct answer in your reflection.
- Do not include any implementation details in the partial program; only define the functions using "..." as the implementation.
- Every function used in the program trace must be defined in the partial program, and vice versa. Remove any unused functions.
- The program trace should only show function calls and returns, without any explanations or inline comments.
- Each function call in the program trace should represent a single step, and you must not skip any steps. It is acceptable to call a function multiple times if needed.
- Do not use Unicode; always use LaTeX for mathematical expressions."""
############################################################################################################
CONVERSION = """Your goal is to simulate a reasoning process using high-level partial programs and detailed program traces. Your final answer must match the original reasoning process's final answer.

# A Conversion Example:
<PTP>

# Reasoning Process to convert:
<reasoning>

# Instructions:
- Simulate the reasoning process using high-level partial programs and detailed program traces.
- The partial program should define general functions with no implementation details (use "..." as a placeholder) and remain abstract.
- The program trace must be specific and include every single step as an individual function call; do not skip or merge steps, even if the original reasoning omits intermediate steps. For example, when solving an equation, include each individual step (e.g., expansion, simplification, isolation of variables) as separate function calls, rather than jumping directly to the final answer.
- All mathematical expressions and calculations throughout the trace must be represented using LaTeX. Do not use Unicode characters; convert them to LaTeX. For example, convert \u03C0 to $\pi$, convert \u2264 to $\leq$, and convert \u2261 to $\equiv$, etc. Use $\\sqrt{{}}$ instead of $\\sqrt()$. Even if a question is provided in plain text, ensure that all mathematical content is rendered using LaTeX.
- The final output must begin with some reasoning, then output the final conversion in the exact format as the example.
- All defined functions in the program trace must be present in the partial program, and vice versa. Remove any unused functions.
- Do NOT show steps within return values.
- Do NOT use any inline comments in either the partial program or the program trace.
- Do NOT express any reasoning inside <think> tags.
- Do NOT include functions like "verify_final_answer" or "formulate_final_answer" that convert natural language into the final answer.
- YOU MUST box your final answer using $\\boxed{{ANSWER}}$ (including the dollar signs)."""
############################################################################################################
VERIFY = """Compare the two answers below and determine if they are equivalent in meaning and numerical value. Ignore differences in format, wording, or minor variations in expression.

## Attempt
{attempt}

## Correct answer
{solution}

Provide a brief explanation of your reasoning. Your final response must end with either "Yes" or "No" on a new line."""
############################################################################################################
LATEX_CONVERSION = """You are an expert in LaTeX. Your task is to convert any unicode equations or symbols in the following text into LaTeX mathematical expressions.
You MUST NOT change the text itself, only do the conversion.

Directly output the converted text without any additional comments or explanations.
If no conversion is needed, simply return the original text.

# Text to convert:
{content}
"""
