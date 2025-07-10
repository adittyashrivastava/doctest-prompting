import sys
# Force stdout to use UTF-8 in Python 3.7+
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from contextlib import redirect_stdout
import json
import re
import time
import os

# run a prompt on a set of examples and save the result in a log file

import arg_util
import llm_util
import local_model_util

# Import attention_viz for attention analysis and ATTRIEVAL
try:
    from attention_viz import AttentionExtractor, AttrievelRetriever, AttrievelConfig
    import torch
    import transformers
    ATTENTION_VIZ_AVAILABLE = True
    print("✅ attention_viz module loaded successfully")
except ImportError as e:
    print(f"⚠️  attention_viz not available: {e}")
    ATTENTION_VIZ_AVAILABLE = False

# helper functions

def parse_output(args, output):
    """Find xxx in the 'Final output: xxx' line.

    Works for output tagged with 'Final answer: ...', and sometimes
    for multiple choice answers not so tagged, if
    args.baseline_template_format is set.
    """
    # if baseline_template_format, return last occurrence of (X) for
    # any X
    if args.baseline_template_format:
        last_option = None
        for line in output.split('\n'):
            m = re.search(r'([A-H])\)', line) or re.search('Option\s+([A-H])', line)
            if m:
                last_option = m.group(1)
        if last_option is not None:
            return last_option
    # else scan for "Final answer: .+"
    for line in output.split('\n'):
        line = line.strip()
        m = re.search(r'Final answer: (.+)', line)
        if m:
            return m.group(1)
    return '**parse failed**'

def normalize_target(target):
    """Normalize a multiple-choice or numeric answer.
    """
    # parens around a multiple-choice answer are optional
    m = re.search(r'\(([A-Z])\)', target)
    if m: return m.group(1)
    else:
        # .0 at the end of a numerical answer is also optional
        m = re.search(r'([0-9]+)\.0+$', target)
        if m: return m.group(1)
        else: return target

def echo(fp, x):
    """Print something to stdout as well as a file.
    """
    print(x)
    with redirect_stdout(fp):
        print(x)

def check_answer(args, output, target):
    """Check the prediction in an output.

    Returns a triple: 
      predicted value, or '**parse failed**' if not extracted from output 
      whether that value is correct, after normalization
      whether the predicted value was extracted
    """
    prediction = parse_output(args, output)
    is_correct = (normalize_target(prediction) == normalize_target(target))
    return (prediction, is_correct, (prediction == '**parse failed**'))

def setup_attention_analysis(args):
    """Setup attention analysis components if enabled"""
    if not getattr(args, 'enable_attention_analysis', False) or not ATTENTION_VIZ_AVAILABLE:
        return None, None

    try:
        print("🔧 Setting up attention analysis...")

        # Setup output directory
        log_file = arg_util.log_file(args)
        log_dir = os.path.dirname(log_file)
        output_dir = os.path.join(log_dir, "attention_analysis")
        os.makedirs(output_dir, exist_ok=True)

        print(f"✅ Attention analysis setup complete")
        print(f"📁 Results will be saved to: {output_dir}")

        return output_dir, True

    except Exception as e:
        print(f"❌ Failed to setup attention analysis: {e}")
        return None, None

def perform_attention_analysis(prompt, response, input_context, target, output_dir, example_idx, model_obj, tokenizer):
    """Perform attention analysis for a single example"""
    if not ATTENTION_VIZ_AVAILABLE or model_obj is None:
        return None

    try:
        print(f"🔍 Analyzing example {example_idx}...")

        # Initialize attention extractor and retriever
        extractor = AttentionExtractor(model_obj, tokenizer)
        config = AttrievelConfig(
            layer_fraction=0.25,      # Use last 25% of layers
            top_k=10,                 # Top 10 tokens per CoT token
            frequency_threshold=0.99, # Filter attention sinks
            max_facts=10              # Retrieve top 10 facts
        )
        retriever = AttrievelRetriever(extractor, config)

        # Run ATTRIEVAL fact retrieval
        retrieval_result = retriever.retrieve_facts(
            context=input_context,
            question=input_context,  # For doctest problems, question is same as context
            cot_response=response,
            use_cross_evaluation=True
        )

        # Save results
        example_dir = os.path.join(output_dir, f"example_{example_idx:04d}")
        os.makedirs(example_dir, exist_ok=True)

        # Save top facts
        top_facts = {
            "example_idx": example_idx,
            "input": input_context,
            "response": response,
            "target": target,
            "retrieved_facts": retrieval_result['retrieved_facts'],
            "num_facts": len(retrieval_result['retrieved_facts'])
        }

        with open(os.path.join(example_dir, "top_facts.json"), "w") as f:
            json.dump(top_facts, f, indent=2)

        print(f"✅ Analysis complete for example {example_idx} - {len(retrieval_result['retrieved_facts'])} facts retrieved")
        return example_dir

    except Exception as e:
        print(f"❌ Analysis failed for example {example_idx}: {e}")
        return None

#
# main
#

def main(args=None):

    """Main routine.
    """

    # parse args and echo them

    if args is None:
        parser = arg_util.baseparser()
        parser.add_argument(
            '--enable_attention_analysis',
            action='store_true',
            help='Enable attention analysis and ATTRIEVAL for each prompt-response pair (only works with local service)')
        args = parser.parse_args()
        arg_util.apply_shortcuts(args)
    print(args)

    # Setup attention analysis if enabled
    attention_output_dir, attention_enabled = setup_attention_analysis(args)

    if args.json_output:
        local_model_util.build_json(args)
    else:

        log_filename = arg_util.log_file(args) 
        print(f'logging to {log_filename}')
        log_filemode = 'a' if args.append_to_log else 'w'
        with open(log_filename, log_filemode, encoding="utf-8", errors="replace") as log_fp:
            echo(log_fp, args)

            # load template file and echo it

            with open(args.template_file) as fp:
                template = fp.read()
                if args.baseline_template_format:
                    canary_sep = '\n-----\n'
                    template = template[template.find(canary_sep)+len(canary_sep):]
                    template += '\n\nQ: {input_str}'

            echo(log_fp, f'{"=" * 30} prompt template {"=" * 30}')
            echo(log_fp, template)

            # load partial_program_file and echo modified template

            if not args.baseline_template_format:
                partial_program_file = arg_util.partial_program_file(args)
                with open(partial_program_file) as fp:
                    partial_program = fp.read()
                # do NOT use format here, since any empty sets written out in the program traces
                # will confuse the format code
                template = template.replace('{task_name}', args.task)
                template = template.replace('{partial_program}', partial_program)

            echo(log_fp, f'{"=" * 30} template with program {"=" * 30}')
            template_lines = template.split('\n')
            if len(template_lines) < 100:
                echo(log_fp, template)
            else:
                for line in template_lines[0:50]:
                    echo(log_fp, line.strip())
                echo(log_fp, '.' * 50)
                echo(log_fp, f'{len(template_lines) - 100} lines skipped')
                echo(log_fp, '.' * 50)
                for line in template_lines[-50:]:
                    echo(log_fp, line.strip())

            # load examples

            example_file = arg_util.example_file(args)
            with open(example_file) as fp:
                examples = json.loads(fp.read())['examples']

            parse_failures = correct = total = 0
            attention_results = []
            
            for example_idx, ex in enumerate(arg_util.active_subset(args, examples)):
                x = ex['input']
                y = ex['target'] 
                # do NOT use format here, since any empty sets written out in the program traces
                # will confuse the format code
                prompt = template.replace('{input_str}', x)

                echo(log_fp, f'prompting {args.service}:{args.model}')
                echo(log_fp, '-' * 30 + ' input ' + '-' * 30)
                echo(log_fp, x)    
                if args.service is None:
                    raise ValueError('--service must be set')
                
                # Get response and model objects for attention analysis
                if args.service == 'local' and attention_enabled:
                    output, model_obj, tokenizer = llm_util.llm_with_model(prompt, service=args.service, model=args.model)
                else:
                    output = llm_util.llm(prompt, service=args.service, model=args.model)
                    model_obj, tokenizer = None, None
                
                echo(log_fp, '-' * 30 + ' output ' + '-' * 30)
                echo(log_fp, output)
                prediction, is_correct, parse_failed = check_answer(args, output, y)
                total += 1
                if is_correct: correct += 1
                if parse_failed: parse_failures += 1
                
                # Perform attention analysis if enabled and model is available
                attention_analysis_dir = None
                if attention_enabled and model_obj is not None and tokenizer is not None:
                    attention_analysis_dir = perform_attention_analysis(
                        prompt, output, x, y, attention_output_dir, example_idx, model_obj, tokenizer
                    )
                
                if attention_analysis_dir:
                    attention_results.append({
                        "example_idx": example_idx,
                        "attention_analysis_dir": os.path.relpath(attention_analysis_dir, os.path.dirname(log_filename)),
                        "has_attention_analysis": True
                    })
                
                echo(log_fp, 
                    '-' * 30 + f' {correct=} {total=} {parse_failures=} {prediction=} {y=} {is_correct=} ' + '-' * 30)
                time.sleep(args.delay)

            if parse_failures:
                parsed = total - parse_failures
                acc = correct / parsed
                echo(log_fp, f'Final totals (ignoring parse failures) {correct=} {parsed=} {acc=}') 
            acc = correct / total
            echo(log_fp, f'Final totals {correct=} {total=} {acc=}')
            
            # Report attention analysis summary
            if attention_enabled and attention_results:
                echo(log_fp, "=" * 30 + "Attention Analysis Summary" + "=" * 30)
                echo(log_fp, f"Attention analysis completed for {len(attention_results)}/{total} examples")
                echo(log_fp, f"Results saved to: {attention_output_dir}")

if __name__ == "__main__":
    main(None)
