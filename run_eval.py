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
    """Perform attention analysis for a single example with GPU memory optimization"""
    if not ATTENTION_VIZ_AVAILABLE or model_obj is None:
        return None

    try:
        print(f"🔍 Analyzing example {example_idx}...")

        # Step 1: Compact the input to reduce memory usage
        print(f"🔍 Step 1: Compacting input for memory efficiency...")
        
        # Compact the prompt by truncating very long examples while preserving key info
        max_prompt_length = 1024  # Reasonable limit for attention analysis
        if len(prompt) > max_prompt_length:
            # Keep the beginning (task description) and end (question) of the prompt
            prompt_start = prompt[:max_prompt_length//2]
            prompt_end = prompt[-(max_prompt_length//2):]
            compact_prompt = prompt_start + "\n...[content truncated for memory efficiency]...\n" + prompt_end
            print(f"🔪 Compacted prompt from {len(prompt)} to {len(compact_prompt)} characters")
        else:
            compact_prompt = prompt
            print(f"✅ Prompt length acceptable: {len(prompt)} characters")

        # Combine compacted prompt and response for attention extraction
        full_text = compact_prompt + response

        # Tokenize with aggressive length management
        max_total_tokens = 1024  # Conservative limit for GPU memory
        inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=max_total_tokens)
        inputs = {k: v.to(model_obj.device) for k, v in inputs.items()}

        seq_len = inputs['input_ids'].shape[1]
        print(f"📏 Final sequence length: {seq_len} tokens (target: <={max_total_tokens})")

        # Step 2: Layer-selective attention extraction for memory efficiency
        print(f"🔍 Step 2: Layer-selective attention extraction...")
        
        # Clear GPU cache before attention extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Cleared GPU cache")

        # Configure model for layer-selective attention extraction
        original_output_attentions = getattr(model_obj.config, 'output_attentions', False)
        original_attn_implementation = getattr(model_obj.config, '_attn_implementation', None)
        original_use_cache = getattr(model_obj.config, 'use_cache', True)

        try:
            with torch.no_grad():
                # Enable attention extraction with memory-efficient settings
                model_obj.config.output_attentions = True
                if hasattr(model_obj.config, '_attn_implementation'):
                    model_obj.config._attn_implementation = 'eager'
                model_obj.config.use_cache = False  # Disable cache to save memory

                # Enable gradient checkpointing for memory efficiency
                if hasattr(model_obj, 'gradient_checkpointing_enable'):
                    model_obj.gradient_checkpointing_enable()
                    print(f"🔧 Enabled gradient checkpointing")

                print(f"🔧 Extracting attention with memory-optimized settings...")
                outputs = model_obj(**inputs, output_attentions=True)

                # Extract only the most relevant attention layers (last 25% of layers)
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    all_attentions = outputs.attentions
                    num_layers = len(all_attentions)
                    
                    # Keep only the last 25% of layers (most relevant for reasoning)
                    keep_layers = max(1, num_layers // 4)  # At least 1 layer
                    start_layer = num_layers - keep_layers
                    
                    # Extract only relevant layers and move to CPU to free GPU memory
                    relevant_attentions = []
                    for i in range(start_layer, num_layers):
                        attention_layer = all_attentions[i].cpu()  # Move to CPU immediately
                        relevant_attentions.append(attention_layer)
                    
                    print(f"✅ Extracted {len(relevant_attentions)} relevant layers (layers {start_layer}-{num_layers-1}) out of {num_layers} total")
                    
                    # Clear the full attention from GPU memory
                    del outputs.attentions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"🧹 Freed GPU memory after attention extraction")

                else:
                    print(f"❌ No attention weights extracted!")
                    return None

        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️  GPU OOM during attention extraction: {e}")
            print(f"🔧 Falling back to CPU-based attention extraction...")
            
            # Fallback: Move model to CPU for attention extraction
            try:
                # Move inputs to CPU
                inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
                
                # Move model to CPU temporarily
                model_obj_cpu = model_obj.cpu()
                
                with torch.no_grad():
                    model_obj_cpu.config.output_attentions = True
                    if hasattr(model_obj_cpu.config, '_attn_implementation'):
                        model_obj_cpu.config._attn_implementation = 'eager'

                    outputs_cpu = model_obj_cpu(**inputs_cpu, output_attentions=True)
                    
                    # Extract relevant layers
                    if hasattr(outputs_cpu, 'attentions') and outputs_cpu.attentions is not None:
                        all_attentions = outputs_cpu.attentions
                        num_layers = len(all_attentions)
                        keep_layers = max(1, num_layers // 4)
                        start_layer = num_layers - keep_layers
                        
                        relevant_attentions = [all_attentions[i] for i in range(start_layer, num_layers)]
                        print(f"✅ CPU extraction: {len(relevant_attentions)} layers extracted")
                    else:
                        print(f"❌ CPU extraction failed!")
                        return None

                # Move model back to GPU
                model_obj = model_obj_cpu.cuda()
                print(f"🔙 Model moved back to GPU")

            except Exception as e2:
                print(f"❌ CPU fallback failed: {e2}")
                return None

        finally:
            # Restore original model settings
            model_obj.config.output_attentions = original_output_attentions
            if original_attn_implementation is not None:
                model_obj.config._attn_implementation = original_attn_implementation
            else:
                if hasattr(model_obj.config, '_attn_implementation'):
                    delattr(model_obj.config, '_attn_implementation')
            model_obj.config.use_cache = original_use_cache

            # Disable gradient checkpointing
            if hasattr(model_obj, 'gradient_checkpointing_disable'):
                model_obj.gradient_checkpointing_disable()

        # Step 3: Initialize ATTRIEVAL with layer-selective attention
        print(f"🔍 Step 3: Setting up ATTRIEVAL with optimized config...")
        
        extractor = AttentionExtractor(model_obj, tokenizer)
        config = AttrievelConfig(
            layer_fraction=0.25,      # Use last 25% of layers (matches our extraction)
            top_k=5,                  # Reduce to 5 tokens per CoT token for memory
            frequency_threshold=0.95, # Lower threshold to get more facts
            max_facts=5               # Reduce to 5 facts for faster processing
        )
        retriever = AttrievelRetriever(extractor, config)

        # Step 4: Run ATTRIEVAL fact retrieval (let it extract attention internally)
        print(f"🔍 Step 4: Running ATTRIEVAL fact retrieval...")
        
        try:
            # Use original input_context (not compacted) for fact retrieval
            retrieval_result = retriever.retrieve_facts(
                context=input_context,
                question=input_context,  # For doctest problems, question is same as context
                cot_response=response,
                use_cross_evaluation=False  # Disable cross-evaluation to save memory
            )
            
            print(f"✅ ATTRIEVAL completed, found {len(retrieval_result.get('retrieved_facts', []))} facts")
            
        except Exception as e:
            print(f"❌ ATTRIEVAL failed: {e}")
            # Create minimal fallback result
            retrieval_result = {
                'retrieved_facts': [],
                'fact_scores': [],
                'attention_data': None
            }

        # Step 5: Process and clean results
        print(f"🔍 Step 5: Processing results...")
        retrieved_facts = retrieval_result.get('retrieved_facts', [])
        
        # Debug first few facts
        for i, fact in enumerate(retrieved_facts[:3]):
            score = fact.get('score', 'No score')
            text = fact.get('text', 'No text')[:30]
            print(f"   Fact {i}: score={score}, text='{text}...'")

        # Clean up scores
        cleaned_facts = []
        for fact in retrieved_facts:
            score = fact.get('score', 0)
            if isinstance(score, (int, float)) and not (score != score or score == float('inf') or score == float('-inf')):
                cleaned_facts.append(fact)
            else:
                fact_copy = fact.copy()
                fact_copy['score'] = 0.0
                cleaned_facts.append(fact_copy)

        print(f"✅ Final results: {len(cleaned_facts)} valid facts")

        # Step 6: Save results with minimal memory footprint
        example_dir = os.path.join(output_dir, f"example_{example_idx:04d}")
        os.makedirs(example_dir, exist_ok=True)

        # Save compact results
        results = {
            "example_idx": example_idx,
            "input_preview": input_context[:200] + "..." if len(input_context) > 200 else input_context,
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "target": target,
            "retrieved_facts": cleaned_facts,
            "num_facts": len(cleaned_facts),
            "optimization_info": {
                "sequence_length": seq_len,
                "max_tokens_used": max_total_tokens,
                "prompt_compacted": len(prompt) > max_prompt_length,
                "layers_extracted": len(relevant_attentions) if 'relevant_attentions' in locals() else 0,
                "memory_strategy": "layer_selective_extraction"
            }
        }

        with open(os.path.join(example_dir, "top_facts.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✅ Analysis complete for example {example_idx} - {len(cleaned_facts)} facts retrieved")
        return example_dir

    except Exception as e:
        print(f"❌ Analysis failed for example {example_idx}: {e}")
        import traceback
        traceback.print_exc()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
                    echo(log_fp, f"🧠 Starting attention analysis for example {example_idx}...")
                    attention_analysis_dir = perform_attention_analysis(
                        prompt, output, x, y, attention_output_dir, example_idx, model_obj, tokenizer
                    )
                    if attention_analysis_dir:
                        # Load and display summary of attention results
                        try:
                            results_file = os.path.join(attention_analysis_dir, "top_facts.json")
                            if os.path.exists(results_file):
                                with open(results_file, 'r') as f:
                                    results = json.load(f)
                                
                                num_facts = len(results.get('retrieved_facts', []))
                                echo(log_fp, f"✅ Attention analysis complete: {num_facts} facts extracted")
                                
                                # Show top 3 attention scores
                                facts = results.get('retrieved_facts', [])[:3]
                                if facts:
                                    echo(log_fp, f"🔍 Top attention scores:")
                                    for i, fact in enumerate(facts, 1):
                                        score = fact.get('attention_score', 0)
                                        text_preview = fact.get('text', 'No text')[:60] + "..."
                                        echo(log_fp, f"   {i}. Score: {score:.8f} - {text_preview}")
                                else:
                                    echo(log_fp, f"⚠️  No attention facts extracted")
                            else:
                                echo(log_fp, f"⚠️  Attention results file not found: {results_file}")
                        except Exception as e:
                            echo(log_fp, f"⚠️  Error reading attention results: {e}")
                    else:
                        echo(log_fp, f"❌ Attention analysis failed for example {example_idx}")
                
                if attention_analysis_dir:
                    attention_results.append({
                        "example_idx": example_idx,
                        "attention_analysis_dir": os.path.relpath(attention_analysis_dir, os.path.dirname(log_filename)),
                        "has_attention_analysis": True
                    })
                
                echo(log_fp, 
                    '-' * 30 + f' {correct=} {total=} {parse_failures=} {prediction=} {y=} {is_correct=} ' + '-' * 30)
                if args.delay is not None and args.delay > 0:
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
