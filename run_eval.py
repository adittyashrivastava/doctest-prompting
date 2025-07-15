import sys
# Force stdout to use UTF-8 in Python 3.7+
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from contextlib import redirect_stdout
import json
import re
import time
import os
import time
import traceback
from datetime import datetime

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

def save_run_summary(log_filename, model_name, task_name, method_name, correct, total, parse_failures, attention_enabled=False):
    """Save run summary to a central summary log file"""
    try:
        # Create summary directory if it doesn't exist
        log_dir = os.path.dirname(log_filename)
        summary_dir = os.path.join(log_dir, "run_summaries")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Create summary file path
        summary_file = os.path.join(summary_dir, "evaluation_summary.jsonl")
        
        # Calculate accuracies
        accuracy = correct / total if total > 0 else 0.0
        valid_examples = total - parse_failures
        accuracy_no_parse = correct / valid_examples if valid_examples > 0 else 0.0
        
        # Create summary entry
        summary_entry = {
            "timestamp": datetime.now().isoformat(),
            "log_file": os.path.basename(log_filename),
            "model_name": model_name,
            "task_name": task_name,
            "method_name": method_name,
            "total_examples": total,
            "correct_examples": correct,
            "parse_failures": parse_failures,
            "accuracy": accuracy,
            "accuracy_no_parse_failures": accuracy_no_parse,
            "attention_analysis_enabled": attention_enabled,
            "run_status": "completed"
        }
        
        # Append to summary file
        with open(summary_file, "a") as f:
            f.write(json.dumps(summary_entry) + "\n")
        
        print(f"📊 Run summary saved to: {summary_file}")
        print(f"📈 Final accuracy: {accuracy:.3f} ({correct}/{total})")
        if parse_failures > 0:
            print(f"📈 Accuracy (no parse failures): {accuracy_no_parse:.3f} ({correct}/{valid_examples})")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not save run summary: {e}")

def extract_method_calls_from_trace(trace_text):
    """Extract method calls and their context from program trace"""
    method_calls = []
    lines = trace_text.split('\n')
    
    for i, line in enumerate(lines):
        # Look for function calls in the trace
        if '(' in line and ')' in line and '->' in line:
            # This looks like a function call with return value
            method_calls.append({
                'line_num': i,
                'call_text': line.strip(),
                'context_before': lines[max(0, i-2):i],
                'context_after': lines[i+1:min(len(lines), i+3)]
            })
        elif line.strip().startswith('Calling ') and '...' in line:
            # Alternative format: "Calling function_name(args)..."
            method_calls.append({
                'line_num': i,
                'call_text': line.strip(),
                'context_before': lines[max(0, i-2):i],
                'context_after': lines[i+1:min(len(lines), i+3)]
            })
    
    return method_calls

def extract_thoughts_from_response_with_attention(response, method_calls, tokenizer, attention_weights, input_ids, prompt=None):
    """Extract program traces/thoughts from model response with real attention scores and token positions"""
    thoughts = []
    
    if attention_weights is None or input_ids is None:
        print("⚠️  No attention weights available for thoughts analysis")
        return thoughts
    
    try:
        # Calculate prompt length for better attention computation
        if prompt is not None:
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
            prompt_length = len(prompt_tokens)
        else:
            prompt_length = None
        
        # The thoughts are the actual program traces - use the method calls that were already extracted
        # These represent the reasoning steps/simulation from PTP
        
        for method_call in method_calls:
            call_text = method_call['call_text']
            
            # Find the real token positions of this method call in the response
            token_start, token_end = find_token_positions(call_text, response, tokenizer)
            
            # Compute attention scores for this thought using the real token positions
            attention_score = compute_attention_score_for_thought_at_positions(
                token_start, token_end, attention_weights, input_ids, tokenizer, prompt_length
            )
            
            # Create a thought entry with real attention score and token positions
            thought = {
                'text': call_text,
                'score': attention_score,
                'attention_score': attention_score,
                'frequency': 1.0,  # Each thought appears once
                'token_start': token_start,
                'token_end': token_end,
                'type': 'program_trace',
                'line_num': method_call['line_num'],
                'context_before': method_call.get('context_before', []),
                'context_after': method_call.get('context_after', [])
            }
            thoughts.append(thought)
        
        # Sort by attention score and return top-k
        thoughts.sort(key=lambda x: x['attention_score'], reverse=True)
        return thoughts[:10]  # Return top 10 thoughts
        
    except Exception as e:
        print(f"❌ Error computing attention for thoughts: {e}")
        return thoughts


def find_token_positions(text, full_response, tokenizer):
    """Find the real token start and end positions of text within the full response"""
    try:
        # Find character positions first
        char_start = full_response.find(text)
        if char_start == -1:
            # Try a more flexible search
            text_words = text.split()
            if len(text_words) > 0:
                first_word = text_words[0]
                char_start = full_response.find(first_word)
                if char_start == -1:
                    return 0, 1  # Default fallback
            else:
                return 0, 1
        
        char_end = char_start + len(text)
        
        # Use return_offsets_mapping for precise character-to-token mapping
        encoding = tokenizer(full_response, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']
        
        # Find tokens that overlap with our text span
        token_start_idx = 0
        token_end_idx = 1
        
        for i, (start_char, end_char) in enumerate(offsets):
            # Check if this token overlaps with our text
            if start_char <= char_start < end_char:
                token_start_idx = i
            if start_char < char_end <= end_char:
                token_end_idx = i + 1
                break
            elif start_char >= char_end:
                token_end_idx = i
                break
        
        # Ensure we have a valid range
        if token_end_idx <= token_start_idx:
            token_end_idx = token_start_idx + 1
            
        return token_start_idx, token_end_idx
        
    except Exception as e:
        print(f"❌ Error finding token positions: {e}")
        return 0, 1

def compute_attention_score_for_thought_at_positions(token_start, token_end, attention_weights, input_ids, tokenizer, prompt_length=None):
    """Compute real attention score for a thought using actual token positions"""
    try:
        if not attention_weights or not input_ids.size():
            return 0.0
        
        # Get the sequence length
        seq_len = input_ids.size(1)
        
        # Ensure token positions are within bounds
        token_start = max(0, min(token_start, seq_len - 1))
        token_end = max(token_start + 1, min(token_end, seq_len))
        
        # Use actual prompt length if provided, otherwise estimate
        if prompt_length is not None:
            input_length = min(prompt_length, seq_len)
        else:
            input_length = seq_len // 2  # Fallback estimate
        
        # Compute attention scores properly
        total_attention = 0.0
        num_valid_positions = 0
        
        for layer_attention in attention_weights:
            # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
            batch_size, num_heads, seq_len_att, _ = layer_attention.shape
            
            # Average across heads to get [batch_size, seq_len, seq_len]
            avg_attention = layer_attention.mean(dim=1)
            
            # For each position in the thought span
            for pos in range(token_start, min(token_end, seq_len_att)):
                # Get attention from this position to the input (prompt) positions
                attention_to_input = avg_attention[0, pos, :input_length]
                
                # Use max attention instead of sum to match ATTRIEVAL-style normalization
                # This prevents artificially inflated scores from summing across all input tokens
                if attention_to_input.numel() > 0:
                    max_attention = attention_to_input.max().item()
                    # Only count positions with significant attention
                    if max_attention > 1e-6:
                        total_attention += max_attention
                        num_valid_positions += 1
        
        # Normalize by number of valid positions
        if num_valid_positions > 0:
            total_attention /= num_valid_positions
        
        return total_attention
        
    except Exception as e:
        print(f"❌ Error in attention computation: {e}")
        return 0.0

def compute_attention_score_for_thought(thought_tokens, attention_weights, input_ids, tokenizer):
    """Compute real attention score for a thought using attention weights (legacy function)"""
    # This function is kept for backward compatibility
    return compute_attention_score_for_thought_at_positions(0, 1, attention_weights, input_ids, tokenizer)

def analyze_method_attention_on_thoughts(method_calls, thoughts, facts, attention_weights=None):
    """Analyze which thoughts and facts each method call might be related to"""
    method_attention = {}
    
    for i, method_call in enumerate(method_calls):
        method_name = extract_method_name(method_call['call_text'])
        
        # Find the corresponding thought for this method call
        # Since thoughts are now the actual method calls, we find the matching one
        matching_thought = None
        for thought in thoughts:
            if thought['text'] == method_call['call_text']:
                matching_thought = thought
                break
        
        # Find facts related to this method
        related_facts = []
        for fact in facts:
            if method_name.lower() in fact.get('text', '').lower():
                related_facts.append(fact)
        
        method_attention[str(i)] = {
            'method_name': method_name,
            'method_call_text': method_call['call_text'],
            'matching_thought': matching_thought,
            'related_facts': related_facts,
            'thought_attention_score': matching_thought['attention_score'] if matching_thought else 0.0,
            'fact_attention_score': sum(f.get('attention_score', f.get('score', 0)) for f in related_facts),
            'total_attention_score': (
                (matching_thought['attention_score'] if matching_thought else 0.0) + 
                sum(f.get('attention_score', f.get('score', 0)) for f in related_facts)
            ),
            'token_positions': {
                'start': matching_thought['token_start'] if matching_thought else 0,
                'end': matching_thought['token_end'] if matching_thought else 0
            }
        }
    
    return method_attention

def perform_attention_analysis(prompt, response, input_context, target, output_dir, example_idx, model_obj, tokenizer):
    """Perform attention analysis for a single example with enhanced thought analysis"""
    if not ATTENTION_VIZ_AVAILABLE or model_obj is None:
        return None

    try:
        print(f"🔍 Analyzing example {example_idx}...")

        # Step 1: Extract method calls from the response for separate analysis
        method_calls = extract_method_calls_from_trace(response)
        print(f"🔍 Found {len(method_calls)} method calls in trace")

        # Step 2: Compact the input to reduce memory usage
        print(f"🔍 Step 2: Compacting input for memory efficiency...")
        
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

        # Step 3: Layer-selective attention extraction for memory efficiency
        print(f"🔍 Step 3: Layer-selective attention extraction...")
        
        # Save original model settings
        original_output_attentions = getattr(model_obj.config, 'output_attentions', False)
        original_attn_implementation = getattr(model_obj.config, '_attn_implementation', None)
        original_use_cache = getattr(model_obj.config, 'use_cache', True)

        attention_weights = None
        try:
            # Configure for attention extraction
            model_obj.config.output_attentions = True
            model_obj.config._attn_implementation = 'eager'
            model_obj.config.use_cache = False
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(model_obj, 'gradient_checkpointing_enable'):
                model_obj.gradient_checkpointing_enable()

            # Extract attention weights
            with torch.no_grad():
                outputs = model_obj(**inputs, output_attentions=True)
                
                # Move attention weights to CPU immediately to free GPU memory
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    # Only keep the last 25% of layers for memory efficiency
                    num_layers = len(outputs.attentions)
                    start_layer = int(num_layers * 0.75)
                    
                    attention_weights = []
                    for i in range(start_layer, num_layers):
                        # Move to CPU and convert to float32 for numerical stability
                        attn_cpu = outputs.attentions[i].cpu().float()
                        attention_weights.append(attn_cpu)
                    
                    print(f"✅ Extracted attention from {len(attention_weights)} layers (last 25%)")
                else:
                    print("❌ No attention weights extracted")
                    attention_weights = None

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

        # Step 4: Extract thoughts from response using real attention scores
        print(f"🔍 Step 4: Extracting thoughts with real attention scores...")
        retrieved_thoughts = extract_thoughts_from_response_with_attention(
            response, method_calls, tokenizer, attention_weights, inputs['input_ids'], compact_prompt
        )
        print(f"🧠 Extracted {len(retrieved_thoughts)} thoughts with attention scores")

        # Step 5: Initialize ATTRIEVAL for facts extraction
        print(f"🔍 Step 5: Setting up ATTRIEVAL for facts extraction...")
        
        extractor = AttentionExtractor(model_obj, tokenizer)
        config = AttrievelConfig(
            layer_fraction=0.25,      # Use last 25% of layers (matches our extraction)
            top_k=10,                 # Increase to get more facts
            frequency_threshold=0.95, # Lower threshold to get more facts
            max_facts=10              # Increase to get more facts
        )
        retriever = AttrievelRetriever(extractor, config)

        # Step 6: Run ATTRIEVAL fact retrieval
        print(f"🔍 Step 6: Running ATTRIEVAL for facts extraction...")
        
        try:
            # Use original input_context (not compacted) for fact retrieval
            retrieval_result = retriever.retrieve_facts(
                context=input_context,
                question=input_context,  # For doctest problems, question is same as context
                cot_response=response,
                use_cross_evaluation=False  # Disable cross-evaluation to save memory
            )
            
            retrieved_facts = retrieval_result.get('retrieved_facts', [])
            print(f"✅ ATTRIEVAL completed - {len(retrieved_facts)} facts extracted")
            
        except Exception as e:
            print(f"❌ ATTRIEVAL failed: {e}")
            # Create minimal fallback result
            retrieved_facts = []

        # Step 7: Analyze method attention on both facts and thoughts
        print(f"🔍 Step 7: Analyzing method attention patterns...")
        
        method_attention = analyze_method_attention_on_thoughts(
            method_calls, retrieved_thoughts, retrieved_facts, attention_weights
        )
        
        # Step 8: Process and clean results
        print(f"🔍 Step 8: Processing results...")
        
        # Debug first few facts and thoughts
        for i, fact in enumerate(retrieved_facts[:3]):
            score = fact.get('score', 'No score')
            text = fact.get('text', 'No text')[:30]
            print(f"   Fact {i}: score={score}, text='{text}...'")
            
        for i, thought in enumerate(retrieved_thoughts[:3]):
            score = thought.get('attention_score', 'No score')
            text = thought.get('text', 'No text')[:30]
            print(f"   Thought {i}: attention_score={score}, text='{text}...'")

        # Clean up scores for both facts and thoughts
        cleaned_facts = []
        for fact in retrieved_facts:
            score = fact.get('score', 0)
            if isinstance(score, (int, float)) and not (score != score or score == float('inf') or score == float('-inf')):
                cleaned_facts.append(fact)
            else:
                fact_copy = fact.copy()
                fact_copy['score'] = 0.0
                cleaned_facts.append(fact_copy)
        
        cleaned_thoughts = []
        for thought in retrieved_thoughts:
            score = thought.get('attention_score', 0)
            if isinstance(score, (int, float)) and not (score != score or score == float('inf') or score == float('-inf')):
                cleaned_thoughts.append(thought)
            else:
                thought_copy = thought.copy()
                thought_copy['attention_score'] = 0.0
                cleaned_thoughts.append(thought_copy)

        print(f"✅ Final results: {len(cleaned_facts)} facts, {len(cleaned_thoughts)} thoughts")

        # Step 9: Save results with enhanced analysis
        example_dir = os.path.join(output_dir, f"example_{example_idx:04d}")
        os.makedirs(example_dir, exist_ok=True)

        # Save comprehensive results
        comprehensive_results = {
            "example_idx": example_idx,
            "input": input_context,
            "response": response,
            "target": target,
            "retrieved_facts": cleaned_facts,
            "retrieved_thoughts": cleaned_thoughts,
            "method_calls": method_calls,
            "method_call_attention": method_attention,
            "num_facts": len(cleaned_facts),
            "num_thoughts": len(cleaned_thoughts),
            "analysis_timestamp": datetime.now().isoformat()
        }

        with open(os.path.join(example_dir, "comprehensive_analysis.json"), "w") as f:
            json.dump(comprehensive_results, f, indent=2)

        # Save method-specific attention visualization
        method_attention_viz = create_method_attention_visualization(
            method_calls, method_attention, input_context, cleaned_facts, cleaned_thoughts
        )
        
        with open(os.path.join(example_dir, "method_attention_visualization.json"), "w") as f:
            json.dump(method_attention_viz, f, indent=2)

        print(f"✅ Comprehensive analysis saved to: {example_dir}")
        return example_dir

    except Exception as e:
        print(f"❌ Analysis failed for example {example_idx}: {e}")
        traceback.print_exc()
        return None

def create_method_attention_visualization(method_calls, method_attention, input_context, facts, thoughts):
    """Create a visualization of which input sentences each method call attends to"""
    visualization = {
        "summary": {
            "total_method_calls": len(method_calls),
            "input_context_length": len(input_context),
            "total_facts": len(facts),
            "total_thoughts": len(thoughts)
        },
        "method_call_analysis": []
    }
    
    # Split input context into sentences for analysis
    input_sentences = [s.strip() for s in input_context.split('.') if s.strip()]
    
    for i, method_call in enumerate(method_calls):
        method_name = extract_method_name(method_call['call_text'])
        
        # Get attention data for this method call
        attention_data = method_attention.get(str(i), {})
        
        # Get matching thought and facts from the attention analysis
        matching_thought = attention_data.get('matching_thought')
        related_facts = attention_data.get('related_facts', [])
        
        # Find top attended sentences using actual attention computation
        top_sentences = []
        
        # Get method call token positions
        method_call_text = method_call['call_text']
        
        # Try to find this method call in the full input context to get attention
        # For now, use a simplified approach based on related facts
        # This is a limitation - we need the actual attention weights here
        for j, sentence in enumerate(input_sentences):
            # Calculate attention score based on related facts that mention this sentence
            attention_score = 0.0
            sentence_lower = sentence.lower()
            
            # Check if any facts reference this sentence
            for fact in related_facts:
                fact_text = fact.get('text', '').lower()
                if sentence_lower in fact_text or any(word in fact_text for word in sentence_lower.split()[:3]):
                    attention_score += fact.get('attention_score', fact.get('score', 0))
            
            # Also check if the method name appears in the sentence
            if method_name.lower() in sentence_lower:
                attention_score += 0.1  # Base score for method relevance
            
            if attention_score > 0:
                top_sentences.append({
                    "sentence_idx": j,
                    "sentence": sentence,
                    "attention_score": attention_score
                })
        
        # Sort by attention score
        top_sentences.sort(key=lambda x: x['attention_score'], reverse=True)
        
        method_analysis = {
            "method_call_idx": i,
            "method_name": method_name,
            "method_call_text": method_call['call_text'],
            "top_attended_sentences": top_sentences[:5],  # Top 5 sentences
            "related_facts": related_facts[:3],  # Top 3 related facts
            "matching_thought": matching_thought,  # The program trace for this method call
            "attention_summary": {
                "total_attention_score": attention_data.get('total_attention_score', 0),
                "thought_attention_score": attention_data.get('thought_attention_score', 0),
                "fact_attention_score": attention_data.get('fact_attention_score', 0),
                "num_attended_sentences": len(top_sentences),
                "has_matching_thought": matching_thought is not None,
                "num_related_facts": len(related_facts),
                "max_attention_score": max([s['attention_score'] for s in top_sentences], default=0),
                "token_positions": attention_data.get('token_positions', {'start': 0, 'end': 0})
            }
        }
        
        visualization["method_call_analysis"].append(method_analysis)
    
    return visualization

def extract_method_name(call_text):
    """Extract method name from call text"""
    import re
    # Try to extract function name from various formats
    patterns = [
        r'(\w+)\(',  # function_name(
        r'Calling (\w+)\(',  # Calling function_name(
        r'\.\.\.(\w+) returned',  # ...function_name returned
    ]
    
    for pattern in patterns:
        match = re.search(pattern, call_text)
        if match:
            return match.group(1)
    
    return "unknown_method"

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
            
            # Extract model and task information for summary
            model_name = getattr(args, 'model', 'unknown_model')
            task_name = getattr(args, 'task', 'unknown_task')
            method_name = "PTP"  # Assuming PTP method
            
            for example_idx, ex in enumerate(arg_util.active_subset(args, examples)):
                try:
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
                                results_file = os.path.join(attention_analysis_dir, "comprehensive_analysis.json")
                                if os.path.exists(results_file):
                                    with open(results_file, 'r') as f:
                                        results = json.load(f)
                                    
                                    num_facts = len(results.get('retrieved_facts', []))
                                    num_thoughts = len(results.get('retrieved_thoughts', []))
                                    num_methods = len(results.get('method_calls', []))
                                    
                                    echo(log_fp, f"✅ Attention analysis complete:")
                                    echo(log_fp, f"   📊 Facts extracted: {num_facts}")
                                    echo(log_fp, f"   🧠 Thoughts extracted: {num_thoughts}")
                                    echo(log_fp, f"   🎯 Method calls found: {num_methods}")
                                    
                                    # Show top 3 attention scores from facts
                                    facts = results.get('retrieved_facts', [])[:3]
                                    if facts:
                                        echo(log_fp, f"🔍 Top attention scores (facts):")
                                        for i, fact in enumerate(facts, 1):
                                            score = fact.get('attention_score', fact.get('score', 0))
                                            text_preview = fact.get('text', 'No text')[:60] + "..."
                                            echo(log_fp, f"   {i}. Score: {score:.8f} - {text_preview}")
                                    
                                    # Show top 3 attention scores from thoughts
                                    thoughts = results.get('retrieved_thoughts', [])[:3]
                                    if thoughts:
                                        echo(log_fp, f"🔍 Top attention scores (thoughts):")
                                        for i, thought in enumerate(thoughts, 1):
                                            score = thought.get('attention_score', thought.get('score', 0))
                                            text_preview = thought.get('text', 'No text')[:60] + "..."
                                            echo(log_fp, f"   {i}. Score: {score:.8f} - {text_preview}")
                                    
                                    if not facts and not thoughts:
                                        echo(log_fp, f"⚠️  No attention facts or thoughts extracted")
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

                except Exception as e:
                    echo(log_fp, f"❌ Error processing example {example_idx}: {e}")
                    echo(log_fp, f"Traceback: {traceback.format_exc()}")
                    echo(log_fp, f"Continuing to next example...")
                    continue
            
            # At the end of main, save the run summary
            try:
                save_run_summary(
                    log_filename=log_filename,
                    model_name=model_name,
                    task_name=task_name,
                    method_name=method_name,
                    correct=correct,
                    total=total,
                    parse_failures=parse_failures,
                    attention_enabled=attention_enabled
                )
            except Exception as e:
                print(f"Warning: Could not save run summary: {e}")

            try:
                if parse_failures:
                    parsed = total - parse_failures
                    if parsed > 0:
                        acc = correct / parsed
                        echo(log_fp, f'Final totals (ignoring parse failures) {correct=} {parsed=} {acc=}')
                    else:
                        echo(log_fp, f'Final totals (ignoring parse failures) {correct=} {parsed=} acc=0.0 (all examples failed to parse)')
                
                if total > 0:
                    acc = correct / total
                    echo(log_fp, f'Final totals {correct=} {total=} {acc=}')
                else:
                    echo(log_fp, f'Final totals {correct=} {total=} acc=0.0 (no examples processed)')
                    
            except ZeroDivisionError as e:
                echo(log_fp, f'Error calculating accuracy: {e}')
                echo(log_fp, f'Final totals {correct=} {total=} {parse_failures=} - cannot calculate accuracy')
            
            # Report attention analysis summary
            if attention_enabled and attention_results:
                echo(log_fp, "=" * 30 + "Attention Analysis Summary" + "=" * 30)
                echo(log_fp, f"Attention analysis completed for {len(attention_results)}/{total} examples")
                echo(log_fp, f"Results saved to: {attention_output_dir}")

if __name__ == "__main__":
    main(None)
