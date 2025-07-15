#!/usr/bin/env python3
"""
Fix for attention analysis bugs.

Issues identified:
1. Attention scores for thoughts always 1.0 - flawed computation
2. Top attended sentences always empty - broken heuristic
3. Token position finding broken - unreliable mapping
4. Model predictions failing - needs investigation

This script provides fixed versions of the key functions.
"""

import torch
import numpy as np
import re
from typing import List, Dict, Tuple, Optional

def find_token_positions_fixed(text: str, full_response: str, tokenizer) -> Tuple[int, int]:
    """
    Fixed version of token position finding.
    
    The original version had issues with character-to-token mapping.
    This version uses a more reliable approach.
    """
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
        
        # Tokenize the full response and get character spans
        # Use return_offsets_mapping to get precise character-to-token mapping
        encoding = tokenizer(full_response, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']
        
        # Find tokens that overlap with our text span
        token_start_idx = 0
        token_end_idx = 0
        
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
        print(f"❌ Error in fixed token position finding: {e}")
        return 0, 1

def compute_attention_score_fixed(token_start: int, token_end: int, 
                                 attention_weights: List[torch.Tensor], 
                                 input_ids: torch.Tensor, 
                                 prompt_length: int) -> float:
    """
    Fixed version of attention score computation.
    
    The original version had wrong assumptions about input length and
    incorrect normalization.
    """
    try:
        if not attention_weights or not input_ids.size():
            return 0.0
        
        # Get the sequence length
        seq_len = input_ids.size(1)
        
        # Ensure token positions are within bounds
        token_start = max(0, min(token_start, seq_len - 1))
        token_end = max(token_start + 1, min(token_end, seq_len))
        
        # Use the actual prompt length instead of seq_len // 2
        # This is passed in from the tokenization step
        input_length = min(prompt_length, seq_len)
        
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
                
                # Sum attention to input, excluding padding and special tokens
                valid_attention = attention_to_input[attention_to_input > 1e-6]  # Filter tiny values
                if len(valid_attention) > 0:
                    total_attention += valid_attention.sum().item()
                    num_valid_positions += 1
        
        # Normalize by number of valid positions
        if num_valid_positions > 0:
            total_attention /= num_valid_positions
        
        return total_attention
        
    except Exception as e:
        print(f"❌ Error in fixed attention computation: {e}")
        return 0.0

def extract_sentences_with_attention_fixed(input_context: str, 
                                          attention_weights: List[torch.Tensor],
                                          input_ids: torch.Tensor,
                                          tokenizer,
                                          method_call_tokens: Tuple[int, int]) -> List[Dict]:
    """
    Fixed version of sentence extraction with attention.
    
    The original version used a broken heuristic. This version actually
    computes attention from method call tokens to input sentences.
    """
    try:
        # Split input into sentences more robustly
        sentences = []
        
        # Use multiple sentence boundary markers
        sentence_boundaries = r'[.!?]+\s+'
        raw_sentences = re.split(sentence_boundaries, input_context)
        
        current_pos = 0
        for sent in raw_sentences:
            sent = sent.strip()
            if sent:  # Skip empty sentences
                start_pos = input_context.find(sent, current_pos)
                if start_pos != -1:
                    sentences.append({
                        'text': sent,
                        'char_start': start_pos,
                        'char_end': start_pos + len(sent)
                    })
                    current_pos = start_pos + len(sent)
        
        # Get tokenization with character offsets for the input
        input_encoding = tokenizer(input_context, return_offsets_mapping=True, add_special_tokens=False)
        input_offsets = input_encoding['offset_mapping']
        
        # Calculate attention scores for each sentence
        sentence_attentions = []
        method_start, method_end = method_call_tokens
        
        for sentence in sentences:
            # Find tokens that correspond to this sentence
            sentence_tokens = []
            for i, (start_char, end_char) in enumerate(input_offsets):
                if (start_char >= sentence['char_start'] and 
                    start_char < sentence['char_end']):
                    sentence_tokens.append(i)
            
            if not sentence_tokens:
                continue
                
            # Compute attention from method call to this sentence
            total_attention = 0.0
            num_layers = len(attention_weights)
            
            for layer_attention in attention_weights:
                # Average across heads
                avg_attention = layer_attention.mean(dim=1)  # [batch, seq, seq]
                
                # Average attention from method call tokens to sentence tokens
                for method_pos in range(method_start, min(method_end, avg_attention.size(1))):
                    for sent_token in sentence_tokens:
                        if sent_token < avg_attention.size(2):
                            total_attention += avg_attention[0, method_pos, sent_token].item()
            
            # Normalize by number of layers and token pairs
            if num_layers > 0 and len(sentence_tokens) > 0:
                total_attention /= (num_layers * len(sentence_tokens) * (method_end - method_start))
            
            sentence_attentions.append({
                'sentence': sentence['text'],
                'sentence_idx': len(sentence_attentions),
                'attention_score': total_attention,
                'num_tokens': len(sentence_tokens)
            })
        
        # Sort by attention score
        sentence_attentions.sort(key=lambda x: x['attention_score'], reverse=True)
        
        return sentence_attentions
        
    except Exception as e:
        print(f"❌ Error in fixed sentence extraction: {e}")
        return []

def debug_attention_analysis(attention_dir: str, example_idx: int = 0):
    """Debug the attention analysis for a specific example."""
    import json
    import os
    
    example_dir = os.path.join(attention_dir, f"example_{example_idx:04d}")
    
    if not os.path.exists(example_dir):
        print(f"❌ Example directory not found: {example_dir}")
        return
    
    # Load attention results
    attention_file = os.path.join(example_dir, "attention_results.json")
    if not os.path.exists(attention_file):
        print(f"❌ Attention results file not found: {attention_file}")
        return
    
    with open(attention_file, 'r') as f:
        attention_data = json.load(f)
    
    print(f"🔍 DEBUGGING EXAMPLE {example_idx}")
    print("="*60)
    
    # Debug thoughts
    thoughts = attention_data.get('retrieved_thoughts', [])
    print(f"📝 Thoughts: {len(thoughts)}")
    for i, thought in enumerate(thoughts[:3]):
        score = thought.get('attention_score', 'N/A')
        text = thought.get('text', 'N/A')[:50]
        token_start = thought.get('token_start', 'N/A')
        token_end = thought.get('token_end', 'N/A')
        print(f"  {i+1}. Score: {score}, Tokens: {token_start}-{token_end}")
        print(f"      Text: {text}...")
    
    # Debug facts
    facts = attention_data.get('retrieved_facts', [])
    print(f"\n💡 Facts: {len(facts)}")
    for i, fact in enumerate(facts[:3]):
        score = fact.get('attention_score', fact.get('score', 'N/A'))
        text = fact.get('text', 'N/A')[:50]
        print(f"  {i+1}. Score: {score}")
        print(f"      Text: {text}...")
    
    # Debug method attention
    method_file = os.path.join(example_dir, "method_attention.json")
    if os.path.exists(method_file):
        with open(method_file, 'r') as f:
            method_data = json.load(f)
        
        print(f"\n🎯 Method Attention:")
        for method_idx, method_info in method_data.items():
            method_name = method_info.get('method_name', 'unknown')
            total_score = method_info.get('total_attention_score', 0)
            thought_score = method_info.get('thought_attention_score', 0)
            fact_score = method_info.get('fact_attention_score', 0)
            
            print(f"  Method {method_idx}: {method_name}")
            print(f"    Total: {total_score}, Thought: {thought_score}, Fact: {fact_score}")
    
    # Debug visualization
    viz_file = os.path.join(example_dir, "method_attention_visualization.json")
    if os.path.exists(viz_file):
        with open(viz_file, 'r') as f:
            viz_data = json.load(f)
        
        print(f"\n📊 Visualization Data:")
        for method_analysis in viz_data.get('method_call_analysis', []):
            method_name = method_analysis.get('method_name', 'unknown')
            top_sentences = method_analysis.get('top_attended_sentences', [])
            
            print(f"  Method: {method_name}")
            print(f"    Top sentences: {len(top_sentences)}")
            for i, sent in enumerate(top_sentences[:2]):
                score = sent.get('attention_score', 0)
                text = sent.get('sentence', 'N/A')[:40]
                print(f"      {i+1}. [{score:.6f}] {text}...")

def check_prediction_accuracy(log_dir: str):
    """Check why model predictions are failing."""
    import json
    import os
    
    # Look for the results file
    results_file = None
    for file in os.listdir(log_dir):
        if file.endswith('.jsonl') and 'traces' in file:
            results_file = os.path.join(log_dir, file)
            break
    
    if not results_file:
        print("❌ No results file found")
        return
    
    print(f"🔍 CHECKING PREDICTION ACCURACY")
    print("="*60)
    print(f"📁 Results file: {results_file}")
    
    correct = 0
    total = 0
    parse_failures = 0
    
    with open(results_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    result = json.loads(line.strip())
                    total += 1
                    
                    is_correct = result.get('is_correct', False)
                    prediction = result.get('prediction', 'N/A')
                    target = result.get('target', 'N/A')
                    
                    if is_correct:
                        correct += 1
                    
                    if prediction == '**parse failed**':
                        parse_failures += 1
                    
                    # Show first few examples
                    if line_num <= 5:
                        status = "✅" if is_correct else "❌"
                        print(f"  {line_num}. {status} Pred: {prediction}, Target: {target}")
                        
                except json.JSONDecodeError:
                    print(f"  ❌ Line {line_num}: Invalid JSON")
    
    print(f"\n📊 Summary:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {correct/total*100:.1f}% ({correct}/{total})")
    print(f"  Parse failures: {parse_failures}")
    
    if correct == 0:
        print(f"\n🚨 ALL PREDICTIONS FAILED!")
        print(f"  This suggests a fundamental issue with:")
        print(f"  1. Model configuration")
        print(f"  2. Prompt template")
        print(f"  3. Output parsing")
        print(f"  4. Task setup")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix attention analysis bugs")
    parser.add_argument("--debug-attention", help="Debug attention for specific example")
    parser.add_argument("--check-accuracy", help="Check prediction accuracy in log directory")
    parser.add_argument("--example", type=int, default=0, help="Example index to debug")
    
    args = parser.parse_args()
    
    if args.debug_attention:
        debug_attention_analysis(args.debug_attention, args.example)
    
    if args.check_accuracy:
        check_prediction_accuracy(args.check_accuracy)
        
    if not args.debug_attention and not args.check_accuracy:
        print("🔧 ATTENTION ANALYSIS FIXES AVAILABLE")
        print("="*60)
        print("This script provides fixes for:")
        print("1. ✅ Fixed token position finding")
        print("2. ✅ Fixed attention score computation")
        print("3. ✅ Fixed sentence extraction with attention")
        print("4. ✅ Debug tools for diagnosis")
        print()
        print("Usage:")
        print("  python fix_attention_analysis.py --debug-attention logs/.../attention_analysis --example 0")
        print("  python fix_attention_analysis.py --check-accuracy logs/.../") 