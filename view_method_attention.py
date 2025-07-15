#!/usr/bin/env python3
"""
Simple Method Call Attention Viewer

This script provides an easy-to-understand view of which input sentences
each method call in the program trace attends to.
"""

import json
import os
import argparse

def load_attention_data(attention_dir, example_idx=None):
    """Load attention analysis data"""
    if example_idx is not None:
        # Load specific example
        example_dir = os.path.join(attention_dir, f"example_{example_idx:04d}")
        viz_file = os.path.join(example_dir, "method_attention_visualization.json")
        
        if os.path.exists(viz_file):
            with open(viz_file, 'r') as f:
                return [json.load(f)]
        else:
            return []
    else:
        # Load all examples
        examples = []
        example_dirs = [d for d in os.listdir(attention_dir) if d.startswith('example_')]
        example_dirs.sort()
        
        for example_dir in example_dirs:
            viz_file = os.path.join(attention_dir, example_dir, "method_attention_visualization.json")
            if os.path.exists(viz_file):
                with open(viz_file, 'r') as f:
                    examples.append(json.load(f))
        
        return examples

def display_method_attention(examples, show_detailed=False):
    """Display method attention patterns in a readable format"""
    for i, example in enumerate(examples):
        print(f"\n{'='*60}")
        print(f"📋 EXAMPLE {i}")
        print(f"{'='*60}")
        
        summary = example.get('summary', {})
        print(f"📊 Summary:")
        print(f"   • Method calls: {summary.get('total_method_calls', 0)}")
        print(f"   • Input length: {summary.get('input_context_length', 0)} chars")
        print(f"   • Facts found: {summary.get('total_facts', 0)}")
        print(f"   • Thoughts found: {summary.get('total_thoughts', 0)}")
        
        for method_analysis in example.get('method_call_analysis', []):
            method_name = method_analysis.get('method_name', 'unknown')
            call_text = method_analysis.get('method_call_text', 'unknown')
            
            print(f"\n🎯 METHOD: {method_name}")
            print(f"   Call: {call_text}")
            
            # Show top attended sentences
            top_sentences = method_analysis.get('top_attended_sentences', [])
            if top_sentences:
                print(f"   📝 What it's looking at:")
                for j, sentence_info in enumerate(top_sentences[:3], 1):
                    score = sentence_info.get('attention_score', 0)
                    sentence = sentence_info.get('sentence', 'No text')
                    
                    # Truncate long sentences
                    if len(sentence) > 80:
                        sentence = sentence[:77] + "..."
                    
                    print(f"      {j}. [{score:.6f}] \"{sentence}\"")
            else:
                print(f"   📝 No significant attention found")
            
            # Show related facts and thoughts
            if show_detailed:
                related_facts = method_analysis.get('related_facts', [])
                if related_facts:
                    print(f"   💡 Related facts:")
                    for j, fact in enumerate(related_facts[:2], 1):
                        score = fact.get('attention_score', fact.get('score', 0))
                        text = fact.get('text', 'No text')
                        if len(text) > 60:
                            text = text[:57] + "..."
                        print(f"      {j}. [{score:.6f}] \"{text}\"")
                
                matching_thought = method_analysis.get('matching_thought')
                if matching_thought:
                    print(f"   🧠 Program Trace:")
                    score = matching_thought.get('attention_score', 0)
                    text = matching_thought.get('text', 'No text')
                    if len(text) > 60:
                        text = text[:57] + "..."
                    token_start = matching_thought.get('token_start', 0)
                    token_end = matching_thought.get('token_end', 0)
                    print(f"      [{score:.6f}] \"{text}\" (tokens: {token_start}-{token_end})")
            
            # Show enhanced attention summary
            attention_summary = method_analysis.get('attention_summary', {})
            total_score = attention_summary.get('total_attention_score', 0)
            thought_score = attention_summary.get('thought_attention_score', 0)
            fact_score = attention_summary.get('fact_attention_score', 0)
            num_sentences = attention_summary.get('num_attended_sentences', 0)
            has_thought = attention_summary.get('has_matching_thought', False)
            num_facts = attention_summary.get('num_related_facts', 0)
            max_score = attention_summary.get('max_attention_score', 0)
            
            print(f"   📊 Attention Summary:")
            print(f"      • Total: {total_score:.6f}")
            print(f"      • Thoughts: {thought_score:.6f} (has trace: {has_thought})")
            print(f"      • Facts: {fact_score:.6f} ({num_facts} facts)")
            print(f"      • Sentences: {num_sentences}, Max: {max_score:.6f}")

def find_interesting_patterns(examples):
    """Find interesting attention patterns across examples"""
    print(f"\n{'='*60}")
    print(f"🔍 INTERESTING PATTERNS")
    print(f"{'='*60}")
    
    # Collect method attention patterns
    method_patterns = {
        'total_scores': {},
        'thought_scores': {},
        'fact_scores': {}
    }
    
    for example in examples:
        for method_analysis in example.get('method_call_analysis', []):
            method_name = method_analysis.get('method_name', 'unknown')
            attention_summary = method_analysis.get('attention_summary', {})
            
            # Initialize if not exists
            for pattern_type in method_patterns:
                if method_name not in method_patterns[pattern_type]:
                    method_patterns[pattern_type][method_name] = []
            
            # Collect scores
            method_patterns['total_scores'][method_name].append(
                attention_summary.get('total_attention_score', 0)
            )
            method_patterns['thought_scores'][method_name].append(
                attention_summary.get('thought_attention_score', 0)
            )
            method_patterns['fact_scores'][method_name].append(
                attention_summary.get('fact_attention_score', 0)
            )
    
    # Show patterns
    for method_name in method_patterns['total_scores']:
        total_scores = method_patterns['total_scores'][method_name]
        thought_scores = method_patterns['thought_scores'][method_name]
        fact_scores = method_patterns['fact_scores'][method_name]
        
        if len(total_scores) > 1:
            print(f"🎯 {method_name}:")
            print(f"   Total attention: avg={sum(total_scores)/len(total_scores):.6f}, "
                  f"max={max(total_scores):.6f}, appearances={len(total_scores)}")
            print(f"   Thought attention: avg={sum(thought_scores)/len(thought_scores):.6f}")
            print(f"   Fact attention: avg={sum(fact_scores)/len(fact_scores):.6f}")
            
            # Show which is more important for this method
            avg_thought = sum(thought_scores) / len(thought_scores)
            avg_fact = sum(fact_scores) / len(fact_scores)
            
            if avg_thought > avg_fact:
                print(f"   → More thought-focused (ratio: {avg_thought/avg_fact:.2f})")
            else:
                print(f"   → More fact-focused (ratio: {avg_fact/avg_thought:.2f})")
            print()

def main():
    parser = argparse.ArgumentParser(description="View method call attention patterns")
    parser.add_argument("attention_dir", help="Directory containing attention analysis results")
    parser.add_argument("--example", type=int, default=None, help="Show specific example only")
    parser.add_argument("--detailed", action="store_true", help="Show detailed facts and thoughts")
    parser.add_argument("--patterns", action="store_true", help="Show interesting patterns")
    
    args = parser.parse_args()
    
    # Load attention data
    print("📂 Loading attention data...")
    examples = load_attention_data(args.attention_dir, args.example)
    
    if not examples:
        print("❌ No attention data found")
        return
    
    print(f"✅ Loaded {len(examples)} example(s)")
    
    # Display method attention
    display_method_attention(examples, args.detailed)
    
    # Show patterns if requested and multiple examples
    if args.patterns and len(examples) > 1:
        find_interesting_patterns(examples)

if __name__ == "__main__":
    main() 