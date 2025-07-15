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
            
            # Show related facts if detailed view
            if show_detailed:
                related_facts = method_analysis.get('related_facts', [])
                if related_facts:
                    print(f"   💡 Related facts:")
                    for j, fact in enumerate(related_facts[:2], 1):
                        score = fact.get('score', 0)
                        text = fact.get('text', 'No text')
                        if len(text) > 60:
                            text = text[:57] + "..."
                        print(f"      {j}. [{score:.6f}] \"{text}\"")
                
                related_thoughts = method_analysis.get('related_thoughts', [])
                if related_thoughts:
                    print(f"   🧠 Related thoughts:")
                    for j, thought in enumerate(related_thoughts[:2], 1):
                        score = thought.get('score', 0)
                        text = thought.get('text', 'No text')
                        if len(text) > 60:
                            text = text[:57] + "..."
                        print(f"      {j}. [{score:.6f}] \"{text}\"")
            
            # Show attention summary
            attention_summary = method_analysis.get('attention_summary', {})
            total_score = attention_summary.get('total_attention_score', 0)
            num_sentences = attention_summary.get('num_attended_sentences', 0)
            max_score = attention_summary.get('max_attention_score', 0)
            
            print(f"   📊 Attention: total={total_score:.6f}, sentences={num_sentences}, max={max_score:.6f}")

def find_interesting_patterns(examples):
    """Find interesting attention patterns across examples"""
    print(f"\n{'='*60}")
    print(f"🔍 INTERESTING PATTERNS")
    print(f"{'='*60}")
    
    # Collect method attention patterns
    method_patterns = {}
    
    for example in examples:
        for method_analysis in example.get('method_call_analysis', []):
            method_name = method_analysis.get('method_name', 'unknown')
            attention_summary = method_analysis.get('attention_summary', {})
            
            if method_name not in method_patterns:
                method_patterns[method_name] = []
            
            method_patterns[method_name].append(attention_summary.get('total_attention_score', 0))
    
    # Show patterns
    for method_name, scores in method_patterns.items():
        if len(scores) > 1:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            print(f"🎯 {method_name}:")
            print(f"   Average attention: {avg_score:.6f}")
            print(f"   Range: {min_score:.6f} - {max_score:.6f}")
            print(f"   Appearances: {len(scores)}")

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