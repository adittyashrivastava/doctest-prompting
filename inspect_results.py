#!/usr/bin/env python3
"""
Script to inspect results from 100-example evaluation run.
Usage: python inspect_results.py
"""

import json
import os
from datetime import datetime
from collections import defaultdict, Counter

def load_summary_log(log_dir):
    """Load overall accuracy summary"""
    summary_file = os.path.join(log_dir, "summary.jsonl")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            if lines:
                return json.loads(lines[-1].strip())  # Latest entry
    return None

def load_detailed_results(log_dir):
    """Load detailed per-example results"""
    results_file = os.path.join(log_dir, "medcalc_rules_traces_1.5b.jsonl")
    if os.path.exists(results_file):
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                results.append(json.loads(line.strip()))
        return results
    return []

def analyze_attention_patterns(attention_dir):
    """Analyze attention patterns across all examples"""
    if not os.path.exists(attention_dir):
        return None
    
    example_dirs = [d for d in os.listdir(attention_dir) if d.startswith('example_')]
    analysis = {
        'total_examples': len(example_dirs),
        'examples_with_attention': 0,
        'examples_with_no_attention': 0,
        'avg_facts_per_example': 0,
        'avg_thoughts_per_example': 0,
        'top_attended_methods': Counter(),
        'attention_score_distribution': [],
        'examples_breakdown': []
    }
    
    total_facts = 0
    total_thoughts = 0
    
    for example_dir in sorted(example_dirs):
        example_path = os.path.join(attention_dir, example_dir)
        example_num = int(example_dir.split('_')[1])
        
        # Load attention results
        attention_file = os.path.join(example_path, "attention_results.json")
        method_attention_file = os.path.join(example_path, "method_attention.json")
        
        example_info = {
            'example_num': example_num,
            'has_attention': False,
            'num_facts': 0,
            'num_thoughts': 0,
            'top_attention_score': 0,
            'methods_analyzed': []
        }
        
        if os.path.exists(attention_file):
            with open(attention_file, 'r') as f:
                attention_data = json.load(f)
                
                facts = attention_data.get('retrieved_facts', [])
                thoughts = attention_data.get('retrieved_thoughts', [])
                
                if facts or thoughts:
                    example_info['has_attention'] = True
                    analysis['examples_with_attention'] += 1
                    
                    example_info['num_facts'] = len(facts)
                    example_info['num_thoughts'] = len(thoughts)
                    total_facts += len(facts)
                    total_thoughts += len(thoughts)
                    
                    # Find top attention score
                    all_scores = []
                    for fact in facts:
                        score = fact.get('attention_score', fact.get('score', 0))
                        all_scores.append(score)
                    for thought in thoughts:
                        score = thought.get('attention_score', thought.get('score', 0))
                        all_scores.append(score)
                    
                    if all_scores:
                        example_info['top_attention_score'] = max(all_scores)
                        analysis['attention_score_distribution'].extend(all_scores)
                else:
                    analysis['examples_with_no_attention'] += 1
        else:
            analysis['examples_with_no_attention'] += 1
            
        # Load method attention
        if os.path.exists(method_attention_file):
            with open(method_attention_file, 'r') as f:
                method_data = json.load(f)
                for method_analysis in method_data.get('method_call_analysis', []):
                    method_name = method_analysis.get('method_name', 'unknown')
                    analysis['top_attended_methods'][method_name] += 1
                    example_info['methods_analyzed'].append(method_name)
        
        analysis['examples_breakdown'].append(example_info)
    
    if analysis['examples_with_attention'] > 0:
        analysis['avg_facts_per_example'] = total_facts / analysis['examples_with_attention']
        analysis['avg_thoughts_per_example'] = total_thoughts / analysis['examples_with_attention']
    
    return analysis

def display_performance_analysis(results):
    """Display per-example performance breakdown"""
    print("\n📊 PERFORMANCE BREAKDOWN")
    print("=" * 60)
    
    correct = 0
    total = len(results)
    
    # Group by correctness
    correct_examples = []
    incorrect_examples = []
    
    for i, result in enumerate(results):
        is_correct = result.get('is_correct', False)
        
        if is_correct:
            correct += 1
            correct_examples.append(i)
        else:
            incorrect_examples.append(i)
    
    print(f"✅ Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"❌ Incorrect: {total-correct}/{total} ({(total-correct)/total*100:.1f}%)")
    
    if len(incorrect_examples) <= 10:
        print(f"\n❌ Incorrect examples: {incorrect_examples}")
    else:
        print(f"\n❌ First 10 incorrect examples: {incorrect_examples[:10]}")
        print(f"   ... and {len(incorrect_examples)-10} more")
    
    return correct_examples, incorrect_examples

def display_attention_summary(attention_analysis):
    """Display attention analysis summary"""
    if not attention_analysis:
        print("\n🧠 No attention analysis data found")
        return
    
    print("\n🧠 ATTENTION ANALYSIS SUMMARY")
    print("=" * 60)
    
    total = attention_analysis['total_examples']
    with_attention = attention_analysis['examples_with_attention']
    without_attention = attention_analysis['examples_with_no_attention']
    
    print(f"📊 Attention Coverage:")
    print(f"  • Examples with attention: {with_attention}/{total} ({with_attention/total*100:.1f}%)")
    print(f"  • Examples with no attention: {without_attention}/{total} ({without_attention/total*100:.1f}%)")
    
    if with_attention > 0:
        print(f"\n📈 Attention Statistics:")
        print(f"  • Avg facts per example: {attention_analysis['avg_facts_per_example']:.1f}")
        print(f"  • Avg thoughts per example: {attention_analysis['avg_thoughts_per_example']:.1f}")
        
        scores = attention_analysis['attention_score_distribution']
        if scores:
            print(f"  • Attention score range: {min(scores):.6f} - {max(scores):.6f}")
            print(f"  • Mean attention score: {sum(scores)/len(scores):.6f}")
    
    print(f"\n🎯 Most Analyzed Methods:")
    for method, count in attention_analysis['top_attended_methods'].most_common(5):
        print(f"  • {method}: {count} examples")

def inspect_specific_example(log_dir, attention_dir, example_num):
    """Inspect a specific example in detail"""
    print(f"\n🔍 DETAILED INSPECTION: Example {example_num}")
    print("=" * 60)
    
    # Load detailed result
    results = load_detailed_results(log_dir)
    if example_num < len(results):
        result = results[example_num]
        
        print(f"📋 Example {example_num}:")
        print(f"  • Input: {result.get('input', 'N/A')[:100]}...")
        print(f"  • Target: {result.get('target', 'N/A')}")
        print(f"  • Predicted: {result.get('predicted', 'N/A')}")
        print(f"  • Correct: {result.get('is_correct', False)}")
        
        # Show method calls if available
        method_calls = result.get('method_calls', [])
        if method_calls:
            print(f"  • Method calls: {len(method_calls)}")
            for i, call in enumerate(method_calls[:3]):
                print(f"    {i+1}. {call.get('call_text', 'N/A')[:60]}...")
    
    # Load attention data
    example_dir = os.path.join(attention_dir, f"example_{example_num:04d}")
    if os.path.exists(example_dir):
        print(f"\n🧠 Attention Analysis:")
        
        attention_file = os.path.join(example_dir, "attention_results.json")
        if os.path.exists(attention_file):
            with open(attention_file, 'r') as f:
                attention_data = json.load(f)
                
                facts = attention_data.get('retrieved_facts', [])
                thoughts = attention_data.get('retrieved_thoughts', [])
                
                print(f"  • Facts: {len(facts)}")
                for i, fact in enumerate(facts[:3]):
                    score = fact.get('attention_score', fact.get('score', 0))
                    text = fact.get('text', 'N/A')[:60]
                    print(f"    {i+1}. [{score:.6f}] {text}...")
                
                print(f"  • Thoughts: {len(thoughts)}")
                for i, thought in enumerate(thoughts[:3]):
                    score = thought.get('attention_score', thought.get('score', 0))
                    text = thought.get('text', 'N/A')[:60]
                    print(f"    {i+1}. [{score:.6f}] {text}...")
    else:
        print(f"  • No attention data found")

def main():
    print("🔍 COMPREHENSIVE RESULTS INSPECTION")
    print("=" * 60)
    
    # Find the most recent log directory
    log_base = "logs/medcalc_rules"
    if os.path.exists(log_base):
        subdirs = [d for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d))]
        if subdirs:
            latest_subdir = subdirs[0]  # Should be something like "local"
            model_dirs = os.listdir(os.path.join(log_base, latest_subdir))
            if model_dirs:
                log_dir = os.path.join(log_base, latest_subdir, model_dirs[0])
                attention_dir = os.path.join(log_dir, "attention_analysis")
                
                print(f"📁 Log directory: {log_dir}")
                print(f"🧠 Attention directory: {attention_dir}")
                
                # Load and display overall summary
                summary = load_summary_log(log_dir)
                if summary:
                    print(f"\n📊 OVERALL SUMMARY")
                    print(f"  • Model: {summary.get('model_name', 'N/A')}")
                    print(f"  • Accuracy: {summary.get('accuracy', 0):.3f}")
                    print(f"  • Total examples: {summary.get('total_examples', 0)}")
                    print(f"  • Timestamp: {summary.get('timestamp', 'N/A')}")
                
                # Load and analyze detailed results
                results = load_detailed_results(log_dir)
                if results:
                    correct_examples, incorrect_examples = display_performance_analysis(results)
                    
                    # Analyze attention patterns
                    attention_analysis = analyze_attention_patterns(attention_dir)
                    display_attention_summary(attention_analysis)
                    
                    # Show specific examples
                    print(f"\n🔍 EXAMPLE INSPECTIONS")
                    print("=" * 60)
                    
                    # Show a correct example
                    if correct_examples:
                        inspect_specific_example(log_dir, attention_dir, correct_examples[0])
                    
                    # Show an incorrect example
                    if incorrect_examples:
                        inspect_specific_example(log_dir, attention_dir, incorrect_examples[0])
                    
                    print(f"\n💡 USAGE COMMANDS:")
                    print(f"  • View method attention: python view_method_attention.py {attention_dir} --example 0")
                    print(f"  • View all attention: python view_attention_results.py {attention_dir}")
                    print(f"  • Debug specific example: python debug_attention.py --example 5")
                    
                else:
                    print("❌ No detailed results found")
            else:
                print("❌ No model directories found")
        else:
            print("❌ No log subdirectories found")
    else:
        print("❌ Log directory not found")

if __name__ == "__main__":
    main() 