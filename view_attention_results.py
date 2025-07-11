#!/usr/bin/env python3
"""
Simple script to view attention analysis results from log files.
Usage: python view_attention_results.py [path_to_attention_analysis_dir]
"""

import sys
import os
import json
from glob import glob

def load_attention_results(analysis_dir):
    """Load attention analysis results from a directory"""
    results = []
    
    # Find all example directories
    example_dirs = sorted(glob(os.path.join(analysis_dir, "example_*")))
    
    for example_dir in example_dirs:
        results_file = os.path.join(example_dir, "top_facts.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                results.append(data)
            except Exception as e:
                print(f"❌ Error loading {results_file}: {e}")
    
    return results

def display_attention_summary(results):
    """Display a summary of attention analysis results"""
    print("🧠 ATTENTION ANALYSIS SUMMARY")
    print("=" * 60)
    
    total_examples = len(results)
    total_facts = sum(len(r.get('retrieved_facts', [])) for r in results)
    
    print(f"📊 Overall Statistics:")
    print(f"  • Total examples analyzed: {total_examples}")
    print(f"  • Total facts extracted: {total_facts}")
    print(f"  • Average facts per example: {total_facts/total_examples:.1f}")
    print("")
    
    for i, result in enumerate(results):
        example_idx = result.get('example_idx', i)
        facts = result.get('retrieved_facts', [])
        target = result.get('target', 'Unknown')
        
        print(f"📋 Example {example_idx} (Target: {target})")
        print(f"   Input: {result.get('input_preview', 'N/A')[:80]}...")
        print(f"   Facts extracted: {len(facts)}")
        
        if facts:
            print(f"   🔍 Top attention scores:")
            for j, fact in enumerate(facts[:3], 1):
                score = fact.get('attention_score', 0)
                frequency = fact.get('frequency', 0)
                text = fact.get('text', 'No text')[:60]
                print(f"      {j}. Score: {score:.8f} (freq: {frequency:.4f})")
                print(f"         Text: \"{text}...\"")
        else:
            print(f"   ⚠️ No facts extracted")
        
        print("")

def display_detailed_facts(results, example_idx=None):
    """Display detailed facts for a specific example or all examples"""
    print("🔍 DETAILED ATTENTION FACTS")
    print("=" * 60)
    
    for i, result in enumerate(results):
        current_example = result.get('example_idx', i)
        
        # Skip if specific example requested and this isn't it
        if example_idx is not None and current_example != example_idx:
            continue
            
        facts = result.get('retrieved_facts', [])
        target = result.get('target', 'Unknown')
        
        print(f"📋 Example {current_example} (Target: {target})")
        print(f"   Input: {result.get('input_preview', 'N/A')[:100]}...")
        print(f"   Response: {result.get('response_preview', 'N/A')[:100]}...")
        print("")
        
        if facts:
            for j, fact in enumerate(facts, 1):
                score = fact.get('attention_score', 0)
                frequency = fact.get('frequency', 0)
                text = fact.get('text', 'No text')
                token_range = f"{fact.get('token_start', 'N/A')}-{fact.get('token_end', 'N/A')}"
                
                print(f"   🎯 Fact {j}:")
                print(f"      • Attention Score: {score:.8f}")
                print(f"      • Frequency: {frequency:.6f}")
                print(f"      • Token Range: {token_range}")
                print(f"      • Text: \"{text}\"")
                print("")
        else:
            print(f"   ⚠️ No facts extracted for this example")
        
        print("-" * 60)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        analysis_dir = sys.argv[1]
    else:
        # Auto-detect latest attention analysis directory
        log_dirs = glob("logs/*/medcalc_rules/attention_analysis")
        if not log_dirs:
            print("❌ No attention analysis directories found in logs/")
            print("Usage: python view_attention_results.py [path_to_attention_analysis_dir]")
            return
        
        # Use the most recent one
        analysis_dir = sorted(log_dirs)[-1]
        print(f"📁 Auto-detected: {analysis_dir}")
    
    if not os.path.exists(analysis_dir):
        print(f"❌ Directory not found: {analysis_dir}")
        return
    
    # Load results
    print(f"🔄 Loading attention analysis results from: {analysis_dir}")
    results = load_attention_results(analysis_dir)
    
    if not results:
        print("❌ No attention analysis results found!")
        return
    
    # Display summary
    display_attention_summary(results)
    
    # Ask user what they want to see
    while True:
        print("\n" + "=" * 60)
        print("OPTIONS:")
        print("  1. Show summary (default)")
        print("  2. Show detailed facts for all examples")
        print("  3. Show detailed facts for specific example")
        print("  4. Exit")
        
        choice = input("\nEnter choice (1-4) or press Enter for summary: ").strip()
        
        if choice == "" or choice == "1":
            display_attention_summary(results)
        elif choice == "2":
            display_detailed_facts(results)
        elif choice == "3":
            try:
                example_num = int(input("Enter example number: "))
                display_detailed_facts(results, example_num)
            except ValueError:
                print("❌ Invalid example number")
        elif choice == "4":
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 