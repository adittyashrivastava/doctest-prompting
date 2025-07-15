#!/usr/bin/env python3
"""
Visualize Method Call Attention Analysis

This script provides easy-to-understand visualization of which input sentences
or tokens each method call in the program trace attends to.
"""

import json
import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_method_attention_data(attention_dir):
    """Load method attention data from analysis results"""
    method_data = []
    
    # Find all example directories
    example_dirs = [d for d in os.listdir(attention_dir) if d.startswith('example_')]
    example_dirs.sort()
    
    for example_dir in example_dirs:
        viz_file = os.path.join(attention_dir, example_dir, "method_attention_visualization.json")
        if os.path.exists(viz_file):
            with open(viz_file, 'r') as f:
                data = json.load(f)
                method_data.append(data)
    
    return method_data

def print_method_attention_summary(method_data):
    """Print a summary of method attention patterns"""
    print("🔍 METHOD CALL ATTENTION ANALYSIS")
    print("=" * 80)
    
    for i, example_data in enumerate(method_data):
        print(f"\n📋 Example {i}")
        print(f"   Total method calls: {example_data['summary']['total_method_calls']}")
        print(f"   Input context length: {example_data['summary']['input_context_length']}")
        print(f"   Total facts: {example_data['summary']['total_facts']}")
        print(f"   Total thoughts: {example_data['summary']['total_thoughts']}")
        
        for method_analysis in example_data['method_call_analysis']:
            method_name = method_analysis['method_name']
            method_call = method_analysis['method_call_text']
            
            print(f"\n   🎯 Method: {method_name}")
            print(f"      Call: {method_call}")
            
            # Show top attended sentences
            top_sentences = method_analysis['top_attended_sentences']
            if top_sentences:
                print(f"      📝 Top Attended Sentences:")
                for j, sentence_info in enumerate(top_sentences[:3], 1):
                    score = sentence_info['attention_score']
                    sentence = sentence_info['sentence'][:60] + "..." if len(sentence_info['sentence']) > 60 else sentence_info['sentence']
                    print(f"         {j}. Score: {score:.6f} - \"{sentence}\"")
            else:
                print(f"      📝 No significant attention found")
            
            # Show related facts
            related_facts = method_analysis['related_facts']
            if related_facts:
                print(f"      💡 Related Facts:")
                for j, fact in enumerate(related_facts[:2], 1):
                    fact_text = fact.get('text', 'No text')[:50] + "..." if len(fact.get('text', '')) > 50 else fact.get('text', 'No text')
                    score = fact.get('score', 0)
                    print(f"         {j}. Score: {score:.6f} - \"{fact_text}\"")
            
            # Show matching thought (program trace)
            matching_thought = method_analysis.get('matching_thought')
            if matching_thought:
                print(f"      🧠 Program Trace:")
                thought_text = matching_thought.get('text', 'No text')[:50] + "..." if len(matching_thought.get('text', '')) > 50 else matching_thought.get('text', 'No text')
                score = matching_thought.get('attention_score', 0)
                token_start = matching_thought.get('token_start', 0)
                token_end = matching_thought.get('token_end', 0)
                print(f"         Score: {score:.6f} - \"{thought_text}\" (tokens: {token_start}-{token_end})")
            
            # Show attention summary
            attention_summary = method_analysis['attention_summary']
            print(f"      📊 Attention Summary:")
            print(f"         Total score: {attention_summary['total_attention_score']:.6f}")
            print(f"         Sentences attended: {attention_summary['num_attended_sentences']}")
            print(f"         Max score: {attention_summary['max_attention_score']:.6f}")

def create_attention_heatmap(method_data, output_dir):
    """Create heatmap visualizations of method attention patterns"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, example_data in enumerate(method_data):
        fig, axes = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        method_names = []
        sentence_scores = []
        
        for method_analysis in example_data['method_call_analysis']:
            method_name = method_analysis['method_name']
            method_names.append(method_name)
            
            # Get attention scores for sentences
            top_sentences = method_analysis['top_attended_sentences']
            scores = [s['attention_score'] for s in top_sentences[:10]]  # Top 10 sentences
            
            # Pad with zeros if needed
            while len(scores) < 10:
                scores.append(0.0)
            
            sentence_scores.append(scores)
        
        if sentence_scores:
            # Create heatmap
            heatmap_data = np.array(sentence_scores)
            
            im = axes.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            
            # Set labels
            axes.set_xticks(range(10))
            axes.set_xticklabels([f'Sentence {i+1}' for i in range(10)], rotation=45)
            axes.set_yticks(range(len(method_names)))
            axes.set_yticklabels(method_names)
            
            # Add colorbar
            plt.colorbar(im, ax=axes, label='Attention Score')
            
            # Add title
            axes.set_title(f'Method Call Attention Heatmap - Example {i}')
            
            # Add text annotations
            for j in range(len(method_names)):
                for k in range(10):
                    if k < len(sentence_scores[j]):
                        text = f'{sentence_scores[j][k]:.4f}'
                        axes.text(k, j, text, ha="center", va="center", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'attention_heatmap_example_{i}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Heatmap visualizations saved to: {output_dir}")

def analyze_attention_patterns(method_data):
    """Analyze overall attention patterns across all examples"""
    print("\n🔍 OVERALL ATTENTION PATTERN ANALYSIS")
    print("=" * 80)
    
    # Collect statistics
    method_attention_stats = defaultdict(list)
    sentence_position_stats = defaultdict(list)
    
    for example_data in method_data:
        for method_analysis in example_data['method_call_analysis']:
            method_name = method_analysis['method_name']
            attention_summary = method_analysis['attention_summary']
            
            method_attention_stats[method_name].append(attention_summary['total_attention_score'])
            
            # Analyze sentence position preferences
            for sentence_info in method_analysis['top_attended_sentences']:
                sentence_idx = sentence_info['sentence_idx']
                attention_score = sentence_info['attention_score']
                sentence_position_stats[sentence_idx].append(attention_score)
    
    # Print method-wise statistics
    print("\n📈 Method-wise Attention Statistics:")
    for method_name, scores in method_attention_stats.items():
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        print(f"   {method_name}: avg={avg_score:.6f}, max={max_score:.6f}, min={min_score:.6f}, count={len(scores)}")
    
    # Print sentence position preferences
    print("\n📍 Sentence Position Preferences:")
    for position, scores in sorted(sentence_position_stats.items()):
        avg_score = np.mean(scores)
        print(f"   Position {position}: avg_attention={avg_score:.6f}, frequency={len(scores)}")

def generate_method_attention_report(method_data, output_file):
    """Generate a comprehensive report of method attention patterns"""
    with open(output_file, 'w') as f:
        f.write("METHOD CALL ATTENTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for i, example_data in enumerate(method_data):
            f.write(f"EXAMPLE {i}\n")
            f.write("-" * 40 + "\n")
            
            f.write(f"Summary:\n")
            f.write(f"  - Total method calls: {example_data['summary']['total_method_calls']}\n")
            f.write(f"  - Input context length: {example_data['summary']['input_context_length']}\n")
            f.write(f"  - Total facts: {example_data['summary']['total_facts']}\n")
            f.write(f"  - Total thoughts: {example_data['summary']['total_thoughts']}\n\n")
            
            for method_analysis in example_data['method_call_analysis']:
                method_name = method_analysis['method_name']
                f.write(f"Method: {method_name}\n")
                f.write(f"Call: {method_analysis['method_call_text']}\n")
                
                # Top attended sentences
                f.write(f"Top Attended Sentences:\n")
                for j, sentence_info in enumerate(method_analysis['top_attended_sentences'][:5], 1):
                    f.write(f"  {j}. Score: {sentence_info['attention_score']:.6f}\n")
                    f.write(f"     Sentence: {sentence_info['sentence']}\n")
                
                # Related facts
                f.write(f"Related Facts:\n")
                for j, fact in enumerate(method_analysis['related_facts'][:3], 1):
                    f.write(f"  {j}. Score: {fact.get('score', 0):.6f}\n")
                    f.write(f"     Text: {fact.get('text', 'No text')}\n")
                
                # Matching thought (program trace)
                matching_thought = method_analysis.get('matching_thought')
                if matching_thought:
                    f.write(f"Program Trace:\n")
                    f.write(f"  Score: {matching_thought.get('attention_score', 0):.6f}\n")
                    f.write(f"  Text: {matching_thought.get('text', 'No text')}\n")
                    f.write(f"  Token positions: {matching_thought.get('token_start', 0)}-{matching_thought.get('token_end', 0)}\n")
                else:
                    f.write(f"Program Trace: None\n")
                
                f.write(f"Attention Summary:\n")
                attention_summary = method_analysis['attention_summary']
                f.write(f"  - Total attention score: {attention_summary['total_attention_score']:.6f}\n")
                f.write(f"  - Sentences attended: {attention_summary['num_attended_sentences']}\n")
                f.write(f"  - Max attention score: {attention_summary['max_attention_score']:.6f}\n\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"📄 Comprehensive report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize method call attention patterns")
    parser.add_argument("attention_dir", help="Directory containing attention analysis results")
    parser.add_argument("--output-dir", default="attention_visualizations", help="Output directory for visualizations")
    parser.add_argument("--create-heatmap", action="store_true", help="Create heatmap visualizations")
    parser.add_argument("--generate-report", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--example-idx", type=int, default=None, help="Analyze specific example only")
    
    args = parser.parse_args()
    
    # Load method attention data
    print("📂 Loading method attention data...")
    method_data = load_method_attention_data(args.attention_dir)
    
    if not method_data:
        print("❌ No method attention data found in the specified directory")
        return
    
    print(f"✅ Loaded attention data for {len(method_data)} examples")
    
    # Filter to specific example if requested
    if args.example_idx is not None:
        if args.example_idx < len(method_data):
            method_data = [method_data[args.example_idx]]
            print(f"🎯 Analyzing example {args.example_idx} only")
        else:
            print(f"❌ Example {args.example_idx} not found (only {len(method_data)} examples available)")
            return
    
    # Print summary
    print_method_attention_summary(method_data)
    
    # Create heatmap visualization
    if args.create_heatmap:
        print("\n📊 Creating heatmap visualizations...")
        create_attention_heatmap(method_data, args.output_dir)
    
    # Analyze overall patterns
    if len(method_data) > 1:
        analyze_attention_patterns(method_data)
    
    # Generate comprehensive report
    if args.generate_report:
        report_file = os.path.join(args.output_dir, "method_attention_report.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        generate_method_attention_report(method_data, report_file)

if __name__ == "__main__":
    main() 