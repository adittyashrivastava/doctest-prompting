#!/usr/bin/env python3
"""
Monitor progress of the 100-example evaluation run.
Usage: python monitor_progress.py
"""

import json
import os
import time
from datetime import datetime

def check_progress():
    """Check current progress of the evaluation"""
    log_base = "logs/medcalc_rules"
    
    # Find the most recent log directory
    if os.path.exists(log_base):
        subdirs = [d for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d))]
        if subdirs:
            latest_subdir = subdirs[0]  # Should be something like "local"
            model_dirs = os.listdir(os.path.join(log_base, latest_subdir))
            if model_dirs:
                log_dir = os.path.join(log_base, latest_subdir, model_dirs[0])
                
                # Check main results file
                results_file = os.path.join(log_dir, "medcalc_rules_traces_1.5b.jsonl")
                attention_dir = os.path.join(log_dir, "attention_analysis")
                
                completed_examples = 0
                completed_attention = 0
                
                # Count completed examples
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        completed_examples = sum(1 for line in f if line.strip())
                
                # Count attention analysis
                if os.path.exists(attention_dir):
                    example_dirs = [d for d in os.listdir(attention_dir) if d.startswith('example_')]
                    completed_attention = len(example_dirs)
                
                print(f"📊 Progress Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*50}")
                print(f"📁 Log directory: {log_dir}")
                print(f"✅ Completed examples: {completed_examples}/100")
                print(f"🧠 Attention analysis: {completed_attention}/100")
                print(f"📈 Overall progress: {max(completed_examples, completed_attention)}/100 ({max(completed_examples, completed_attention):.0f}%)")
                
                # Show recent activity
                if completed_examples > 0:
                    print(f"\n📋 Recent examples:")
                    with open(results_file, 'r') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines[-3:], len(lines)-2):
                            try:
                                result = json.loads(line.strip())
                                is_correct = result.get('is_correct', False)
                                status = "✅" if is_correct else "❌"
                                print(f"   {i}: {status} {result.get('target', 'N/A')}")
                            except:
                                print(f"   {i}: Processing...")
                
                return completed_examples, completed_attention
    
    print("❌ No log directory found - evaluation may not have started yet")
    return 0, 0

def main():
    """Monitor progress continuously"""
    print("🔍 EVALUATION PROGRESS MONITOR")
    print("Press Ctrl+C to stop monitoring")
    print("="*50)
    
    try:
        while True:
            completed, attention = check_progress()
            
            if completed >= 100 and attention >= 100:
                print("\n🎉 EVALUATION COMPLETE!")
                print("Run: python inspect_results.py")
                break
            
            time.sleep(30)  # Check every 30 seconds
            print()  # Add spacing
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    main() 