#!/usr/bin/env python3
"""
Demo script showing how to use the enhanced doctest-prompting features.

This script demonstrates the 5 major enhancements:
1. Overall accuracy tracking with summary logs
2. Top-k thoughts analysis in attention
3. Robust error handling
4. Input/output only program traces
5. Method call attention visualization
"""

import json
import os
import subprocess
import sys
from datetime import datetime

def run_command(cmd, description):
    """Run a command and show its output"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Command completed successfully")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ Command failed with exit code {result.returncode}")
            if result.stderr:
                print("Error:")
                print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out (5 minutes)")
    except Exception as e:
        print(f"❌ Error running command: {e}")

def check_file_exists(filepath, description):
    """Check if a file exists and show its content"""
    print(f"\n📁 Checking {description}: {filepath}")
    if os.path.exists(filepath):
        print("✅ File exists")
        if filepath.endswith('.jsonl'):
            print("Sample content:")
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[-3:], 1):  # Show last 3 lines
                    try:
                        data = json.loads(line.strip())
                        print(f"  {i}. {data.get('model_name', 'Unknown')} - {data.get('accuracy', 'N/A'):.3f} accuracy")
                    except:
                        print(f"  {i}. {line.strip()[:80]}...")
        elif filepath.endswith('.json'):
            print("File size:", os.path.getsize(filepath), "bytes")
    else:
        print("❌ File does not exist")

def main():
    print("🚀 ENHANCED DOCTEST-PROMPTING SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the 5 major enhancements:")
    print("1. Overall accuracy tracking with summary logs")
    print("2. Top-k thoughts analysis in attention")
    print("3. Robust error handling")
    print("4. Input/output only program traces")
    print("5. Method call attention visualization")
    print()
    
    # Check if we have the required files
    required_files = [
        "run_eval.py",
        "view_method_attention.py",
        "training/data/constants.py",
        "templates/predict_output_with_traces.txt",
        "run_eval_with_traces_and_attention.sh"
    ]
    
    print("🔍 Checking required files...")
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing!")
            
    print()
    
    # Demo 1: Run evaluation with enhanced features
    print("📊 DEMO 1: Running evaluation with enhanced features")
    
    # Check if the shell script exists
    shell_script = "./run_eval_with_traces_and_attention.sh"
    if os.path.exists(shell_script):
        run_command(
            f"{shell_script} medcalc_rules qwen2.5-1.5b 0 2 --enable-attention",
            "Run evaluation with attention analysis using shell script (first 2 examples)"
        )
    else:
        print(f"❌ Shell script not found: {shell_script}")
        print("Trying direct python command as fallback...")
        run_command(
            "python run_eval.py medcalc_rules --lo 0 --hi 2 --enable_attention_analysis --model Qwen2.5-1.5B-Instruct --service hf_local --log_dir logs",
            "Run evaluation with attention analysis (fallback command)"
        )
    
    # Demo 2: Check attention analysis results
    print("\n🧠 DEMO 2: Checking attention analysis results")
    
    # Look for attention analysis directory created by shell script
    possible_attention_dirs = [
        "logs/local-Qwen/Qwen2.5-1.5B-Instruct/medcalc_rules/attention_analysis",
        "logs/hf_local/Qwen2.5-1.5B-Instruct/medcalc_rules/attention_analysis", 
        "logs/local/Qwen2.5-1.5B-Instruct/medcalc_rules/attention_analysis",
        "logs/hf_local/qwen2.5-1.5b/medcalc_rules/attention_analysis",
        "logs/local-Qwen/qwen2.5-1.5b/medcalc_rules/attention_analysis"
    ]
    
    attention_dir = None
    for possible_dir in possible_attention_dirs:
        if os.path.exists(possible_dir):
            attention_dir = possible_dir
            break
    
    if attention_dir is None:
        # Default to first option
        attention_dir = possible_attention_dirs[0]
        print(f"⚠️  No attention analysis directory found. Checked: {possible_attention_dirs}")
    else:
        print(f"✅ Found attention analysis directory: {attention_dir}")
    
    if os.path.exists(attention_dir):
        print(f"✅ Attention analysis directory exists: {attention_dir}")
        example_dirs = [d for d in os.listdir(attention_dir) if d.startswith('example_')]
        print(f"📊 Found {len(example_dirs)} analyzed examples")
        
        for example_dir in example_dirs[:2]:  # Show first 2
            example_path = os.path.join(attention_dir, example_dir)
            files = os.listdir(example_path)
            print(f"  📁 {example_dir}: {files}")
    else:
        print(f"❌ Attention analysis directory not found: {attention_dir}")
    
    # Demo 3: View method attention patterns
    print("\n🎯 DEMO 3: Viewing method attention patterns")
    if os.path.exists(attention_dir):
        run_command(
            f"python view_method_attention.py {attention_dir} --example 0",
            "View method attention for example 0"
        )
    else:
        print("❌ Cannot demo method attention - no attention data found")
    
    
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("Summary of enhancements demonstrated:")
    print("✅ 1. Overall accuracy tracking with summary logs")
    print("✅ 2. Top-k thoughts analysis in attention")
    print("✅ 3. Method call attention visualization")

if __name__ == "__main__":
    main() 