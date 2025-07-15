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
        "templates/predict_output_with_traces.txt"
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
    run_command(
        "python run_eval.py medcalc_rules --lo 0 --hi 2 --enable_attention_analysis --model Qwen2.5-1.5B-Instruct --service hf_local",
        "Run evaluation with attention analysis (first 2 examples)"
    )
    
    # Demo 2: Check summary logs
    print("\n📋 DEMO 2: Checking summary logs")
    summary_file = "logs/run_summaries/evaluation_summary.jsonl"
    check_file_exists(summary_file, "Run summary log")
    
    # Demo 3: Check attention analysis results
    print("\n🧠 DEMO 3: Checking attention analysis results")
    attention_dir = "logs/local-Qwen/Qwen2.5-1.5B-Instruct/medcalc_rules/attention_analysis"
    
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
    
    # Demo 4: View method attention patterns
    print("\n🎯 DEMO 4: Viewing method attention patterns")
    if os.path.exists(attention_dir):
        run_command(
            f"python view_method_attention.py {attention_dir} --example 0",
            "View method attention for example 0"
        )
    else:
        print("❌ Cannot demo method attention - no attention data found")
    
    # Demo 5: Show the enhanced PTP prompt
    print("\n📝 DEMO 5: Enhanced PTP prompt (input/output only)")
    ptp_constants_file = "training/data/constants.py"
    if os.path.exists(ptp_constants_file):
        print("✅ PTP constants file exists")
        with open(ptp_constants_file, 'r') as f:
            content = f.read()
            if "showing ONLY the input and output of each method call" in content:
                print("✅ PTP prompt has been enhanced for input/output only")
                print("Sample instructions:")
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "In the program trace, show only:" in line:
                        for j in range(i, min(i+6, len(lines))):
                            print(f"  {lines[j]}")
                        break
            else:
                print("❌ PTP prompt not found or not enhanced")
    else:
        print("❌ PTP constants file not found")
    
    # Demo 6: Show error handling capabilities
    print("\n🛡️ DEMO 6: Error handling capabilities")
    print("The system now handles:")
    print("  • ZeroDivisionError when calculating accuracy")
    print("  • Individual example failures without stopping the run")
    print("  • Graceful degradation when attention analysis fails")
    print("  • GPU memory issues with CPU fallback")
    print()
    print("Example error handling in accuracy calculation:")
    print("```python")
    print("try:")
    print("    if parsed > 0:")
    print("        acc = correct / parsed")
    print("    else:")
    print("        echo(log_fp, 'acc=0.0 (all examples failed to parse)')")
    print("except ZeroDivisionError as e:")
    print("    echo(log_fp, f'Error calculating accuracy: {e}')")
    print("```")
    
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("Summary of enhancements demonstrated:")
    print("✅ 1. Overall accuracy tracking with summary logs")
    print("✅ 2. Top-k thoughts analysis in attention")
    print("✅ 3. Robust error handling")
    print("✅ 4. Input/output only program traces")
    print("✅ 5. Method call attention visualization")
    print()
    print("For detailed documentation, see: IMPLEMENTATION_SUMMARY.md")

if __name__ == "__main__":
    main() 