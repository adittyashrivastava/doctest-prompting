#!/usr/bin/env python3
"""
Quick Test Script for vLLM Integration

This script quickly tests the vLLM integration before running the full SLURM job.
"""

import sys
import os
import json
import time

def test_vllm_installation():
    """Test if vLLM is properly installed and working."""
    print("🔍 Testing vLLM Installation...")

    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")

        import torch
        print(f"✅ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("❌ CUDA not available")
            return False

    except ImportError as e:
        print(f"❌ vLLM not installed: {e}")
        return False

def test_job_util_vllm_import():
    """Test if job_util_vllm.py can be imported."""
    print("\n🔍 Testing job_util_vllm.py import...")

    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        # Test importing the module
        import job_util_vllm
        print("✅ job_util_vllm.py imported successfully")

        # Test if main functions exist
        if hasattr(job_util_vllm, 'setup_vllm_model'):
            print("✅ setup_vllm_model function exists")
        else:
            print("❌ setup_vllm_model function missing")
            return False

        if hasattr(job_util_vllm, 'generate_with_vllm'):
            print("✅ generate_with_vllm function exists")
        else:
            print("❌ generate_with_vllm function missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Failed to import job_util_vllm: {e}")
        return False

def test_attention_viz_integration():
    """Test if attention_viz integration works."""
    print("\n🔍 Testing attention_viz integration...")

    try:
        from attention_viz import AttentionExtractor, AttrievelRetriever, AttrievelConfig
        print("✅ attention_viz modules imported successfully")
        return True
    except ImportError as e:
        print(f"⚠️  attention_viz not available: {e}")
        print("This is optional but recommended for attention analysis")
        return True  # Don't fail the test for this

def test_model_loading():
    """Test if a small model can be loaded with vLLM."""
    print("\n🔍 Testing vLLM model loading...")

    try:
        from vllm import LLM, SamplingParams

        # Try loading a small model for testing
        print("Loading small test model...")
        llm = LLM(
            model="microsoft/DialoGPT-small",  # Small model for testing
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512
        )

        # Test generation
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=10
        )

        outputs = llm.generate(["Hello, how are you?"], sampling_params)
        result = outputs[0].outputs[0].text

        print(f"✅ vLLM model loading and generation successful")
        print(f"   Test output: {result}")
        return True

    except Exception as e:
        print(f"❌ vLLM model loading failed: {e}")
        return False

def test_config_files():
    """Test if required config files exist."""
    print("\n🔍 Testing config files...")

    config_files = [
        "conf.d/medcalc.conf",
        "conf.d/qwen2.5-7b.conf"
    ]

    missing_files = []
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} exists")
        else:
            print(f"❌ {config_file} missing")
            missing_files.append(config_file)

    if missing_files:
        print(f"⚠️  Missing config files: {missing_files}")
        print("You may need to create these config files or adjust the paths")
        return False

    return True

def main():
    """Run all tests."""
    print("🚀 Quick vLLM Integration Test")
    print("="*50)

    tests = [
        ("vLLM Installation", test_vllm_installation),
        ("job_util_vllm Import", test_job_util_vllm_import),
        ("Attention Viz Integration", test_attention_viz_integration),
        ("vLLM Model Loading", test_model_loading),
        ("Config Files", test_config_files)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name}: {status}")

    print(f"\nPassed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 All tests passed! Ready to run SLURM job.")
        print("You can now submit: sbatch medcalc_attention_analysis_vllm.slurm")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please fix issues before running SLURM job.")

    print("="*50)

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)