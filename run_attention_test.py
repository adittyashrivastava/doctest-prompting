#!/usr/bin/env python3
"""
Simple script to run attention module testing
"""

import sys
import os

# Set CUDA debugging for better error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable device-side assertions for detailed error info

# Add current directory to path
sys.path.append('.')

import torch  # Add torch import for CUDA operations

# Take command line arguments
import argparse

parser = argparse.ArgumentParser(description='Run attention test')
parser.add_argument('--examples', type=int, default=10, help='Number of examples to run')
args = parser.parse_args()

def main():
    """Main function to run a basic attention test"""
    print("🚀 Starting Attention Module Test")
    print("=" * 50)
    
    # Check available resources
    import torch
    if torch.cuda.is_available():
        print(f"🚀 GPU acceleration available")
        print(f"🖥️  GPU device: {torch.cuda.get_device_name()}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🔧 Using GPU with memory optimizations")
    else:
        print(f"🖥️  Using CPU execution")
        print(f"💾 Available CPU cores: {torch.get_num_threads()}")
    
    # Import the test module
    try:
        from test_attention_top_k_facts import AttentionFactTestSuite
        print("✅ Test module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        return 1
    
    # Create test suite with simplified parameters
    test_suite = AttentionFactTestSuite(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        k=3  # Start with top 3 facts for faster testing
    )
    
    try:
        print("🔧 Setting up test suite...")
        test_suite.setup_model()
        test_suite.create_test_dataset()
        
        # Run tests on examples specified by the user
        original_examples = test_suite.test_examples
        print(f"🔍 Running tests on {args.examples} examples...")
        test_suite.test_examples = test_suite.test_examples[:args.examples]
        
        # Run the tests
        test_suite.results = []
        for i, example in enumerate(test_suite.test_examples):
            try:
                print(f"\n📝 Testing: {example.id} - {example.description}")
                result = test_suite.evaluate_example(example)
                test_suite.results.append(result)
                
                # Print immediate results
                print(f"   Top-{test_suite.k} Containment Score: {result.top_k_containment_score:.3f}")
                print(f"   Retrieved Facts: {result.top_k_facts_text}")
                print(f"   🤖 LLM Response: {result.llm_response}")
                print(f"   ✅ Expected: {result.expected_answer}")
                
            except Exception as example_error:
                print(f"   ❌ Failed to evaluate example {example.id}: {example_error}")
                print("   Continuing with next example...")
                continue
            
        if not test_suite.results:
            print("❌ No examples were successfully evaluated!")
            return 1
            
        # Generate summary
        test_suite.generate_summary()
        test_suite.save_results()
        
        print("\n🎉 Test completed successfully!")
        
        # Cleanup GPU memory
        test_suite.cleanup()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try cleanup even on failure
        try:
            if 'test_suite' in locals():
                test_suite.cleanup()
        except:
            pass
            
        return 1

if __name__ == "__main__":
    sys.exit(main()) 