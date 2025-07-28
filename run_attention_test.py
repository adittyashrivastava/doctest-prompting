#!/usr/bin/env python3
"""
Simple script to run attention module testing
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def main():
    """Main function to run a basic attention test"""
    print("🚀 Starting Attention Module Test")
    print("=" * 50)
    
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
        
        # Run a subset of tests first (first 5 examples)
        print(f"🔍 Running tests on first 5 examples...")
        original_examples = test_suite.test_examples
        test_suite.test_examples = test_suite.test_examples[:5]
        
        # Run the tests
        test_suite.results = []
        for example in test_suite.test_examples:
            print(f"\n📝 Testing: {example.id} - {example.description}")
            result = test_suite.evaluate_example(example)
            test_suite.results.append(result)
            
            # Print immediate results
            print(f"   Precision: {result.precision:.3f}")
            print(f"   Recall: {result.recall:.3f}")
            print(f"   F1 Score: {result.f1_score:.3f}")
            print(f"   Top-{test_suite.k} Accuracy: {result.top_k_accuracy:.3f}")
            
        # Generate summary
        test_suite.generate_summary()
        test_suite.save_results()
        
        print("\n🎉 Test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 