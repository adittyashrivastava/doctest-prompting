#!/usr/bin/env python3
"""
Runner script for General English Attention Module Testing
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def main():
    """Main function to run general English attention test"""
    print("🚀 Starting General English Attention Module Test")
    print("=" * 60)
    
    # Import the test module
    try:
        from test_attention_general_english import GeneralEnglishAttentionTestSuite
        print("✅ General English test module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        return 1
    
    # Create test suite
    test_suite = GeneralEnglishAttentionTestSuite(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        k=3  # Start with top 3 facts for faster testing
    )
    
    try:
        print("🔧 Setting up test suite...")
        test_suite.setup_model()
        test_suite.create_test_dataset()
        
        # Run a subset of tests first (first 8 examples - about half)
        print(f"🔍 Running tests on first 8 examples...")
        original_examples = test_suite.test_examples
        test_suite.test_examples = test_suite.test_examples[:8]
        
        # Run the tests
        test_suite.results = []
        for example in test_suite.test_examples:
            print(f"\n📝 Testing: {example.id} ({example.domain}) - {example.description}")
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
        
        print("\n🎉 General English test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 