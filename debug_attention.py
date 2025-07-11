#!/usr/bin/env python3
"""
Debug script for attention analysis to identify NaN score issues.
"""

import sys
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add current directory to path
sys.path.append('.')

# Import attention_viz if available
try:
    from attention_viz import AttentionExtractor, AttrievelRetriever, AttrievelConfig
    ATTENTION_VIZ_AVAILABLE = True
    print("✅ attention_viz module loaded successfully")
except ImportError as e:
    print(f"❌ attention_viz not available: {e}")
    ATTENTION_VIZ_AVAILABLE = False
    sys.exit(1)

def test_basic_attention_extraction():
    """Test basic attention extraction from a simple model"""
    print("🔧 Testing basic attention extraction...")
    
    # Use a smaller model for testing
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    try:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for stability
            device_map="cpu",  # Force CPU for debugging
            low_cpu_mem_usage=True
        )
        model.eval()
        
        print(f"✅ Model loaded successfully")
        
        # Test prompt
        test_prompt = "What is 2 + 2?"
        test_response = "2 + 2 = 4"
        
        # Tokenize
        full_text = test_prompt + test_response
        inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
        
        print(f"📏 Input shape: {inputs['input_ids'].shape}")
        print(f"📏 Input tokens: {inputs['input_ids'].shape[1]}")
        
        # Test attention extraction
        print("🔍 Testing attention extraction...")
        
        with torch.no_grad():
            # Enable attention output
            model.config.output_attentions = True
            if hasattr(model.config, '_attn_implementation'):
                original_impl = model.config._attn_implementation
                model.config._attn_implementation = 'eager'
            
            outputs = model(**inputs, output_attentions=True)
            
            # Check attention weights
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                print(f"✅ Extracted attention from {len(outputs.attentions)} layers")
                
                for i, attn in enumerate(outputs.attentions):
                    print(f"   Layer {i}:")
                    print(f"     Shape: {attn.shape}")
                    print(f"     Min: {attn.min().item():.6f}")
                    print(f"     Max: {attn.max().item():.6f}")
                    print(f"     Mean: {attn.mean().item():.6f}")
                    print(f"     Has NaN: {torch.isnan(attn).any()}")
                    print(f"     Has Inf: {torch.isinf(attn).any()}")
                    
                    # Check if attention sums to 1 (approximately)
                    attn_sum = attn.sum(dim=-1)
                    print(f"     Attention sum range: {attn_sum.min().item():.6f} - {attn_sum.max().item():.6f}")
                    
                return True, model, tokenizer, outputs.attentions
            else:
                print("❌ No attention weights extracted")
                return False, None, None, None
                
    except Exception as e:
        print(f"❌ Basic attention extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

def test_attrieval_components(model, tokenizer, attention_weights):
    """Test ATTRIEVAL components step by step"""
    print("\n🔧 Testing ATTRIEVAL components...")
    
    try:
        # Test AttentionExtractor
        print("Testing AttentionExtractor...")
        extractor = AttentionExtractor(model, tokenizer)
        print("✅ AttentionExtractor created successfully")
        
        # Test AttrievelConfig
        print("Testing AttrievelConfig...")
        config = AttrievelConfig(
            layer_fraction=0.5,       # Use more layers for testing
            top_k=5,                  # Smaller top_k for testing
            frequency_threshold=0.9,  # Lower threshold
            max_facts=5               # Fewer facts for testing
        )
        print("✅ AttrievelConfig created successfully")
        
        # Test AttrievelRetriever
        print("Testing AttrievelRetriever...")
        retriever = AttrievelRetriever(extractor, config)
        print("✅ AttrievelRetriever created successfully")
        
        return True, retriever
        
    except Exception as e:
        print(f"❌ ATTRIEVAL component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_fact_retrieval(retriever):
    """Test fact retrieval with simple inputs"""
    print("\n🔧 Testing fact retrieval...")
    
    test_context = "A patient is 85 years old and has diabetes."
    test_question = "What is the patient's age?"
    test_response = "The patient is 85 years old."
    
    try:
        print(f"Context: {test_context}")
        print(f"Question: {test_question}")
        print(f"Response: {test_response}")
        
        # Test retrieval
        result = retriever.retrieve_facts(
            context=test_context,
            question=test_question,
            cot_response=test_response,
            use_cross_evaluation=True
        )
        
        print(f"✅ Retrieval completed")
        print(f"Result keys: {list(result.keys())}")
        
        facts = result.get('retrieved_facts', [])
        print(f"Retrieved {len(facts)} facts:")
        
        for i, fact in enumerate(facts):
            score = fact.get('score', 'No score')
            text = fact.get('text', 'No text')
            print(f"  Fact {i}: score={score}, text='{text[:50]}...'")
            
            # Check for NaN
            if isinstance(score, (int, float)):
                if score != score:  # NaN check
                    print(f"    ⚠️  NaN score detected!")
                elif score == float('inf') or score == float('-inf'):
                    print(f"    ⚠️  Infinite score detected!")
                else:
                    print(f"    ✅ Valid score: {score}")
            else:
                print(f"    ⚠️  Non-numeric score: {type(score)}")
        
        return True, facts
        
    except Exception as e:
        print(f"❌ Fact retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def main():
    """Main debug function"""
    print("🚀 Starting attention analysis debug session")
    print("=" * 60)
    
    # Test 1: Basic attention extraction
    success, model, tokenizer, attention_weights = test_basic_attention_extraction()
    if not success:
        print("❌ Basic attention extraction failed, stopping here")
        return
    
    # Test 2: ATTRIEVAL components
    success, retriever = test_attrieval_components(model, tokenizer, attention_weights)
    if not success:
        print("❌ ATTRIEVAL components failed, stopping here")
        return
    
    # Test 3: Fact retrieval
    success, facts = test_fact_retrieval(retriever)
    if not success:
        print("❌ Fact retrieval failed")
        return
    
    print("\n" + "=" * 60)
    print("🎉 Debug session completed successfully!")
    
    # Summary
    nan_count = sum(1 for fact in facts if isinstance(fact.get('score'), (int, float)) and fact.get('score') != fact.get('score'))
    inf_count = sum(1 for fact in facts if isinstance(fact.get('score'), (int, float)) and abs(fact.get('score')) == float('inf'))
    valid_count = len(facts) - nan_count - inf_count
    
    print(f"📊 Score Analysis:")
    print(f"  Total facts: {len(facts)}")
    print(f"  Valid scores: {valid_count}")
    print(f"  NaN scores: {nan_count}")
    print(f"  Infinite scores: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("⚠️  Issues detected! Check the output above for details.")
    else:
        print("✅ No NaN or infinite scores detected!")

if __name__ == "__main__":
    main() 