#!/usr/bin/env python3
"""
Debug script to validate tokenizer and basic model operations
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def test_tokenizer_and_model():
    """Test basic tokenizer and model operations"""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print("🔍 Testing tokenizer and model...")
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        
        # Test tokenization
        test_text = "What is 2 + 2?"
        print(f"\n🧪 Testing tokenization with: '{test_text}'")
        
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=100)
        input_ids = inputs['input_ids']
        
        print(f"   Input IDs shape: {input_ids.shape}")
        print(f"   Input IDs: {input_ids.tolist()}")
        print(f"   Min ID: {input_ids.min().item()}")
        print(f"   Max ID: {input_ids.max().item()}")
        
        # Validate token IDs
        if input_ids.min() < 0:
            print("❌ ERROR: Negative token IDs detected!")
            return False
        
        if input_ids.max() >= tokenizer.vocab_size:
            print(f"❌ ERROR: Token IDs exceed vocab size! Max: {input_ids.max()}, Vocab: {tokenizer.vocab_size}")
            return False
        
        print("✅ Token IDs are valid")
        
        # Test decoding
        decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"   Decoded: '{decoded}'")
        
        if device == "cuda":
            print(f"\n🖥️  Testing GPU operations...")
            
            # Load model
            print("🔧 Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="eager"
            )
            print("✅ Model loaded successfully")
            
            # Move inputs to GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print("✅ Inputs moved to GPU")
            
            # Test simple generation
            print("🤖 Testing simple generation...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            print(f"✅ Generated: '{generated}'")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            print("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting tokenizer and model validation")
    print("=" * 50)
    
    success = test_tokenizer_and_model()
    
    if success:
        print("\n🎉 All tests passed! Your setup should work for attention testing.")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        
    print("\nNext steps:")
    print("- If tests passed: Run 'python run_attention_test.py'")
    print("- If tests failed: Check CUDA installation and model compatibility") 