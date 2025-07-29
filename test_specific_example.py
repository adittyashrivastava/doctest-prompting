#!/usr/bin/env python3
"""
Focused test script to debug specific CUDA issues
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set enhanced CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable device-side assertions for better error info

def test_specific_example():
    """Test a specific example that's causing CUDA errors"""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # Example context and question from general English test
    context = """Titanium dioxide is a chemical compound with the formula TiO2. When used as a pigment, it is called titanium white, Pigment White 6 (PW6), or CI 77891. Generally, it is sourced from ilmenite, rutile, and anatase. It has a wide range of applications, including paint, sunscreen, and food coloring. As a food additive, titanium dioxide has E number E171."""
    
    question = "What is the chemical element that forms the basis of titanium dioxide?"
    
    print("🔍 Testing specific problematic example...")
    print(f"📝 Context length: {len(context)} chars")
    print(f"❓ Question: {question}")
    
    try:
        # Load tokenizer
        print("\n📚 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Tokenizer loaded")
        
        # Load model
        print("\n🔧 Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else "cpu",
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        model.eval()
        print("✅ Model loaded")
        
        # Test the exact prompt format
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        print(f"\n📏 Full prompt length: {len(prompt)} chars")
        print(f"📄 First 200 chars of prompt: {prompt[:200]}...")
        
        # Tokenize
        print("\n🔤 Tokenizing...")
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=False,
            add_special_tokens=True
        )
        
        input_ids = inputs['input_ids']
        print(f"📊 Token stats: shape={input_ids.shape}, min={input_ids.min().item()}, max={input_ids.max().item()}")
        print(f"🔢 Vocab size: {tokenizer.vocab_size}")
        
        # Check for problematic tokens
        if input_ids.min() < 0:
            print("❌ ERROR: Negative token IDs!")
            return False
            
        if input_ids.max() >= tokenizer.vocab_size:
            print(f"❌ ERROR: Token ID {input_ids.max().item()} exceeds vocab size {tokenizer.vocab_size}!")
            return False
        
        # Move to device
        print(f"\n🖥️  Moving to {device}...")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Test generation with minimal settings
        print("\n🤖 Testing generation...")
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,  # Very small to avoid memory issues
                    do_sample=False,    # Deterministic
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(
                    outputs[0][input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                print(f"✅ Generation successful: '{generated_text}'")
                return True
                
            except Exception as gen_error:
                print(f"❌ Generation failed: {gen_error}")
                print(f"   Error type: {type(gen_error).__name__}")
                
                # Additional debugging info
                print(f"   Input tensor device: {inputs['input_ids'].device}")
                print(f"   Input tensor dtype: {inputs['input_ids'].dtype}")
                print(f"   Model device: {next(model.parameters()).device}")
                
                return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Specific CUDA Issue")
    print("=" * 50)
    
    success = test_specific_example()
    
    if success:
        print("\n🎉 Test passed! The issue might be elsewhere.")
    else:
        print("\n❌ Test failed. Check the detailed error output above.") 