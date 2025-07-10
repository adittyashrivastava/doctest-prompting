#!/usr/bin/env python3
"""
Local model inference script that can be called separately to avoid import issues.
This script handles the actual model loading and inference.
"""

import sys
import json
import os

def run_local_inference(prompt, model_name):
    """Run inference with a local model."""
    
    try:
        # Set environment variables to handle potential issues
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
        
        # Import here to isolate potential issues
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print(f"Loading model {model_name}...", file=sys.stderr)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        model = model.to('cpu')
        
        print(f"Model {model_name} loaded successfully", file=sys.stderr)
        
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # Fallback format
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        return f"Error in local inference: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python local_inference.py <prompt> <model_name>")
        sys.exit(1)
    
    prompt = sys.argv[1]
    model_name = sys.argv[2]
    
    result = run_local_inference(prompt, model_name)
    print(result) 