#!/usr/bin/python3

import dataclasses
import json
import functools
import os
import re
import string
import time

import anthropic
import google.generativeai as genai
import groq
import llamaapi
import mistralai.client
import mistralai.models
import openai
import together

# infrastructure for checkpointing LLMs results and running against
# multiple models

@dataclasses.dataclass
class CheckPointer:
  """Context manager for checkpointing while you accumulate results.

  Example of using this:
  def run_eval(data, prompt_fn, parse_fn, delay=0.0, service='groq', **kw):
    # prompt_fn creates an LLM input from a 'datapoint', which is a dict
    # parse_fn postprocesses that to get a short summary
    with CheckPointer(**kw) as cp:
      for d in data:
        long_answer = llm(prompt_fn(d), service=service)
        short_answer = parse_fn(long_answer)
        d.update(short_answer=short_answer, long_answer=long_answer)
        # append the updated dictionary to the checkpoint
        cp.append(d)
        time.sleep(delay)
    # these are all the updated d's but they are
    # also written in little batchs out to disk so not
    # much is lost if something crashes
    return cp.results
  """

  # write data after every k_interval calls to 'append'
  # or after every time_interval seconds
  k_interval: int = 0
  time_interval: float = 60.0
  # data is checkpointed to files with this name
  filestem: str = 'checkpt'
  # internal book-keeping, only need to set when you re-start
  last_chkpt_k: int = 0
  last_chkpt_time: float = 0.0
  k: int = 0
  # all the results appended so far
  results: list = dataclasses.field(default_factory=list)

  def __enter__(self):
    self.last_chkpt_time = time.time()
    print(f'initialize Checkpointing: {self.k=} {self.filestem=}')
    return self

  def __exit__(self, _type, _value, _traceback):
    self.flush(forced=True)
  
  def _num_ready_to_flush(self, forced):
    """Return (n, flush_now) where n is number of unflushed examples,
    and flush_now is true if that's enough to flush.
    """
    num_unflushed = self.k - self.last_chkpt_k
    if self.k_interval:
      if self.k >= self.last_chkpt_k + self.k_interval:
        return num_unflushed, True
    if self.time_interval:
      if time.time() >= self.last_chkpt_time + self.time_interval:
        return num_unflushed, True
    return num_unflushed, forced

  def flush(self, forced=False):
    """Write out unflushed results to a file.
    """
    num, flush_now = self._num_ready_to_flush(forced)
    if not flush_now or num==0:
      return
    elapsed = time.time() - self.last_chkpt_time
    hi = self.k
    lo = self.k - num
    file = f'{self.filestem}.{lo:04d}-{hi:04d}.jsonl'
    with open(file, 'w') as fp:
      for r in self.results[lo:hi]:
        fp.write(json.dumps(r) + '\n')
    print(f'write {file} with {num} outputs in {elapsed:.2f} sec')
    self.last_chkpt_k = self.k
    self.last_chkpt_time = time.time()
  
  def append(self, result):
    """Append a new result to the running list."""
    self.results.append(result)
    self.k += 1
    self.flush(forced=False)


def llm_with_model(prompt, service='groq', model=None):
  """Use an LLM model and return response with model objects for attention analysis.
  
  For local service, returns (response, model_obj, tokenizer).
  For other services, returns (response, None, None).
  """
  if service == 'local':
    # Local model inference using transformers with model object return
    if not model:
      model = 'Qwen/Qwen2.5-7B-Instruct'
    
    try:
      import torch
      import transformers
      from transformers import AutoTokenizer, AutoModelForCausalLM
      
      print(f'Loading local model {model} for attention analysis...')
      
      # Load tokenizer and model with specific configuration for attention extraction
      tokenizer = AutoTokenizer.from_pretrained(model)
      
      # Ensure we have padding token
      if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
      
      # Determine if this is a large model for special handling
      is_large_model = '7B' in model or '13B' in model or '70B' in model
      
      # Use GPU with float32 for optimal stability and speed
      if torch.cuda.is_available():
        print(f"🔧 Using CUDA + float32 for model {model} (optimal stability)")
        torch_dtype = torch.float32
        device_map = "auto"
      else:
        print(f"🔧 Using CPU + float32 for model {model}")
        torch_dtype = torch.float32
        device_map = None
      
      model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        # Configure for attention extraction
        output_attentions=False,  # We'll enable this during inference
        attn_implementation="eager"  # Force eager attention for better extraction
      )
      
      # Ensure model is in eval mode for consistent attention patterns
      model_obj.eval()
      
      print(f'Model {model} loaded successfully for attention analysis')
      print(f'Model device: {next(model_obj.parameters()).device}')
      print(f'Model dtype: {next(model_obj.parameters()).dtype}')
      
      # Format prompt using chat template if available
      messages = [{"role": "user", "content": prompt}]
      try:
        formatted_prompt = tokenizer.apply_chat_template(
          messages, 
          tokenize=False, 
          add_generation_prompt=True
        )
      except Exception as e:
        print(f"Warning: Chat template failed ({e}), using fallback format")
        # Fallback format for models without chat template
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
      
      # Generate response with stabilized parameters for large models
      inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
      
      # Move inputs to same device as model
      device = next(model_obj.parameters()).device
      inputs = {k: v.to(device) for k, v in inputs.items()}
      
      print(f"Input tokens: {inputs['input_ids'].shape[1]}")
      
      with torch.no_grad():
        # Use standard generation parameters - float32 provides excellent stability
        generation_kwargs = {
          "max_new_tokens": 2048,
          "do_sample": True,
          "temperature": 0.7,
          "top_p": 0.9,
          "top_k": 50,
          "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
          "use_cache": True,
          "repetition_penalty": 1.1,
          "no_repeat_ngram_size": 3
        }
        
        print(f"🔧 Using standard generation parameters with float32 stability")
        
        try:
          outputs = model_obj.generate(**inputs, **generation_kwargs)
          
        except RuntimeError as e:
          if "out of memory" in str(e).lower():
            print(f"⚠️  GPU OOM, falling back to CPU for generation...")
            # Move model to CPU for generation
            model_obj = model_obj.cpu()
            inputs = {k: v.cpu() for k, v in inputs.items()}
            generation_kwargs["max_new_tokens"] = 1024  # Reduce for CPU
            outputs = model_obj.generate(**inputs, **generation_kwargs)
          else:
            raise e
      
      # Decode response (only the new tokens)
      response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
      
      print(f"Generated response length: {len(response)} characters")
      
      # DON'T clean up model from memory - return it for attention analysis
      return response.strip(), model_obj, tokenizer
      
    except Exception as e:
      print(f"Error in local inference with model return: {str(e)}")
      import traceback
      traceback.print_exc()
      return f"Error in local inference: {str(e)}", None, None
  else:
    # For non-local services, use regular llm function and return None for model objects
    response = llm(prompt, service=service, model=model)
    return response, None, None

def llm(prompt, service='groq', model=None):
  """Use an LLM model.
  """
  if service == 'openai':
    if model is None: model='gpt-4o-mini'
    client = openai.OpenAI()
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    return completion.choices[0].message.content
  elif service=='llama_api_via_openai':
    if model is None: model='llama3.1-70b'
    client = openai.OpenAI(
      api_key = os.environ['LLAMA_API_KEY'],
      base_url = "https://api.llama-api.com"
    )
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": prompt}
      ],
    )
    return completion.choices[0].message.content    
  elif service=='llama_api':
    if model is None: model='llama3.1-70b'
    llama = llamaapi.LlamaAPI(os.environ['LLAMA_API_KEY'])
    api_request_json = {
      "model": "llama3.1-405b",
      "messages": [
        {"role": "user", "content": prompt},
      ],
      "stream": False,
      "max_tokens": 2048,
    }
    response = llama.run(api_request_json)
    content = response.json()['choices'][0]['message']['content']
    return content
  elif service == 'deepseek':
    if model is None: model='deepseek-chat'
    client = openai.OpenAI(
      base_url='https://api.deepseek.com',
      api_key=os.environ["DEEPSEEK_API_KEY"]
    )
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    return completion.choices[0].message.content
  elif service == 'gemini':
    if not model: model = 'gemini-1.5-flash'
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)
    try:
      num_output_tokens = model.count_tokens(response.text)
      print(f'**** {num_output_tokens} tokens')
    except ValueError as ex:
      print(ex)
      return '** Gemini response blocked **'
    return response.text
  elif service == 'groq':
    if not model: model = 'mixtral-8x7b-32768'
    client = groq.Groq(api_key=os.environ.get('GROQ_API_KEY'))
    completion = client.chat.completions.create(
      messages = [dict(role='user', content=prompt)],
      model=model)
    return completion.choices[0].message.content
  elif service == 'mistral':
    if not model: model='codestral-latest' #open-mixtral-8x7b
    client = mistralai.client.MistralClient(
      api_key=os.environ.get('MISTRAL_API_KEY'))
    chat_response = client.chat(
      messages=[mistralai.models.chat_completion.ChatMessage(
        role='user', content=prompt)],
      model=model)
    return chat_response.choices[0].message.content
  elif service == 'anthropic':
    if not model: model='claude-3-5-sonnet-20240620'  #claude-3-haiku-20240307, claude-3-sonnet-20240229
    print(f'Using model {model}')
    client = anthropic.Anthropic(
      api_key=os.environ.get('ANTHROPIC_API_KEY'))
    try:
      message = client.messages.create(
        max_tokens=4096,
        model=model,
        #temperature=0,  #changed from default of 1.0 on 6/27
        messages=[{"role": "user", "content": prompt}])
      return message.content[0].text
    except anthropic.BadRequestError as ex:
      return repr(ex)
  elif service == 'together':
    client = together.Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    if not model: model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    completion = client.chat.completions.create(
      model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
      max_tokens=8192,
      messages=[
        {"role": "user", 
         "content": prompt,
         }
      ],
    )
    return completion.choices[0].message.content
  elif service == 'local':
    # Local model inference using transformers
    if not model:
      model = 'Qwen/Qwen2.5-7B-Instruct'
    
    try:
      import torch
      import transformers
      from transformers import AutoTokenizer, AutoModelForCausalLM
      
      print(f'Loading local model {model}...')
      
      # Load tokenizer and model
      tokenizer = AutoTokenizer.from_pretrained(model)
      model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
      )
      
      # Format prompt using chat template if available
      messages = [{"role": "user", "content": prompt}]
      try:
        formatted_prompt = tokenizer.apply_chat_template(
          messages, 
          tokenize=False, 
          add_generation_prompt=True
        )
      except Exception:
        # Fallback format for models without chat template
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
      
      # Generate response
      inputs = tokenizer(formatted_prompt, return_tensors="pt")
      if torch.cuda.is_available():
        inputs = {k: v.to(model_obj.device) for k, v in inputs.items()}
      
      with torch.no_grad():
        outputs = model_obj.generate(
          **inputs,
          max_new_tokens=2048,
          do_sample=True,
          temperature=0.7,
          top_p=0.9,
          pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
        )
      
      # Decode response (only the new tokens)
      response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
      
      # Clean up model from memory
      del model_obj
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      
      return response.strip()
      
    except Exception as e:
      return f"Error in local inference: {str(e)}"
  elif service == 'null':
    return 'null service was used - no answer'
  else:
    raise ValueError(f'invalid service {service}')
