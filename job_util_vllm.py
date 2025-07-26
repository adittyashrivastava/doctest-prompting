import sys
import os
import transformers
import torch
import json
from contextlib import redirect_stdout
import tqdm
from run_eval import check_answer
from run_eval import echo
import arg_util
import local_model_util

# Import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    print("✅ vLLM module loaded successfully")
except ImportError as e:
    print(f"⚠️  vLLM not available: {e}")
    print("Install with: pip install vllm>=0.8.3")
    VLLM_AVAILABLE = False

# Import attention_viz for attention analysis and ATTRIEVAL
try:
    from attention_viz import AttentionExtractor, AttrievelRetriever, AttrievelConfig
    ATTENTION_VIZ_AVAILABLE = True
    print("✅ attention_viz module loaded successfully")
except ImportError as e:
    print(f"⚠️  attention_viz not available: {e}")
    ATTENTION_VIZ_AVAILABLE = False

# TODO: doctests

#PROMPT TEMPLATES FOR DIFFERENT MODELS
META_LLAMA_3_1_PROMPT = "<|begin_of_text|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
CODELLAMA_INSTRUCT_PROMPT = "<s>[INST] {prompt} [/INST]"
CODELLAMA_PYTHON_PROMPT = "[INST]\n{prompt}\n[/INST]\nResponse:"

def fetch_prompts(args):
    """Opens a properly-formatted json file and returns the information it contains.
    """
    # Use variant if provided, otherwise use "baseline"
    variant = getattr(args, 'variant', '') or ''
    base_name = variant if variant else "baseline"

    if args.CoT:
        filename = f"{base_name}-cot"
    else:
        filename = f"{base_name}-dpt"
    if args.lo == 30 and args.hi == 0:
        filename += "-tune.json"
    elif args.lo == 0 and args.hi == 30:
        filename += "-dev.json"
    else:
        filename += f"-{args.lo:03}-{args.hi:03}.json"
    filename = f"{args.task_dir}/{args.task}/{filename}"

    local_model_util.build_json(args)

    with open(filename, 'r') as infile:
        json_file = json.loads(infile.read())

    prompt_template = json_file["prompt_template"]
    tasks = json_file["tasks"]
    prompts = []
    prompt_info = []

    model_prompt = "{prompt}"
    if "Meta-Llama-3.1" in args.model:
        model_prompt = META_LLAMA_3_1_PROMPT
    elif "CodeLlama" in args.model and "Instruct" in args.model:
        model_prompt = CODELLAMA_INSTRUCT_PROMPT
    elif "CodeLlama" in args.model and "Python" in args.model:
        model_prompt = CODELLAMA_PYTHON_PROMPT

    for task in tqdm.tqdm(tasks, delay=1):
        input = task["input"]
        prompt = model_prompt.format(prompt = task["prompt"])
        target = task["target"]
        prompts.append(prompt)
        prompt_info.append({"input": input, "target": target})

    return prompt_template, prompts, prompt_info

def setup_vllm_model(args):
    """Setup vLLM model for high-performance inference"""
    if not VLLM_AVAILABLE:
        print("❌ vLLM not available. Falling back to transformers pipeline.")
        return None, None, None

    model = args.model
    if "/" in model:
        args.service = args.model.split("/")[0]
        args.model = args.model.split("/")[-1]

    # Check if model is a local path (e.g., from huggingface folder)
    if os.path.exists(model):
        print(f"📁 Using local model path: {model}")
    else:
        # Check if it's in the codebase/huggingface folder
        huggingface_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "huggingface", model)
        if os.path.exists(huggingface_path):
            model = huggingface_path
            print(f"📁 Using local huggingface model path: {model}")

    try:
        print("🔧 Loading vLLM model...")

        # Initialize vLLM model
        llm = LLM(
            model=model,
            tensor_parallel_size=getattr(args, 'tensor_parallel_size', 1),
            gpu_memory_utilization=getattr(args, 'gpu_memory_utilization', 0.9),
            max_model_len=getattr(args, 'max_model_len', 4096),
            trust_remote_code=getattr(args, 'trust_remote_code', False),
            dtype=getattr(args, 'dtype', 'auto')
        )

        tokenizer = llm.get_tokenizer()

        print(f"✅ vLLM model loaded successfully")
        print(f"   - Model: {model}")
        print(f"   - Tensor parallel size: {getattr(args, 'tensor_parallel_size', 1)}")
        print(f"   - Max model length: {getattr(args, 'max_model_len', 4096)}")

        return llm, tokenizer, model

    except Exception as e:
        print(f"❌ Failed to load vLLM model: {e}")
        return None, None, None

def setup_attention_analysis(model, tokenizer, args):
    """Setup attention analysis components if enabled"""
    if not args.enable_attention_analysis or not ATTENTION_VIZ_AVAILABLE:
        return None, None

    try:
        print("🔧 Setting up attention analysis...")

        # Initialize attention extractor
        extractor = AttentionExtractor(model, tokenizer)

        # Initialize ATTRIEVAL with simple config
        config = AttrievelConfig(
            layer_fraction=0.25,      # Use last 25% of layers
            top_k=10,                 # Top 10 tokens per CoT token
            frequency_threshold=0.99, # Filter attention sinks
            max_facts=10              # Retrieve top 10 facts
        )
        retriever = AttrievalRetriever(extractor, config)

        # Setup output directory
        log_file = arg_util.log_file(args)
        log_dir = os.path.dirname(log_file)
        output_dir = os.path.join(log_dir, "attention_analysis")
        os.makedirs(output_dir, exist_ok=True)

        print(f"✅ Attention analysis setup complete")
        print(f"📁 Results will be saved to: {output_dir}")

        return retriever, output_dir

    except Exception as e:
        print(f"❌ Failed to setup attention analysis: {e}")
        return None, None

def perform_attention_analysis_vllm(prompt, response, info, retriever, output_dir, example_idx, llm, tokenizer):
    """Perform attention analysis for a single example using vLLM"""
    try:
        print(f"🔍 Analyzing example {example_idx}...")

        # Step 1: Extract attention using vLLM
        print(f"🔍 Step 1: Extracting attention weights via vLLM...")

        # Combine prompt and response for attention extraction
        full_text = prompt + response

        # Tokenize the full text
        inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)

        # For vLLM, we need to use a separate forward pass for attention extraction
        # This is a simplified approach - in practice, you might need to modify vLLM
        # to expose attention weights directly

        # For now, we'll use the original transformers approach for attention extraction
        # but keep vLLM for generation
        print(f"⚠️  Using transformers for attention extraction (vLLM attention extraction not yet implemented)")

        # Create a temporary transformers model for attention extraction
        temp_model = transformers.AutoModelForCausalLM.from_pretrained(
            llm.llm_engine.model_config.model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            offload_buffers=True
        )

        inputs = {k: v.to(temp_model.device) for k, v in inputs.items()}

        # Forward pass with attention extraction
        with torch.no_grad():
            # Temporarily enable attention extraction
            original_output_attentions = getattr(temp_model.config, 'output_attentions', False)
            temp_model.config.output_attentions = True

            # Use eager attention for extraction
            original_attn_implementation = getattr(temp_model.config, '_attn_implementation', None)
            temp_model.config._attn_implementation = 'eager'

            outputs = temp_model(**inputs, output_attentions=True)

            # Restore original settings
            temp_model.config.output_attentions = original_output_attentions
            if original_attn_implementation is not None:
                temp_model.config._attn_implementation = original_attn_implementation
            else:
                # Remove the attribute if it wasn't set originally
                if hasattr(temp_model.config, '_attn_implementation'):
                    delattr(temp_model.config, '_attn_implementation')

        # Step 2: Run ATTRIEVAL fact retrieval with extracted attention
        context = info["input"]

        # Create a modified retriever that uses the pre-extracted attention
        retrieval_result = retriever.retrieve_facts(
            context=context,
            question=context,  # For doctest problems, question is same as context
            cot_response=response,
            use_cross_evaluation=True,
            attention_weights=outputs.attentions if hasattr(outputs, 'attentions') else None,
            input_ids=inputs['input_ids'],
            tokenizer=tokenizer
        )

        # Save results
        example_dir = os.path.join(output_dir, f"example_{example_idx:04d}")
        os.makedirs(example_dir, exist_ok=True)

        # Save top facts
        top_facts = {
            "example_idx": example_idx,
            "input": context,
            "response": response,
            "retrieved_facts": retrieval_result['retrieved_facts'],
            "num_facts": len(retrieval_result['retrieved_facts'])
        }

        with open(os.path.join(example_dir, "top_facts.json"), "w") as f:
            json.dump(top_facts, f, indent=2)

        print(f"✅ Analysis complete for example {example_idx} - {len(retrieval_result['retrieved_facts'])} facts retrieved")
        return example_dir

    except Exception as e:
        print(f"❌ Analysis failed for example {example_idx}: {e}")
        return None

def generate_with_vllm(llm, prompts, args):
    """Generate responses using vLLM"""
    if not VLLM_AVAILABLE or llm is None:
        print("❌ vLLM not available, falling back to transformers")
        return None

    try:
        print(f"🚀 Generating {len(prompts)} responses with vLLM...")

        # Calculate max_new_tokens based on prompt lengths
        new_tokens = [len(prompt) for prompt in prompts]
        if len(new_tokens) == 0:
            max_new_tokens = 100
        else:
            max_new_tokens = sum(new_tokens) // len(new_tokens) // 6

        print(f"📊 Generating with max_new_tokens={max_new_tokens}")

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=getattr(args, 'temperature', 0.7),
            top_p=getattr(args, 'top_p', 0.6),
            max_tokens=max_new_tokens,
            stop=getattr(args, 'stop_tokens', None)
        )

        # Generate responses
        outputs = llm.generate(prompts, sampling_params)

        # Extract generated text
        generations = []
        for output in outputs:
            generated_text = output.outputs[0].text
            generations.append({"generated_text": generated_text})

        print(f"✅ vLLM generation completed successfully")
        return generations

    except Exception as e:
        print(f"❌ vLLM generation failed: {e}")
        return None

def main():
    parser = arg_util.baseparser()
    parser.add_argument(
        '--task_dir',
        default='./tasks',
        help='the directory to look for JSON files in.')
    parser.add_argument(
        '--enable_attention_analysis',
        action='store_true',
        help='Enable attention analysis and ATTRIEVAL for each prompt-response pair')

    # vLLM-specific arguments
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism')
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.9,
        help='Fraction of GPU memory to use')
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=4096,
        help='Maximum model length')
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code for model loading')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        help='Model dtype (auto, float16, float32)')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature')
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.6,
        help='Top-p sampling parameter')
    parser.add_argument(
        '--stop_tokens',
        nargs='*',
        help='Stop tokens for generation')

    args = parser.parse_args()

    # fetch information from a json file
    prompt_template, prompts, prompt_info = fetch_prompts(args)

    # Setup vLLM model
    llm, tokenizer, model_path = setup_vllm_model(args)

    if llm is None:
        print("❌ Failed to setup vLLM model. Exiting.")
        return

    # Setup attention analysis
    retriever, attention_output_dir = setup_attention_analysis(
        model_path,  # Use the original model path for attention analysis
        tokenizer,
        args
    )

    # Generate responses using vLLM
    generations = generate_with_vllm(llm, prompts, args)

    if generations is None:
        print("❌ Generation failed. Exiting.")
        return

    log_file = arg_util.log_file(args)
    with open(log_file, 'w') as log:
        echo(log, f"task: {args.task}, lo={args.lo}, hi={args.hi}, model={args.model}")
        echo(log, "=" * 30 + "prompt template with program" + "=" * 30)
        echo(log, f"{prompt_template}")
        echo(log, "=" * 90)
        echo(log, f"Evaluating on {len(prompts)} examples {args.lo}-{args.hi}")
        echo(log, f"Using vLLM for high-performance inference")

        #Output is also saved as a .json file.
        json_log = {
            "task" : args.task,
            "lo" : args.lo,
            "hi" : args.hi,
            "CoT" : args.CoT,
            "test_set" : args.test_set,
            "template_file" : args.template_file,
            "baseline_template_format" : args.baseline_template_format,
            "prompt_template" : prompt_template,
            "inference_backend" : "vllm",
            "vllm_config" : {
                "tensor_parallel_size": getattr(args, 'tensor_parallel_size', 1),
                "gpu_memory_utilization": getattr(args, 'gpu_memory_utilization', 0.9),
                "max_model_len": getattr(args, 'max_model_len', 4096),
                "temperature": getattr(args, 'temperature', 0.7),
                "top_p": getattr(args, 'top_p', 0.6)
            }
        }
        tasks = []

        acc_correct = 0
        acc_total = 0
        acc_parse_failures = 0
        for example_idx, (info, generation, prompt) in enumerate(zip(prompt_info, generations, prompts)):
            #Output is set as a variable because it gets used twice - in echo() and in check_answer()
            output = generation["generated_text"]

            echo(log, "-" * 30 + " input " + "-" * 30)
            echo(log, info["input"])
            echo(log, "-" * 30 + " output " + "-" * 30)
            echo(log, output)
            echo(log, "-" * 30 + " results " + "-" * 30)

            prediction, is_correct, parse_failed = check_answer(args, output, info["target"])
            if is_correct:
                acc_correct += 1
            elif parse_failed:
                acc_parse_failures += 1
            acc_total += 1

            echo(log, f"prediction={prediction} target={info['target']} is_correct={str(is_correct)}")
            echo(log, f"correct={str(acc_correct)} total={str(acc_total)} parse_failures={str(acc_parse_failures)}")

            # Perform attention analysis if enabled
            attention_analysis_dir = None
            if retriever is not None and attention_output_dir is not None:
                attention_analysis_dir = perform_attention_analysis_vllm(
                    prompt, output, info, retriever, attention_output_dir, example_idx, llm, tokenizer
                )

            task = {
                "input" : info["input"],
                "output" : output,
                "prediction" : prediction,
                "target" : info["target"],
                "is_correct" : is_correct,
                "example_idx" : example_idx
            }

            # Add attention analysis info if available
            if attention_analysis_dir is not None:
                task["attention_analysis_dir"] = os.path.relpath(attention_analysis_dir, os.path.dirname(log_file))
                task["has_attention_analysis"] = True
            else:
                task["has_attention_analysis"] = False

            tasks.append(task)
        json_log["tasks"] = tasks

        acc = acc_correct / acc_total
        adj_acc = acc
        if acc_parse_failures and acc_parse_failures != acc_total:
            adj_acc = acc_correct / (acc_total - acc_parse_failures)
        results = {
            "correct" : acc_correct,
            "total" : acc_total,
            "acc" : acc,
            "parse_failures" : acc_parse_failures,
            "adj_acc" : adj_acc
        }
        json_log["results"] = results

        echo(log, "=" * 30 + "Final Totals" + "=" * 30)
        echo(log, f"correct={str(acc_correct)} total={str(acc_total)} acc={str(acc)}")
        echo(log, f"parse_failures={str(acc_parse_failures)} adj_acc={str(adj_acc)}")

        # Report attention analysis summary
        if retriever is not None:
            num_with_attention = sum(1 for task in tasks if task.get("has_attention_analysis", False))
            echo(log, "=" * 30 + "Attention Analysis Summary" + "=" * 30)
            echo(log, f"Attention analysis completed for {num_with_attention}/{len(tasks)} examples")
            echo(log, f"Results saved to: {attention_output_dir}")

    with open(log_file.replace(".log", ".json"), "w") as outfile:
        json.dump(json_log, outfile, indent = 4)

    print(f"\n{'='*50}")
    print(f"✅ vLLM Processing completed!")
    print(f"📊 Results: {log_file}")
    print(f"📈 Accuracy: {acc:.3f} ({acc_correct}/{acc_total})")
    if retriever is not None:
        num_with_attention = sum(1 for task in tasks if task.get("has_attention_analysis", False))
        print(f"🔍 Attention analysis: {num_with_attention}/{len(tasks)} examples")
        print(f"📁 Attention results: {attention_output_dir}")
    print(f"⚡ Inference backend: vLLM")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()