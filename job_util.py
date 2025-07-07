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
        retriever = AttrievelRetriever(extractor, config)

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

def perform_attention_analysis(prompt, response, info, retriever, output_dir, example_idx, model_obj, tokenizer):
    """Perform attention analysis for a single example using separate forward pass"""
    try:
        print(f"🔍 Analyzing example {example_idx}...")

        # Step 1: Extract attention using separate forward pass
        print(f"🔍 Step 1: Extracting attention weights via separate forward pass...")

        # Combine prompt and response for attention extraction
        full_text = prompt + response

        # Tokenize the full text
        inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model_obj.device) for k, v in inputs.items()}

        # Forward pass with attention extraction
        with torch.no_grad():
            # Temporarily enable attention extraction
            original_output_attentions = getattr(model_obj.config, 'output_attentions', False)
            model_obj.config.output_attentions = True

            # Use eager attention for extraction
            original_attn_implementation = getattr(model_obj.config, '_attn_implementation', None)
            model_obj.config._attn_implementation = 'eager'

            outputs = model_obj(**inputs, output_attentions=True)

            # Restore original settings
            model_obj.config.output_attentions = original_output_attentions
            if original_attn_implementation is not None:
                model_obj.config._attn_implementation = original_attn_implementation
            else:
                # Remove the attribute if it wasn't set originally
                if hasattr(model_obj.config, '_attn_implementation'):
                    delattr(model_obj.config, '_attn_implementation')

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
    args = parser.parse_args()

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

    # fetch information from a json file
    prompt_template, prompts, prompt_info = fetch_prompts(args)

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    # Load model for generation (no attention extraction to save memory)
    print("🔧 Loading model for generation...")
    model_obj = transformers.AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    print(f"Pipeline loaded.")

    # Setup attention analysis
    retriever, attention_output_dir = setup_attention_analysis(
        pipeline.model if hasattr(pipeline, 'model') else None,
        tokenizer,
        args
    )

    new_tokens = [len(prompt) for prompt in prompts]
    if len(new_tokens) == 0:
        new_tokens = 100
    else:
        new_tokens = sum(new_tokens) // len(new_tokens)
    new_tokens = new_tokens // 6
    print(f"Generating {len(prompts)} prompts for {args.task} with a maximum of {new_tokens} new tokens each...")

    generations = pipeline(
        prompts,
        do_sample=True,
        top_p=0.6,
        return_full_text=False,
        max_new_tokens = new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    log_file = arg_util.log_file(args)
    with open(log_file, 'w') as log:
        echo(log, f"task: {args.task}, lo={args.lo}, hi={args.hi}, model={args.model}")
        echo(log, "=" * 30 + "prompt template with program" + "=" * 30)
        echo(log, f"{prompt_template}")
        echo(log, "=" * 90)
        echo(log, f"Evaluating on {len(prompts)} examples {args.lo}-{args.hi}")

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
        }
        tasks = []

        acc_correct = 0
        acc_total = 0
        acc_parse_failures = 0
        for example_idx, (info, generation, prompt) in enumerate(zip(prompt_info, generations, prompts)):
            #Output is set as a variable because it gets used twice - in echo() and in check_answer()
            output = generation[0]["generated_text"]

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
                attention_analysis_dir = perform_attention_analysis(
                    prompt, output, info, retriever, attention_output_dir, example_idx, model_obj, tokenizer
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
    print(f"✅ Processing completed!")
    print(f"📊 Results: {log_file}")
    print(f"📈 Accuracy: {acc:.3f} ({acc_correct}/{acc_total})")
    if retriever is not None:
        num_with_attention = sum(1 for task in tasks if task.get("has_attention_analysis", False))
        print(f"🔍 Attention analysis: {num_with_attention}/{len(tasks)} examples")
        print(f"📁 Attention results: {attention_output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
