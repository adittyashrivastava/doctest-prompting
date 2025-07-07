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
import numpy as np
from datetime import datetime

# Add attention_viz to Python path if available
attention_viz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "attention_viz")
if os.path.exists(attention_viz_path):
    sys.path.insert(0, attention_viz_path)
    print(f"📍 Added attention_viz path: {attention_viz_path}")

# Import attention_viz for attention analysis and ATTRIEVAL
try:
    from attention_viz import AttentionVisualizer, AttentionExtractor, AttentionAnalyzer, AttrievelRetriever, AttrievelConfig
    from attention_viz.utils.helpers import load_model_and_tokenizer
    ATTENTION_VIZ_AVAILABLE = True
    print("✅ attention_viz module imported successfully")
except ImportError as e:
    print(f"⚠️  attention_viz module not available: {e}")
    print("Continuing without attention analysis...")
    ATTENTION_VIZ_AVAILABLE = False

# TODO: doctests

#PROMPT TEMPLATES FOR DIFFERENT MODELS
META_LLAMA_3_1_PROMPT = "<|begin_of_text|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
CODELLAMA_INSTRUCT_PROMPT = "<s>[INST] {prompt} [/INST]"
CODELLAMA_PYTHON_PROMPT = "[INST]\n{prompt}\n[/INST]\nResponse:"

def fetch_prompts(args):
    """Opens a properly-formatted json file and returns the information it contains.
    """
    if args.CoT:
        filename = "baseline-cot"
    else:
        filename = "baseline-dpt"
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
    original_model_name = model
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

    # I believe this is where the actual generation gets submitted and done, so there will be a long hangtime here.
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Pipeline loaded.")

    # Initialize attention analysis components if enabled
    attention_model = None
    attention_tokenizer = None
    shared_extractor = None
    attention_visualizer = None
    attention_analyzer = None
    attrieval_retriever = None
    top_k_output_dir = None

    if args.enable_attention_analysis and ATTENTION_VIZ_AVAILABLE:
        print("🔧 Setting up attention analysis...")

        # Load model separately for attention analysis (use resolved model path)
        attention_model, attention_tokenizer = load_model_for_attention_analysis(model)

        if attention_model is not None and attention_tokenizer is not None:
            # Initialize attention analysis components
            shared_extractor, attention_visualizer, attention_analyzer, attrieval_retriever = setup_attention_analysis(
                attention_model, attention_tokenizer
            )

            if shared_extractor is not None:
                # Setup output directory for top-k facts in doctest-prompting-data structure
                log_file = arg_util.log_file(args)
                log_dir = os.path.dirname(log_file)
                top_k_output_dir = os.path.join(log_dir, "top_k")
                os.makedirs(top_k_output_dir, exist_ok=True)
                print(f"📁 Top-k facts will be saved to: {top_k_output_dir}")
            else:
                print("❌ Failed to initialize attention analysis components")
                args.enable_attention_analysis = False
        else:
            print("❌ Failed to load model for attention analysis")
            args.enable_attention_analysis = False
    elif args.enable_attention_analysis and not ATTENTION_VIZ_AVAILABLE:
        print("⚠️  Attention analysis requested but attention_viz module not available")
        args.enable_attention_analysis = False

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
        for example_idx, (info, generation, prompt) in enumerate(zip(prompt_info, generations, prompts)): #, prompt in zip(prompt_info, prompts)
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
            if args.enable_attention_analysis and shared_extractor is not None:
                try:
                    attention_analysis_dir = perform_attention_analysis(
                        prompt=prompt,
                        response=output,
                        info=info,
                        shared_extractor=shared_extractor,
                        attention_visualizer=attention_visualizer,
                        attention_analyzer=attention_analyzer,
                        attrieval_retriever=attrieval_retriever,
                        output_dir=top_k_output_dir,
                        example_idx=example_idx
                    )
                except Exception as e:
                    print(f"❌ Attention analysis failed for example {example_idx}: {e}")
                    attention_analysis_dir = None

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
                task["attention_analysis_dir"] = os.path.relpath(attention_analysis_dir, os.path.dirname(arg_util.log_file(args)))
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
        if args.enable_attention_analysis and shared_extractor is not None:
            num_with_attention = sum(1 for task in tasks if task.get("has_attention_analysis", False))
            echo(log, "=" * 30 + "Attention Analysis Summary" + "=" * 30)
            echo(log, f"Attention analysis completed for {num_with_attention}/{len(tasks)} examples")
            echo(log, f"Top-k facts saved to: {top_k_output_dir}")

    with open(log_file.replace(".log", ".json"), "w") as outfile:
        json.dump(json_log, outfile, indent = 4)

    # Final summary
    print(f"\n{'='*50}")
    print(f"Processing completed!")
    print(f"📊 Results saved to: {log_file}")
    print(f"📊 JSON data saved to: {log_file.replace('.log', '.json')}")
    print(f"📈 Accuracy: {acc:.3f} ({acc_correct}/{acc_total})")
    if acc_parse_failures > 0:
        print(f"📈 Adjusted accuracy (excluding parse failures): {adj_acc:.3f} ({acc_correct}/{acc_total - acc_parse_failures})")

    if args.enable_attention_analysis:
        if shared_extractor is not None:
            num_with_attention = sum(1 for task in tasks if task.get("has_attention_analysis", False))
            print(f"🔍 Attention analysis completed for {num_with_attention}/{len(tasks)} examples")
            print(f"📁 Top-k facts saved to: {top_k_output_dir}")
            print("   Generated files for each example:")
            print("   - essential_attention_data.npz (compressed attention weights)")
            print("   - attrieval_results.json (comprehensive retrieval results)")
            print("   - attrieval_analysis_report.md (human-readable analysis)")
            print("   - top_facts_summary.json (top retrieved facts)")
            print("   - attrieval_attention_data.npz (attention weights and scores)")
            print("   - analysis_summary.json (metadata and file list)")
        else:
            print("⚠️  Attention analysis was requested but failed to initialize")

    # Clean up attention models to free memory
    if attention_model is not None:
        print("🧹 Cleaning up attention analysis models...")
        del attention_model
        del attention_tokenizer
        del shared_extractor
        del attention_visualizer
        del attention_analyzer
        del attrieval_retriever
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("✅ Memory cleanup completed")

    print(f"{'='*50}\n")

def load_model_for_attention_analysis(model_name):
    """Load model and tokenizer specifically for attention analysis"""
    try:
        print(f"🔧 Loading model for attention analysis: {model_name}")

        # Load model and tokenizer separately (not as pipeline)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            output_attentions=True,  # Critical for attention analysis
            trust_remote_code=True
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✅ Model loaded successfully for attention analysis")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed to load model for attention analysis: {e}")
        return None, None

def setup_attention_analysis(model, tokenizer):
    """Initialize attention analysis components"""
    try:
        print("🔧 Initializing attention analysis components...")

        # Initialize shared attention extractor
        shared_extractor = AttentionExtractor(model, tokenizer)

        # Initialize attention visualization components
        attention_visualizer = AttentionVisualizer(model, tokenizer)
        attention_analyzer = AttentionAnalyzer(shared_extractor)

        # Initialize ATTRIEVAL components
        attrieval_config = AttrievelConfig(
            layer_fraction=0.25,      # Use last 25% of layers
            top_k=10,                 # Top 10 tokens per CoT token
            frequency_threshold=0.99, # Filter attention sinks
            max_facts=10              # Retrieve top 10 facts
        )
        attrieval_retriever = AttrievelRetriever(shared_extractor, attrieval_config)

        # Test basic functionality
        print("🧪 Testing basic attention extraction...")
        test_result = shared_extractor.extract_attention_weights("Hello world, this is a test.")
        print(f"✅ Basic test passed - Model has {test_result['num_layers']} layers, {test_result['num_heads']} heads")

        return shared_extractor, attention_visualizer, attention_analyzer, attrieval_retriever

    except Exception as e:
        print(f"❌ Failed to initialize attention analysis: {e}")
        return None, None, None, None

def perform_attention_analysis(prompt, response, info, shared_extractor, attention_visualizer, attention_analyzer, attrieval_retriever, output_dir, example_idx):
    """Perform attention analysis and ATTRIEVAL for a single prompt-response pair"""
    try:
        print(f"🔍 Performing attention analysis for example {example_idx}...")

        # Create organized output directory for this specific entry
        entry_dir = os.path.join(output_dir, f"example_{example_idx:04d}")
        os.makedirs(entry_dir, exist_ok=True)

        # Combine the full input text for attention analysis
        full_input_text = prompt
        context = info["input"]  # The original problem/context
        question = info["input"]  # Same as context in this case
        cot_response = response  # The model's response

        print(f"📝 Input text length: {len(full_input_text)} characters")
        print(f"🧠 Response length: {len(cot_response)} characters")

        # 1. Export essential attention data (memory-efficient format)
        attention_data = {}
        try:
            print("💾 Exporting essential attention data...")
            essential_data = shared_extractor.extract_attention_weights(full_input_text)

            # Get model architecture info
            num_layers = essential_data["num_layers"]
            num_heads = essential_data["num_heads"]
            target_layer = min(6, num_layers - 1)
            target_head = min(4, num_heads - 1)

            print(f"📊 Model has {num_layers} layers and {num_heads} heads per layer")
            print(f"🎯 Using layer {target_layer} and head {target_head} for analysis")

            # Only keep essential layers to save memory
            essential_layers = [0, target_layer, num_layers-1]
            filtered_attention = []
            for i, layer_attn in enumerate(essential_data["attention_weights"]):
                if i in essential_layers:
                    max_heads_to_keep = min(8, layer_attn.shape[0])
                    filtered_attention.append(layer_attn[:max_heads_to_keep])

            # Create memory-efficient export
            essential_export = {
                "tokens": essential_data["tokens"],
                "num_layers": len(essential_layers),
                "num_heads": max_heads_to_keep,
                "target_layer": target_layer,
                "target_head": target_head,
                "sequence_length": essential_data["sequence_length"],
                "example_idx": example_idx
            }

            # Save as compressed numpy format
            np.savez_compressed(
                os.path.join(entry_dir, "essential_attention_data.npz"),
                attention_weights=np.array(filtered_attention, dtype=object),
                **essential_export
            )
            attention_data["essential_data"] = "essential_attention_data.npz"
            print("✅ Essential attention data export completed")

            # Clear memory
            del essential_data, filtered_attention, essential_export
            import gc
            gc.collect()

        except Exception as e:
            print(f"❌ Essential attention data export failed: {e}")

        # 2. Run ATTRIEVAL fact retrieval
        attrieval_data = {}
        try:
            print("🎯 Running ATTRIEVAL fact retrieval...")
            retrieval_result = attrieval_retriever.retrieve_facts(
                context=context,
                question=question,
                cot_response=cot_response,
                use_cross_evaluation=True
            )

            print(f"📊 Retrieved {len(retrieval_result['retrieved_facts'])} top facts")

            # Save detailed ATTRIEVAL results
            attrieval_retriever.export_retrieval_result(
                retrieval_result,
                os.path.join(entry_dir, "attrieval_results.json")
            )
            attrieval_data["results_json"] = "attrieval_results.json"

            # Generate human-readable report
            readable_report = attrieval_retriever.visualize_retrieved_facts(retrieval_result)
            with open(os.path.join(entry_dir, "attrieval_analysis_report.md"), "w") as f:
                f.write(readable_report)
            attrieval_data["analysis_report"] = "attrieval_analysis_report.md"

            # Save top facts summary
            top_facts_summary = {
                "example_idx": example_idx,
                "input": context,
                "response": cot_response,
                "top_retrieved_facts": retrieval_result['retrieved_facts'],
                "num_facts_retrieved": len(retrieval_result['retrieved_facts']),
                "attrieval_config": retrieval_result['config'],
                "context_length": len(context),
                "response_length": len(cot_response),
                "timestamp": datetime.now().isoformat()
            }

            with open(os.path.join(entry_dir, "top_facts_summary.json"), "w") as f:
                json.dump(top_facts_summary, f, indent=2)
            attrieval_data["top_facts"] = "top_facts_summary.json"

            # Save aggregated attention data
            np.savez_compressed(
                os.path.join(entry_dir, "attrieval_attention_data.npz"),
                aggregated_attention=retrieval_result['aggregated_attention'],
                retriever_tokens=retrieval_result['retriever_tokens'],
                fact_scores=retrieval_result['fact_scores']
            )
            attrieval_data["attention_data"] = "attrieval_attention_data.npz"

            print("✅ ATTRIEVAL analysis completed")

        except Exception as e:
            print(f"❌ ATTRIEVAL analysis failed: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")

        # 3. Save analysis summary
        try:
            analysis_summary = {
                "example_idx": example_idx,
                "input": context,
                "response": cot_response,
                "input_length": len(full_input_text),
                "response_length": len(cot_response),
                "attention_files": attention_data,
                "attrieval_files": attrieval_data,
                "timestamp": datetime.now().isoformat()
            }

            with open(os.path.join(entry_dir, "analysis_summary.json"), "w") as f:
                json.dump(analysis_summary, f, indent=2)

            print(f"✅ Analysis completed for example {example_idx}")
            print(f"📁 Files saved to: {entry_dir}")

        except Exception as e:
            print(f"❌ Analysis summary save failed: {e}")

        return entry_dir

    except Exception as e:
        print(f"❌ Attention analysis failed for example {example_idx}: {e}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main()
