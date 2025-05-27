#!/bin/bash

TASKS="gsm8k,math_500,gpqa_diamond,aime24,mmlu_pro_health,mmlu_pro_biology,medqa,medcalc_bench_rules,medcalc_bench_formulas,pubmedqa"
# TASKS="medcalc_bench_rules_train,medcalc_bench_formulas_train"
EVAL_TYPE="custom"

# Array of models to evaluate
declare -A models
# models["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]="./evals/DeepSeek-R1-Distill-Qwen-7B 32768"
# models["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="./evals/DeepSeek-R1-Distill-Llama-8B 32768"
# models["Qwen/Qwen2.5-7B"]="./evals/Qwen2.5-7B-Base 32768"
# models["Qwen/Qwen2.5-7B-Instruct"]="./evals/Qwen2.5-7B-Instruct"
# models["PTPReasoning/Qwen2.5-7B-Base-SFT-Baseline-V2"]="./evals/Qwen2.5-7B-Base-SFT-Baseline-V2 32768"
# models["PTPReasoning/Qwen2.5-7B-Base-SFT-Clean-V2"]="./evals/Qwen2.5-7B-Base-SFT-Clean-V2 32768"
# models["PTPReasoning/Qwen2.5-7B-Base-RL-Baseline"]="./evals/Qwen2.5-7B-Base-RL-Baseline 32768"
models["PTPReasoning/Qwen2.5-7B-Base-RL-Clean-V2"]="./evals/Qwen2.5-7B-Base-RL-Clean-V2 32768"
# models["ContactDoctor/Bio-Medical-Llama-3-8B"]="./evals/Bio-Medical-Llama-3-8B 8192"
# models["meta-llama/Llama-3.1-8B-Instruct"]="./evals/Llama-3.1-8B-Instruct 32768"
# models["ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"]="./evals/Bio-Medical-Llama-3-2-1B-CoT-012025 32768"
# models["open-r1/OpenR1-Qwen-7B"]="./evals/OpenR1-Qwen-7B 32768"

# Loop through models and run evaluation
for model in "${!models[@]}"; do
    echo "==================================================="
    echo "Evaluating model: $model"
    echo "==================================================="
    
    read -r output_dir max_tokens <<< "${models[$model]}"
    
    if [ -n "$max_tokens" ]; then
        cmd="bash scripts/eval.sh $model $EVAL_TYPE $TASKS $output_dir $max_tokens"
    else
        cmd="bash scripts/eval.sh $model $EVAL_TYPE $TASKS $output_dir"
    fi
    
    echo "Running: $cmd"
    eval $cmd
    
    echo ""
    echo "Completed evaluation for $model"
    echo ""
done

echo "All evaluations completed!"