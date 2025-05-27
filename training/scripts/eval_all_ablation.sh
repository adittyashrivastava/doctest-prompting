#!/bin/bash

TASKS="pubmedqa_dedup"
EVAL_TYPE="custom"

model="PTPReasoning/Qwen2.5-7B-Base-RL-Clean-V2"
output_dir_base="Qwen2.5-7B-Base-RL-Clean-V2"

cmd="bash scripts/eval.sh $model $EVAL_TYPE $TASKS ./evals_ablations/eval_sampling/$output_dir_base 32768 0 1 1.0 0 0 0 0"
echo "Running: $cmd"
eval $cmd

cmd="bash scripts/eval.sh $model $EVAL_TYPE $TASKS ./evals_ablations/eval_greedy/$output_dir_base 32768 0 1 0.0 0 0 0 0"
echo "Running: $cmd"
eval $cmd

for rounds in 3 5 7 9 15 30 60; do
    echo "==================================================="
    echo "Evaluating model: $model with $rounds rounds"
    echo "==================================================="

    cmd="bash scripts/eval.sh $model $EVAL_TYPE $TASKS ./evals_ablations/${rounds}_rounds/evals_audit_consistency/$output_dir_base 32768 0 1 0 0 0 0 $rounds"
    echo "Running: $cmd"
    eval $cmd

    cmd="bash scripts/eval.sh $model $EVAL_TYPE $TASKS ./evals_ablations/${rounds}_rounds/evals_self_consistency/$output_dir_base 32768 0 1 0 0 0 0 0 $rounds"
    echo "Running: $cmd"
    eval $cmd

    echo ""
    echo "Completed evaluation for $model with $rounds rounds"
    echo ""
done