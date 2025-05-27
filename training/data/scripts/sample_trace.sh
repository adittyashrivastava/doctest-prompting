# You will need to create a separate env for this
# You can follow instructions in https://github.com/wwcohen/doctest-prompting

service=$1
model=$2
task=$3
variant=${4:-""}

echo "task: $task"
echo "service: $service"
echo "model: $model"
echo "variant: $variant"

test_flag=""
variant_flag=""
# if task is gsm8k or math500
if [ "$task" == "gsm8k" ] || [ "$task" == "math500" ] || [ "$task" == "math-200k" ]; then
    config="conf.d/mathword.conf"
elif [[ "$task" == *"medcalc"* ]]; then
    config="conf.d/medcalc.conf"
else
    config="conf.d/bbh_adv.conf"
    test_flag="--test_set"
fi
if [ "$variant" != "" ]; then
    variant_flag="--variant $variant"
fi
template_flag=""
if [ "$task" == "math500" ]; then
    echo "using latex template"
    template_flag="--template_file ../templates/predict_output_latex.txt"
fi

python ../run_eval_parallel_json.py $task \
    --parallel 8 \
    --config $config \
    --service $service \
    --model $model \
    --delay 2 \
    $variant_flag \
    $test_flag \
    $template_flag