set -x


if [ "$#" -lt 5 ]; then
    echo "Usage: sft.sh <model_name_or_path> <dataset> <nnodes> <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

# sbatch scripts/sft.sh Qwen/Qwen2.5-7B-Instruct sft-itl 1 8 /home/jixuanl/verl/checkpoints/sft/Qwen2.5-7B-Instruct-SFT-ITL
# sbatch scripts/sft.sh Qwen/Qwen2.5-7B-Instruct sft 1 8 /home/jixuanl/verl/checkpoints/sft/Qwen2.5-7B-Instruct-SFT
# sbatch scripts/sft.sh Qwen/Qwen2.5-7B-Instruct sft-clean 1 8 /home/jixuanl/verl/checkpoints/sft/Qwen2.5-7B-Instruct-SFT-Clean
# sbatch scripts/sft.sh Qwen/Qwen2.5-7B-Instruct sft-baseline 1 8 Qwen2.5-7B-Instruct-SFT-Baseline

# sbatch scripts/sft.sh Qwen/Qwen2.5-7B sft-clean-v2 1 8 Qwen2.5-7B-Base-SFT-Clean-V2
# sbatch scripts/sft.sh Qwen/Qwen2.5-7B sft-baseline-v2 1 8 Qwen2.5-7B-Base-SFT-Baseline

model=$1
dataset=$2
nnodes=$3
nproc_per_node=$4
save_path=$5

SAVE_PATH="/tmp/sft_checkpoints/${save_path}"
echo "Save path: ${SAVE_PATH}"

RANK=0
ADDR="127.0.0.1"
PORT=29500

shift 5

epochs=5                # was 5
warmup_ratios=0.1       # was 0.1

torchrun --standalone --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank="${RANK}" --master_addr="${ADDR}" --master_port=$PORT \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/jixuanl/verl/verl_data/$dataset/train.parquet \
    data.val_files=/home/jixuanl/verl/verl_data/$dataset/train.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=128 \
    data.max_length=16384 \
    optim.lr=1e-5 \
    optim.weight_decay=1e-4 \
    optim.warmup_steps_ratio=$warmup_ratios \
    optim.clip_grad=0.2 \
    use_remove_padding=False \
    model.partial_pretrain=$model \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=True \
    model.use_liger=True \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=verl-sft \
    trainer.experiment_name=$dataset-sft-$model \
    trainer.total_epochs=$epochs \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@


