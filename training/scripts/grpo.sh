set -x

# Demo
# MODEL_PATH='/home/jixuanl/verl/checkpoints/sft/Qwen2.5-7B-Instruct-SFT-ITL/global_step_130'
# MODEL_PATH="/home/jixuanl/verl/checkpoints/sft/${model_name}"

# sbatch '/home/jixuanl/verl/scripts/grpo.sh' 'Qwen2.5-7B-Instruct-SFT-BBH-ITL/global_step_65' 'Qwen2.5-7B-Instruct-RL-BBH-ITL-Medicalc' rl-medicalc
# sbatch '/home/jixuanl/verl/scripts/grpo.sh' 'Qwen2.5-7B-Instruct-SFT-BBH-ITL/global_step_65' 'Qwen2.5-7B-Instruct-RL-BBH-ITL' rl-bbh-itl
# sbatch '/home/jixuanl/verl/scripts/grpo.sh' 'Qwen2.5-7B-Instruct-SFT/global_step_190' 'Qwen2.5-7B-Instruct-RL' rl
# sbatch '/home/jixuanl/verl/scripts/grpo.sh' 'PTPReasoning/Qwen2.5-7B-Instruct-SFT-Clean' 'Qwen2.5-7B-Instruct-RL-Clean' rl-clean

# sbatch '/home/jixuanl/verl/scripts/grpo.sh' 'PTPReasoning/Qwen2.5-7B-Base-SFT-Clean-V2' 'Qwen2.5-7B-Base-RL-Clean-V2' rl-clean-v2

# sbatch '/home/jixuanl/verl/scripts/grpo.sh' 'PTPReasoning/Qwen2.5-7B-Base-SFT-Baseline-V2' 'Qwen2.5-7B-Base-RL-Baseline-V2' rl-clean-baseline-v2

# sbatch '/home/jixuanl/verl/scripts/grpo.sh' 'PTPReasoning/Qwen2.5-7B-Base-SFT-Baseline' 'Qwen2.5-7B-Base-RL-Baseline-V2' rl-clean-baseline-v2


project_name='PTP-rl'
model_name=${1:-'Qwen2.5-7B-Instruct-SFT-BBH-ITL/global_step_65'}
exp_name=${2:-'Qwen2.5-7B-Instruct-RL-BBH-ITL'}
dataset_name=${3:-'rl-bbh-itl'}

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=("$nodes")

head_node_ip=$(hostname --ip-address)
echo "Head node: $head_node_ip"

if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"


RAY_DATA_HOME=/home/doctest-prompting/training/
MODEL_PATH="${model_name}"
CKPTS_DIR=${CKPTS_DIR:-"/tmp/rl_checkpoints_final/${exp_name}"}

train_files="['${RAY_DATA_HOME}/verl_data/${dataset_name}/train.parquet']"
test_files="['${RAY_DATA_HOME}/verl_data/openaimath/test.parquet','${RAY_DATA_HOME}/verl_data/medcalc/test.parquet']"

adv_estimator=grpo
clip_ratio=0.2
clip_ratio_high=0.28
clip_ratio_low=0.2
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
loss_agg_mode="token-mean"

total_epochs=10
entropy_coeff=0.0
grad_clip=1.0
norm_adv_by_std_in_grpo=True


max_prompt_length=2048
max_response_length=4096            # was 4096
n_resp_per_prompt=8
train_prompt_mini_bsz=256
use_dynamic_bsz=True
learning_rate=1e-6                  # was 3e-6
warmup_ratio=0.01                   # was 0.01
ppo_max_token_len_per_gpu=24000
optim_weight_decay=0.1              # was 0.01

temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
max_num_seqs=2048
max_num_batched_tokens=16384        # was 8192

DATA_PARAMS="data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=256 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error'"

MODEL_PARAMS="actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True"

OPTIMIZER_PARAMS="actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${warmup_ratio} \
    actor_rollout_ref.actor.optim.weight_decay=${optim_weight_decay} \
    actor_rollout_ref.actor.optim.weight_decay=0.1"

PPO_PARAMS="actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_c=10.0"

FSDP_PARAMS="actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False"

ROLLOUT_PARAMS="actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}"

TRAINER_PARAMS="trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.total_epochs=${total_epochs} \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=auto"

ray stop

ray start --head --node-ip-address="$head_node_ip" --port=$port --num-gpus 8 --block &

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    ${DATA_PARAMS} \
    ${MODEL_PARAMS} \
    ${OPTIMIZER_PARAMS} \
    ${PPO_PARAMS} \
    ${FSDP_PARAMS} \
    ${ROLLOUT_PARAMS} \
    ${TRAINER_PARAMS}
