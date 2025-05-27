## Env
```shell
uv venv env --python 3.11 && source env/bin/activate && uv pip install --upgrade pip
uv pip install -U vllm --no-cache-dir
uv pip install setuptools && uv pip install flash-attn --no-build-isolation --no-cache-dir
uv pip install -e.['math'] --no-cache-dir
uv pip install -e.['gpu'] --no-cache-dir
GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-cache-dir -r requirements_train.txt --no-build-isolation --no-deps
```

## SFT
### Data Preparetion
To prepare data for SFT, refer to ``data/scripts/process_sft_data.py``. After preprocessing, use ``data/scripts/push_to_hub.sh`` to upload the dataset to your repository.

Next, check ``examples/data_preprocess/ptp_sft.py`` for guidance on formatting the data into the verl format.

Or you can use our processed data at [Hugginface](https://huggingface.co/datasets/PTPReasoning/PTP-SFT-ITL-Final-Clean-V2).

### Training
Plrease refer to ```experiment/sft.sh``` for Stage-1 Supervised Fine-Tuning. Ensure that all file paths are correctly adjusted to match your environment.

To fine-tune SSRMs, verify that the appropriate ``chat_template`` is specified in ``verl/trainer/config/sft_trainer.yaml``.

Example:
```shell
sbatch scripts/sft.sh Qwen/Qwen2.5-7B sft-clean-v2 1 8 Qwen2.5-7B-Base-SFT-Clean-V2
```


## RLVR
### Data Prepartion
We release our data at [Huggingface](https://huggingface.co/datasets/PTPReasoning/PTP-RL-ITL-Final-Clean-V2).

### Training
For Stage-2 RLVR, refer to ``scripts/grpo.sh`` and modify the paths as necessary.

Example:
```shell
sbatch scripts/grpo.sh PTPReasoning/Qwen2.5-7B-Base-SFT-Clean-V2 Qwen2.5-7B-Base-RL-Clean-V2 rl-clean-v2
```


## Evaluation
We use [Lighteval](https://github.com/huggingface/lighteval) for all evaluation.

Please refer to ``eval.sh`` and ``eval_all.sh`` for more details.


## Prompted Models
Before using prompted models, set the PYTHONPATH:
```shell
export PYTHONPATH="${PYTHONPATH}:../"
```

After setting up, please refer to ```training/data/scripts/sample_trace.sh```
for how to sample traces.

We provide our sampled traces in ```training/data/data```, 
and include all experimental logs in ```training/evals```


## Acknowledgement
We release all the datasets and model checkpoints at [HF Organization](https://huggingface.co/PTPReasoning).

We thank [verl](https://github.com/volcengine/verl) for the codebase!

