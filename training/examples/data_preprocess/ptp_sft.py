import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/doctest-prompting/verl_data/sft-baseline-v2')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'PTPReasoning/PTP-SFT-ITL-Final-Baseline-V2'
    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset['train']

    def make_map_fn(split):

        def process_fn(example, idx):
            prompt = example.pop('instruction')
            question = prompt[-1]['content']
            solution = example.pop('response')[-1]['content']
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': solution,
                    "question": question,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    # print a few samples
    print(train_dataset[0])
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
