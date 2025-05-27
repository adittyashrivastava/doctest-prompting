"""
Preprocess the simplescaling/openaimath to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/doctest-prompting/verl_data/openaimath')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    train_data_source = 'simplescaling/openaimath'
    dataset = datasets.load_dataset(train_data_source)

    # all use this source for reward calculation
    data_source = 'PTP-math'

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('problem')
            solution = example.pop('solution')
            extra_info = example.pop('extra_info', None)
            data = {
                "data_source": data_source,
                "prompt": [
                    # {
                    #     'role': "system",
                    #     'content': "You are a helpful AI Assistant, designed to provide well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST be enclosed within <think> and </think> tags. The final answer MUST be enclosed within <answer> and </answer> tags."
                    # },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
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
                    "prefix_type": 0,
                    "task": "math",
                    "calid": 0,
                    "upper_limit": "",
                    "lower_limit": ""
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', data_source), with_indices=True)
    desired_columns = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    train_dataset = datasets.Dataset.from_dict({col: train_dataset[col] for col in desired_columns})
    test_dataset = datasets.Dataset.from_dict({col: test_dataset[col] for col in desired_columns})

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(test_dataset[0])

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
