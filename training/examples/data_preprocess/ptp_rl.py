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
    parser.add_argument('--local_dir', default='/home/doctest-prompting/verl_data/rl-clean-v2')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    train_data_source = 'PTPReasoning/PTP-RL-ITL-Final-Clean-V2'
    train_dataset = datasets.load_dataset(train_data_source)

    train_dataset = train_dataset['train']

    # all use this source for reward calculation
    data_source = 'PTPReasoning/PTP_RL'

    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('problem')
            solution = example.pop('solution')
            prefix_type = example.pop('prefix_type', 0)
            extra_info = example.pop('extra_info', None)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
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
                    "prefix_type": prefix_type,
                    "task": extra_info['task'],
                    "calid": extra_info['calid'],
                    "upper_limit": extra_info['upper_limit'],
                    "lower_limit": extra_info['lower_limit'],
                }
            }
            return data

        return process_fn

    # use train source to trigger ptp parsing
    train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True)
    print(train_dataset[0])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    print(f"Saving train dataset to {local_dir}")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
