
### BBH
python data/process_sft_data.py --data_folder '/home/doctest-prompting/training/afs_data/doctest-prompting-data/logs2/anthropic-claude-3-sonnet-20240229-bbh' --partial_programs_dir '/home/doctest-prompting/training/afs_data/doctest-prompting/bbh/mocks/partialprograms' --save_file data/data/claude_3_bbh.json
### GSM
python data/process_sft_data.py --data_folder '/home/doctest-prompting/training/afs_data/doctest-prompting-data/logs2/anthropic-claude-3-7-sonnet-20250219-gsm' --partial_programs_dir '/home/doctest-prompting/training/afs_data/doctest-prompting/mathword/mocks/partialprograms' --save_file data/data/claude_3_gsm.json
### Math
python data/process_sft_data.py --data_folder '/home/doctest-prompting/training/afs_data/doctest-prompting-data/logs2/anthropic-claude-3-7-sonnet-20250219-math' --partial_programs_dir '/home/doctest-prompting/training/afs_data/doctest-prompting/mathword/mocks/partialprograms' --save_file data/data/claude_3_math.json
### Medcalc
python data/process_sft_data.py --data_folder '/home/doctest-prompting/training/afs_data/doctest-prompting-data/logs2/anthropic-claude-3-7-sonnet-20250219-medcalc' --partial_programs_dir '/home/doctest-prompting/training/afs_data/doctest-prompting/medcalc/mocks/partialprograms' --save_file data/data/claude_3_medcalc.json
# Baseline BBH
python data/process_sft_data_baseline.py --data_folder '/home/doctest-prompting/training/afs_data/doctest-prompting-data/logs2/anthropic-claude-3-sonnet-20240229-bbh' --partial_programs_dir '/home/doctest-prompting/training/afs_data/doctest-prompting/bbh/mocks/partialprograms' --save_file data/baseline_data/claude_3_bbh.json