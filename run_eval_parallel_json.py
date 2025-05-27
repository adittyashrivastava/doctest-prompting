import json
import re
import time
import concurrent.futures
import threading
import tqdm
import queue
import os
import hashlib
import datetime
import traceback
import sys
from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from contextlib import redirect_stdout

# we save on json file now

import arg_util
import llm_util
import local_model_util
import run_eval

log_queue = queue.Queue()
counters_lock = threading.Lock()

class ResultManager:
    def __init__(self, args):
        log_filename = arg_util.log_file(args)
        # create json filename by replacing .log or .txt with .json
        if log_filename.endswith('.log'):
            self.output_file = log_filename[:-4] + '.json'
        elif log_filename.endswith('.txt'):
            self.output_file = log_filename[:-4] + '.json'
        else:
            self.output_file = log_filename + '.json'
        
        self.memory_lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.resuming = False
        
        print(f"Results will be saved to: {os.path.abspath(self.output_file)}")
        
        if os.path.exists(self.output_file):
            try:
                # use file lock even when loading to prevent race conditions
                with self.file_lock:
                    with open(self.output_file, 'r') as f:
                        self.results = json.load(f)
                self.resuming = True
                print(f"Resuming from existing results in {self.output_file}")
                print(f"Found {len(self.results['results'])} completed examples")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                self._initialize_results(args)
        else:
            self._initialize_results(args)
            # do an initial save to make sure we can write to the file
            initial_save_success = self._save()
            if initial_save_success:
                print(f"Initialized new results file: {self.output_file}")
            else:
                print("WARNING: Could not create results file! Results may not be saved.")
    
    def _initialize_results(self, args):
        self.results = {
            "task": args.task,
            "metadata": {
                "model": args.model,
                "service": args.service,
                "start_time": datetime.datetime.now().isoformat(),
                "last_updated": datetime.datetime.now().isoformat(),
                "command_args": vars(args)
            },
            "results": {},
            "summary": {
                "total": 0,
                "correct": 0,
                "parse_failures": 0,
                "accuracy": 0.0,
                "accuracy_without_parse_failures": 0.0
            }
        }
    
    def is_completed(self, example_id):
        with self.memory_lock:
            return example_id in self.results["results"]
    
    def add_result(self, example_id, input_str, output_str, target_str, prediction, is_correct, parse_failed):
        # first, update the in-memory representation with memory lock
        with self.memory_lock:
            # don't add if already exists (double-check in case of race)
            if example_id in self.results["results"]:
                return self.results["summary"]
                
            self.results["results"][example_id] = {
                "input": input_str,
                "output": output_str,
                "target": target_str,
                "prediction": prediction,
                "is_correct": is_correct,
                "parse_failed": parse_failed,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            summary = self.results["summary"]
            summary["total"] = len(self.results["results"])
            summary["correct"] = sum(1 for r in self.results["results"].values() if r["is_correct"])
            summary["parse_failures"] = sum(1 for r in self.results["results"].values() if r["parse_failed"])
            
            if summary["total"] > 0:
                summary["accuracy"] = summary["correct"] / summary["total"]
                
                valid_examples = summary["total"] - summary["parse_failures"]
                if valid_examples > 0:
                    summary["accuracy_without_parse_failures"] = summary["correct"] / valid_examples
                else:
                    summary["accuracy_without_parse_failures"] = 0.0
            
            self.results["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
            
            summary_copy = {
                "total": summary["total"],
                "correct": summary["correct"],
                "parse_failures": summary["parse_failures"],
                "accuracy": summary["accuracy"],
                "accuracy_without_parse_failures": summary["accuracy_without_parse_failures"]
            }
        
        save_success = self._save()
        if not save_success:
            print(f"WARNING: Failed to save results after processing example {example_id[:8]}")
        
        return summary_copy
    
    def _save(self):
        """Save results to JSON file with file-level locking"""
        try:
            # use file lock for the entire save operation
            with self.file_lock:
                # first, refresh from disk if it exists to merge with other potential writers
                if os.path.exists(self.output_file):
                    try:
                        with open(self.output_file, 'r') as f:
                            disk_results = json.load(f)
                            
                        with self.memory_lock:
                            # merge results dictionaries
                            for ex_id, ex_data in disk_results["results"].items():
                                if ex_id not in self.results["results"]:
                                    self.results["results"][ex_id] = ex_data
                            
                            # recalculate summary
                            summary = self.results["summary"]
                            summary["total"] = len(self.results["results"])
                            summary["correct"] = sum(1 for r in self.results["results"].values() if r["is_correct"])
                            summary["parse_failures"] = sum(1 for r in self.results["results"].values() if r["parse_failed"])
                            
                            if summary["total"] > 0:
                                summary["accuracy"] = summary["correct"] / summary["total"]
                                valid_examples = summary["total"] - summary["parse_failures"]
                                if valid_examples > 0:
                                    summary["accuracy_without_parse_failures"] = summary["correct"] / valid_examples
                                else:
                                    summary["accuracy_without_parse_failures"] = 0.0
                    except Exception as e:
                        print(f"Warning: Could not refresh from disk: {e}")
                
                dir_path = os.path.dirname(os.path.abspath(self.output_file))
                
                if dir_path: 
                    os.makedirs(dir_path, exist_ok=True)
                
                # create a temp file to avoid corruption during write
                temp_file = f"{self.output_file}.{os.getpid()}.tmp"
                
                with self.memory_lock:
                    results_to_save = json.dumps(self.results, indent=2)
                
                with open(temp_file, 'w') as f:
                    f.write(results_to_save)
                    f.flush()
                    os.fsync(f.fileno())  
                
                os.replace(temp_file, self.output_file)
                return True
        except Exception as e:
            print(f"Error saving results: {e}")
            traceback.print_exc()
            return False
    
    def get_summary(self):
        with self.memory_lock:
            return {
                "total": self.results["summary"]["total"],
                "correct": self.results["summary"]["correct"],
                "parse_failures": self.results["summary"]["parse_failures"],
                "accuracy": self.results["summary"]["accuracy"],
                "accuracy_without_parse_failures": self.results["summary"]["accuracy_without_parse_failures"]
            }
    
    def is_resuming(self):
        return self.resuming

# for multithread logging
def echo_to_queue(x):
    print(x)
    log_queue.put(x)

def logging_worker(fp):
    while True:
        try:
            message = log_queue.get(timeout=0.5)
            if message == "STOP":
                break
            with redirect_stdout(fp):
                print(message)
            log_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in logging worker: {e}")

# generate an unique has ID for an example
def get_example_id(example):
    x = example.get('input', '')
    y = example.get('target', '')
    content = f"{x}||{y}"
    return hashlib.md5(content.encode()).hexdigest()

def get_answer(x, y, template, args, result_manager, example_id):
    # skip if already completed and we're resuming
    if result_manager.is_resuming() and result_manager.is_completed(example_id):
        with counters_lock:
            main.total += 1
            main.skipped += 1
            return None, None  
    
    try:
        prompt = template.replace('{input_str}', x)
        output_messages = []
        output_messages.append(f'prompting {args.service}:{args.model}')
        output_messages.append('-' * 30 + ' input ' + '-' * 30)
        output_messages.append(x)
        output = llm_util.llm(prompt, service=args.service, model=args.model)
        output_messages.append('-' * 30 + ' output ' + '-' * 30)
        output_messages.append(output)
        prediction, is_correct, parse_failed = run_eval.check_answer(args, output, y)
        
        result_manager.add_result(example_id, x, output, y, prediction, is_correct, parse_failed)
        
        with counters_lock:
            main.total += 1
            if is_correct:
                main.correct += 1
            if parse_failed:
                main.parse_failures += 1
            total = main.total
            correct = main.correct
            parse_failures = main.parse_failures
            skipped = main.skipped
        
        output_messages.append('-' * 30 + f' {correct=} {total=} {parse_failures=} {skipped=} {prediction=} {y=} {is_correct=} ' + '-' * 30)
        
        if args.delay > 0:
            time.sleep(args.delay)

        for message in output_messages:
            echo_to_queue(message)

        return is_correct, parse_failed
    except Exception as e:
        echo_to_queue(f"Error processing example: {e}")
        traceback.print_exc()
        return False, True  

def main():
    main.total = 0
    main.correct = 0
    main.parse_failures = 0
    main.skipped = 0

    parser = arg_util.baseparser()
    args = parser.parse_args()
    arg_util.apply_shortcuts(args)

    log_filename = arg_util.log_file(args) 
    print(f'Log file: {os.path.abspath(log_filename)}')
    
    result_manager = ResultManager(args)
    log_filemode = 'a' if result_manager.is_resuming() else 'w'
    
    with open(log_filename, log_filemode) as log_fp:
        logging_thread = threading.Thread(target=logging_worker, args=(log_fp,), daemon=True)
        logging_thread.start()

        try:
            echo_to_queue(str(args))
            if args.parallel > 1:
                echo_to_queue(f'Using Parallel {args.parallel}')
            if result_manager.is_resuming():
                echo_to_queue('Resuming from previous run')

            with open(args.template_file) as fp:
                template = fp.read()
                if args.baseline_template_format:
                    canary_sep = '\n-----\n'
                    template = template[template.find(canary_sep)+len(canary_sep):]
                    template += '\n\nQ: {input_str}'

            echo_to_queue(f'{"=" * 30} prompt template {"=" * 30}')
            echo_to_queue(template)

            partial_program_file = arg_util.partial_program_file(args)
            with open(partial_program_file) as fp:
                partial_program = fp.read()
            
            # Do NOT use format here, since any empty sets written out in the program traces
            # will confuse the format code
            template = template.replace('{task_name}', args.task)
            template = template.replace('{partial_program}', partial_program)

            echo_to_queue(f'{"=" * 30} template with program {"=" * 30}')
            template_lines = template.split('\n')
            if len(template_lines) < 100:
                echo_to_queue(template)
            else:
                for line in template_lines[0:50]:
                    echo_to_queue(line.strip())
                echo_to_queue('.' * 50)
                echo_to_queue(f'{len(template_lines) - 100} lines skipped')
                echo_to_queue('.' * 50)
                for line in template_lines[-50:]:
                    echo_to_queue(line.strip())

            example_file = arg_util.example_file(args)
            with open(example_file) as fp:
                examples = json.loads(fp.read())['examples']

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = []
                active_examples = list(arg_util.active_subset(args, examples))
                
                echo_to_queue(f"Processing {len(active_examples)} examples ({len(examples)} total in dataset)")
                
                for ex in active_examples:
                    x = ex['input']
                    y = ex['target']
                    example_id = get_example_id(ex)
                    
                    future = executor.submit(
                        get_answer,
                        x,
                        y,
                        template,
                        args,
                        result_manager,
                        example_id
                    )
                    futures.append(future)
                
                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures),
                    desc="Processing examples"
                ):
                    future.result()

            echo_to_queue(f"\nResults saved to: {os.path.abspath(result_manager.output_file)}")
            
            summary = result_manager.get_summary()
            
            echo_to_queue("\n" + "="*50)
            echo_to_queue("EVALUATION COMPLETE")
            echo_to_queue(f"Total examples processed: {main.total}")
            echo_to_queue(f"Skipped (already completed): {main.skipped}")
            echo_to_queue(f"Newly evaluated: {main.total - main.skipped}")
            echo_to_queue(f"Correct: {main.correct}")
            echo_to_queue(f"Parse failures: {main.parse_failures}")
            
            if main.total > main.skipped:
                acc = main.correct / (main.total - main.skipped) if (main.total - main.skipped) > 0 else 0
                echo_to_queue(f"Accuracy (for this run): {acc:.4f}")
            
            echo_to_queue(f"Total accuracy (all runs): {summary['accuracy']:.4f}")
            
            if summary['parse_failures'] > 0:
                echo_to_queue(f"Accuracy (ignoring parse failures): {summary['accuracy_without_parse_failures']:.4f}")
            
            echo_to_queue("="*50)

        except KeyboardInterrupt:
            echo_to_queue("\nSaving current progress and exiting...")
        except Exception as e:
            echo_to_queue(f"Error: {e}")
            traceback.print_exc()
        finally:
            print(f"\nResults saved to: {os.path.abspath(result_manager.output_file)}")
            
            try:
                log_queue.put("STOP", timeout=2)
                logging_thread.join(timeout=5)
            except Exception as e:
                print("Warning: Could not cleanly stop logging thread")


if __name__ == "__main__":
    main()