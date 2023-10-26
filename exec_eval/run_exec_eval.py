import numpy as np
from collections import defaultdict
import logging
from typing import List, Union
import itertools
import argparse
import json
import os

from exec_eval.execution import evaluate_with_test_code, evaluate_with_test_code_no_extraction

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


STOP_TOKEN = ['\nclass', '\ndef', '\n#', '\nif', '\nprint', '\n\'\'\'']

def pass_at_K(results, k = [1, 10, 100]):
    def _turn_list_into_dict(result_lines):
        result_dict = defaultdict(list)
        for line in result_lines:
            task_id = "/".join(line['task_id'].split("/")[:2])
            result_dict[task_id].append(line['passed'])
        return result_dict

    # Calculate pass@k.
    total, correct = [], []
    for passed in _turn_list_into_dict(results).values():
        total.append(len(passed))
        correct.append(sum(passed))

    if len(k) == 1:
        k = [total[0]]
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": round(_estimate_pass_at_k(total, correct, k).mean(), 4)
                 for k in ks if (total >= k).all()}
    logger.info(pass_at_k)
    return pass_at_k

def _estimator(n: int, c: int, k: int) -> float:
    """
    Calculates comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 0
    return np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def _estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([1.0 - _estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def load_jsonl(file_path):
    json_objects = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            json_objects.append(json.loads(line.strip()))
    return json_objects

def load_tasks(task_path):
    result = dict()
    lines = load_jsonl(task_path)
    for line in lines:
        result[line['task_id']] = line
    return result

def solution_extract(content):
    for identifier in STOP_TOKEN:
        if identifier in content:
            content = content.split(identifier)[0]
    return content

def map_task_id_for_solution(source_path, predict_path, no_extraction=False):
    database = dict()
    if source_path is None:
        raw_problems = load_tasks(predict_path)
    else:
        raw_problems = load_tasks(source_path)
    for task_id in raw_problems.keys():
        database[raw_problems[task_id]['prompt']] = raw_problems[task_id]

    result = []
    predictions = load_jsonl(predict_path)
    for pre in predictions:
        task = database[pre['prompt']]
        if not pre['samples']:
            result.append({
                'task_id': task['task_id'],
                'prompt': pre['prompt'],
                'test': task['test'],
                'entry_point': task['entry_point'],
                'completion': 'empty solution here, execution will fail'
            })
        for sample in pre['samples']:
            if no_extraction:
                processed_code = sample
            else:
                processed_code = solution_extract(sample)
            result.append({
                'task_id': task['task_id'],
                'prompt': pre['prompt'],
                'test': task['test'],
                'entry_point': task['entry_point'],
                'completion': processed_code
            })
    return result, len(raw_problems)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, default=None, help="model input file in .jsonl format")
    parser.add_argument("--model_pred_file", type=str, help="model output file in .jsonl format")
    parser.add_argument("--res_file", type=str, help="execution result file in .jsonl format")
    parser.add_argument("--no_extraction", action="store_true", help="whether to extract the prediction or keep the raw output")
    parser.add_argument("--timeout", type=float, default=3, help="how many seconds to wait during execution for each test case")

    args = parser.parse_args()
    
    handled_solutions, task_count = map_task_id_for_solution(args.prompt_file, args.model_pred_file, args.no_extraction)
    assert len(handled_solutions) % task_count == 0, f"handled_solutions: {handled_solutions}, task_count: {task_count}"
    k=[len(handled_solutions) // task_count]
    
    if args.no_extraction:
        ground_truth_exec_result = evaluate_with_test_code_no_extraction(handled_solutions, timeout=args.timeout)
    else:
        if "HumanEval" in args.model_pred_file:
            data = "HumanEval"
        elif "mbpp_sanitized" in args.model_pred_file:
            data = "mbpp"
        ground_truth_exec_result = evaluate_with_test_code(handled_solutions, timeout=args.timeout, data=data)

    logger.info('Pass@K:')
    
    pak = pass_at_K(ground_truth_exec_result,k=k)

    with open(args.res_file, 'w', encoding='utf8') as f:
        for line in ground_truth_exec_result:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    output_dir = os.path.dirname(args.res_file)
    with open(os.path.join(output_dir, 'pass_at_k.json'), 'w', encoding='utf8') as f:
        json.dump(pak, f, ensure_ascii=False, indent=4)