import json
import argparse
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
from utils import timeout, TimeoutError, REFINE_TEMPLATE, tokenize_nltk
import random

random.seed(0)

def build_refine_solution(prompt, completion, result, bm25_processed=None):
    # append completion and result to prompt
    completion_lines = completion.split("\n")
    result_lines = result.split("\n")
    # add "# " at the beginning of each line as a comment
    mod_completion_lines = ["# " + line for line in completion_lines]
    mod_result_lines = ["# " + line for line in result_lines]
    example = {}
    example["mod_completion"] = "\n".join(mod_completion_lines)
    example["mod_result"] = "\n".join(mod_result_lines)
    sol2refine = REFINE_TEMPLATE.format_map(example)
    prompt += sol2refine

    if bm25_processed is not None:
        # retrieve the good solution
        tokenized_completion = tokenize_nltk(completion)
        scores = bm25_processed["bm25"].get_scores(tokenized_completion)
        good_sol_idx = bm25_processed["sol_indices"][np.argmax(scores)]
        return prompt, good_sol_idx
    else:
        return prompt, None
    
def build_refine_solution_plain(prompt, completion, result):
    # append completion and result to prompt
    completion_lines = completion.split("\n")
    if result.startswith("Timed Out"):
        result = "TimeoutException"
    elif result.startswith("Test Case Failed:"):
        result = "AssertionError"
    else:
        result = "BaseException"
    # mod_completion_lines = ["# " + line for line in completion_lines]
    prompt = prompt.strip().rstrip('"""').strip()
    prompt += "\n" + completion + "\n"
    prompt += f"The above code has error: {result}\n"
    prompt += '"""\n'

    return prompt, None


def build_self_refine_ft():
    prompt_map = {} # key: prompt, value: this is to avoid reprocessing the same set of canonical solutions
    py_sol_flag = {} # avoid looping through all canonical solutions for the same task_id

    @timeout(600)
    def process_sample_bm25(sample, solution):
        ns = {"prompt": "", "solution": ""}
        if sample["passed"]:
            ns["prompt"] = sample["prompt"]
            ns["solution"] = sample["completion"]
            for k in sample:
                if k != "prompt":
                    ns[k] = sample[k]
            return ns
        else:
            task_id = int(sample["task_id"].split("/")[1])
            # check if the prompt is in bm25_processed
            if sample["prompt"] not in prompt_map:
                if task_id not in py_sol_flag:
                    py_sol_flag[task_id] = True
                if py_sol_flag[task_id]:
                    task_language = solution["solutions"]["language"]
                    task_solution = solution["solutions"]["solution"]
                    assert len(task_language) == len(task_solution)
                    py_sol = [] # list of python canonical solutions
                    for sol_idx, lang in enumerate(task_language):
                        if lang == 3:
                            py_sol.append((sol_idx, task_solution[sol_idx]))
                    if len(py_sol) == 0: # if no python canonical solution, skip this sample
                        py_sol_flag[task_id] = False
                        return None
                else:
                    return None
                print(f"#Python solution for {task_id}: {len(py_sol)}")
                tokenized_sol = [tokenize_nltk(sol[1]) for sol in py_sol]
                sol_indices = [sol[0] for sol in py_sol]
                bm25 = BM25Okapi(tokenized_sol)
                prompt_map[sample["prompt"]] = {}
                prompt_map[sample["prompt"]]["bm25"] = bm25
                prompt_map[sample["prompt"]]["sol_indices"] = sol_indices
            result = "\n".join(sample["result"][:args.num_test_cases])
            ns["prompt"], sol_i = build_refine_solution(sample["prompt"], sample["completion"], result, prompt_map[sample["prompt"]])
            ns["solution"] = solution["solutions"]["solution"][sol_i]
            for k in sample:
                if k != "prompt":
                    ns[k] = sample[k]
            return ns
    
    @timeout(600)
    def process_sample_random(sample, solution):
        ns = {"prompt": "", "solution": ""}
        if sample["passed"]:
            ns["prompt"] = sample["prompt"]
            ns["solution"] = sample["completion"]
            for k in sample:
                if k != "prompt":
                    ns[k] = sample[k]
            return ns
        else:
            task_id = int(sample["task_id"].split("/")[1])
            # check if the prompt is in prompt_map
            if sample["prompt"] not in prompt_map:
                if task_id not in py_sol_flag:
                    py_sol_flag[task_id] = True
                if py_sol_flag[task_id]:
                    task_language = solution["solutions"]["language"]
                    task_solution = solution["solutions"]["solution"]
                    assert len(task_language) == len(task_solution)
                    py_sol = [] # list of python canonical solutions
                    for sol_idx, lang in enumerate(task_language):
                        if lang == 3:
                            py_sol.append(task_solution[sol_idx])
                    if len(py_sol) == 0: # if no python canonical solution, skip this sample
                        py_sol_flag[task_id] = False
                        return None
                else:
                    return None
                print(f"#Python solution for {task_id}: {len(py_sol)}")
                prompt_map[sample["prompt"]] = py_sol
            result = "\n".join(sample["result"][:args.num_test_cases])
            if args.plain_template:
                ns["prompt"], _ = build_refine_solution_plain(sample["prompt"], sample["completion"], result)
            else:
                ns["prompt"], _ = build_refine_solution(sample["prompt"], sample["completion"], result)
            # randomly select a solution
            ns["solution"] = random.choice(prompt_map[sample["prompt"]])
            for k in sample:
                if k != "prompt":
                    ns[k] = sample[k]
            return ns
    
    @timeout(600)
    def process_sample_good_bad_pair(sample, prompt_good_bad):
        task_id = int(sample["task_id"].split("/")[1])
        pair_list = []
        task_good_language = prompt_good_bad["solutions"]["language"]
        task_good_solution = prompt_good_bad["solutions"]["solution"]
        task_bad_language = prompt_good_bad["incorrect_solutions"]["language"]
        task_bad_solution = prompt_good_bad["incorrect_solutions"]["solution"]
        
        assert len(task_good_language) == len(task_good_solution)
        assert len(task_bad_language) == len(task_bad_solution)
        py_good_sol = [] # list of python canonical solutions
        py_bad_sol = []
        for sol_idx, lang in enumerate(task_good_language):
            if lang == 3:
                py_good_sol.append(task_good_solution[sol_idx])
        if len(py_good_sol) == 0: # if no python canonical solution, skip this sample
            return None
        for sol_idx, lang in enumerate(task_bad_language):
            if lang == 3:
                py_bad_sol.append(task_bad_solution[sol_idx])
        if len(py_bad_sol) == 0: # if no bad solution, only consider good solution
            ns = {"prompt": "", "solution": ""}
            ns["prompt"] = sample["prompt"]
            # randomly select a solution
            ns["solution"] = random.choice(py_good_sol)
            for k in sample:
                if k not in ["prompt", "completion", "result"]:
                    ns[k] = sample[k]
            pair_list.append(ns)
        else:
            ns = {"prompt": "", "solution": ""}
            bad_sol = random.choice(py_bad_sol)
            ns["prompt"], _ = build_refine_solution(sample["prompt"], bad_sol, "")
            ns["solution"] = random.choice(py_good_sol)
            for k in sample:
                if k not in ["prompt", "completion", "result"]:
                    ns[k] = sample[k]
            pair_list.append(ns)
        return pair_list
            

    with open(args.exec_res_file, 'r') as f:
        lines = f.readlines()
    samples = [json.loads(line) for line in lines]

    with open(args.output_file, 'w') as f:
        print("Loading code contest dataset...")
        with open(args.codecontest_all_sol_file, 'r') as f2:
            lines = f2.readlines()
        all_sols = [json.loads(line) for line in tqdm(lines)]

        for sample in tqdm(samples, total=len(samples)):
            # check whether the prompt equals
            task_id = int(sample["task_id"].split("/")[1])
            assert "\"\"\"\n" + all_sols[task_id]["prompt"] + "\n\"\"\"\n\n\n" == sample["prompt"], f"{task_id=} does not match: {all_sols[task_id]['prompt']=}, {sample['prompt']=}"
            try:
                if args.canonical_type == "random":
                    ns = process_sample_random(sample, all_sols[task_id])
                    if ns is not None:
                        f.write(json.dumps(ns) + "\n")
                elif args.canonical_type == "bm25":
                    ns = process_sample_bm25(sample, all_sols[task_id])
                    if ns is not None:
                        f.write(json.dumps(ns) + "\n")
                elif args.canonical_type == "random_good_bad_pair":
                    pair_list = process_sample_good_bad_pair(sample, prompt_good_bad=all_sols[task_id])
                    if pair_list is not None:
                        for ns in pair_list:
                            f.write(json.dumps(ns) + "\n")
                            f.flush()
            except TimeoutError:
                print(f"Timeout for {sample['task_id']}")
                task_id = int(sample["task_id"].split("/")[1])
                py_sol_flag[task_id] = False # skip this task_id
                continue

def build_ft_inf():
    with open(args.test_case_file, 'r') as f:
        lines = f.readlines()

    raw_samples = [json.loads(line) for line in lines]

    with open(args.output_file, 'w') as f:
        for idx, s in tqdm(enumerate(raw_samples), total=len(raw_samples)):
            sample = {}
            sample["task_id"] = f"CodeContests/{idx}"
            sample["prompt"] = "\"\"\"\n" + s["prompt"] + "\n\"\"\"\n\n\n"
            
            test_cases = "\ndef check(candidate): \n"
            test_count = 0
            for i, o in zip(s["public_tests"]["input"], s["public_tests"]["output"]):
                test_cases += f"    assert candidate({json.dumps(i)}) == {json.dumps(o)}\n"
                if o == "":
                    print(f"{s['public_tests']=}, {s['private_tests']=}")
                test_count += 1
            sample["test"] = test_cases
            sample["meta_data"] = {}
            sample["meta_data"]["task_name"] = s["name"]
            sample["meta_data"]["difficulty"] = s["difficulty"]
            sample["meta_data"]["time_limit"] = s["time_limit"]
            sample["canonical_solution"] = ""
            sample["entry_point"] = "solution"
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=["ft_inf", "self_refine_ft"], help="ft_inf: fine-tune inference; self_refine_ft: self-refine fine-tune")
    # parameters for ft_inf
    parser.add_argument('--test_case_file', type=str)
    # parameters for self_refine_ft
    parser.add_argument('--canonical_type', type=str, choices=["bm25", "random", "random_good_bad_pair"], default="bm25")
    parser.add_argument('--num_test_cases', type=int, default=1, help="number of test cases to be shown in the prompt")
    parser.add_argument('--codecontest_all_sol_file', type=str)
    parser.add_argument('--exec_res_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--plain_template', action="store_true", help='use plain template instead of self-refine template')
    args = parser.parse_args()

    if args.task == "ft_inf":
        build_ft_inf()
    elif args.task == "self_refine_ft":
        build_self_refine_ft()