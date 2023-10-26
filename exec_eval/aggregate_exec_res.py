import json
import argparse
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from utils import REFINE_TEMPLATE, REFINE_TEMPLATE_BASELINE
import random

random.seed(0)

def build_refine_prompt(prompt, completion, result, aggr_method, ablation=None):
    # append completion and result to prompt
    completion_lines = completion.split("\n")
    indent = ""
    for l in completion_lines:
        if l.strip() != "":
            # start compute indent
            indent = len(l) - len(l.lstrip(" \t"))
            indent = l[:indent]
            break
    # assert indent != "", f"{prompt}\n***\n***\n{completion_lines}\n***\n***\n{result}"
    result_lines = result.split("\n")

    if aggr_method == "self_refine":
        if REFINE_TEMPLATE.split("\n")[0] in prompt:
            prompt = prompt.split(REFINE_TEMPLATE.split("\n")[0])[0].strip()
        if not prompt.endswith("\n"):
            prompt += "\n"
        # add "# " at the beginning of each line as a comment
        mod_completion_lines = ["# " + line for line in completion_lines]
        mod_result_lines = ["# " + line for line in result_lines]
        example = {}
        example["mod_completion"] = "\n".join(mod_completion_lines)
        example["mod_result"] = "\n".join(mod_result_lines)
        if ablation == "no_completion":
            example["mod_completion"] = ""
        elif ablation == "no_exe":
            example["mod_result"] = ""
        sol2refine = REFINE_TEMPLATE.format_map(example)
    elif aggr_method == "baseline":
        docstring_end = ""
        if prompt.strip().endswith("\"\"\"") or prompt.strip().endswith("'''"):
            docstring_end = prompt.strip()[-3:]
            prompt = prompt.strip()[:-3]
        if REFINE_TEMPLATE_BASELINE.split("\n")[0] in prompt:
            prompt = prompt.split(REFINE_TEMPLATE_BASELINE.split("\n")[0])[0].strip()
        if not prompt.endswith("\n"):
            prompt += "\n"
        example = {}
        example["completion"] = "\n".join(completion_lines)
        example["result"] = "\n".join(result_lines)
        if ablation == "no_completion":
            example["completion"] = ""
        elif ablation == "no_exe":
            example["result"] = ""
        sol2refine = REFINE_TEMPLATE_BASELINE.format_map(example)
        sol2refine += "\n" + docstring_end  # add back the ending """ for the baseline template
    # add indent to sol2refine
    sol2refine_lines = sol2refine.split("\n")
    sol2refine_lines = [indent + line for line in sol2refine_lines]
    sol2refine = "\n".join(sol2refine_lines)
    prompt += sol2refine.rstrip(" ")

    return prompt


def build_self_refine():
    with open(args.exec_res_file, 'r') as f:
        lines = f.readlines()
    exec_samples = [json.loads(line) for line in lines]
    with open(args.pred_file, 'r') as f:
        lines = f.readlines()
    pred_samples = [json.loads(line) for line in lines]

    if len(pred_samples[0]["samples"]) > 1:
        # k-generation, create one pred_sample for each sample in ["samples"]
        new_pred_samples = []
        for pred_sample in pred_samples:
            for sample in pred_sample["samples"]:
                new_pred_sample = {}
                for k in pred_sample:
                    if k != "samples":
                        new_pred_sample[k] = pred_sample[k]
                new_pred_sample["samples"] = [sample]
                new_pred_samples.append(new_pred_sample)
        pred_samples = new_pred_samples[:]
    
    # check whether more than one prediction for each task_id
    task_ids = ["/".join(pred_sample["task_id"].split()[:2]) for pred_sample in pred_samples]
    if len(set(task_ids)) != len(task_ids):
        # algin exec_samples and pred_samples using task_id and prompt
        exec_samples = sorted(exec_samples, key=lambda x: (x["prompt"]))
        pred_samples = sorted(pred_samples, key=lambda x: (x["prompt"]))

    passed_preds = []
    self_refine_inputs = []

    task_id_incres = {}
    for exe, pred in zip(exec_samples, pred_samples):
        assert exe["task_id"].split("/")[:2] == pred["task_id"].split("/")[:2] and exe["prompt"] == pred["prompt"], f"{exe['task_id']}\n{pred['task_id']}\n{exe['prompt']}\n{pred['prompt']}"
        if exe["passed"]:
            passed_preds.append(pred)
        else:
            ns = {}
            exec_result = []
            exec_result.append(exe["result"][0])
            # sort exe["result"][1:] and add them to exec_result
            for t in sorted(exe["result"][1:]):
                exec_result.append(t)
            # if num_test_cases is 1, get the first failed test case
            if args.num_test_cases == 1:
                if "apps_test" in args.exec_res_file and "apps_test" in args.pred_file:
                    result = exec_result[0]
                else:
                    result = exec_result[0]
                    for t in exec_result[1:]:
                        if t.startswith("Test Case Failed:") and "assert" in t:
                            result = t
                            break
            else:
                result = "\n".join(exec_result[1:args.num_test_cases+1])
            ns["prompt"] = build_refine_prompt(exe["prompt"], exe["completion"], result, args.aggr_method, args.ablation)
            for k in exe:
                if k != "prompt":
                    if k == "task_id":
                        # handle the potential duplicate task_id when the one-time generation has more than one prediction
                        task_id = "/".join(exe[k].split("/")[:2])  # only keep the first two parts of the task_id
                        if exe[k] not in task_id_incres:
                            task_id_incres[task_id] = []
                        if len(task_id_incres[task_id]) > 0:
                            task_id_concrete = task_id + "/" + str(len(task_id_incres[task_id]) - 1)
                        else:
                            task_id_concrete = task_id
                        ns[k] = task_id_concrete
                        task_id_incres[task_id].append(task_id_concrete)
                    else:    
                        ns[k] = exe[k]
            self_refine_inputs.append(ns)

    with open(args.passed_pred_file, 'w') as f:
        for sample in passed_preds:
            f.write(json.dumps(sample) + "\n")
    with open(args.self_refine_input_file, 'w') as f:
        for sample in self_refine_inputs:
            f.write(json.dumps(sample) + "\n")

def merge_self_refine():
    with open(args.self_refine_preds_file, 'r') as f:
        lines = f.readlines()
    self_refine_preds = [json.loads(line) for line in lines]
    with open(args.orig_passed_pred_file, 'r') as f:
        lines = f.readlines()
    orig_passed_preds = [json.loads(line) for line in lines]

    merged_preds = self_refine_preds + orig_passed_preds

    # re-assign task_id, if there are duplicate task_id, add a suffix
    task_id_incres = {}
    for pred in merged_preds:
        task_id = "/".join(pred["task_id"].split("/")[:2])
        if task_id not in task_id_incres:
            task_id_incres[task_id] = []
        if len(task_id_incres[task_id]) > 0:
            task_id_concrete = task_id + "/" + str(len(task_id_incres[task_id]) - 1)
        else:
            task_id_concrete = task_id
        pred["task_id"] = task_id_concrete
        task_id_incres[task_id].append(task_id_concrete)

    with open(args.merged_preds_file, 'w') as f:
        for sample in merged_preds:
            f.write(json.dumps(sample) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=["build_self_refine", "merge_self_refine"])
    # args for build_self_refine
    parser.add_argument('--num_test_cases', type=int, default=1, help="number of test cases to be shown in the prompt")
    parser.add_argument('--pred_file', type=str, help="prediction files")
    parser.add_argument('--exec_res_file', type=str, help="execution result file")
    parser.add_argument('--passed_pred_file', type=str, help="passed prediction file containing only passed predictions")
    parser.add_argument('--self_refine_input_file', type=str, help="input file for self refine containing only failed samples")
    parser.add_argument('--aggr_method', type=str, choices=["self_refine", "baseline"], default="self_refine", help="use baseline aggregation method")
    parser.add_argument('--ablation', type=str, default=None, choices=["no_completion", "no_exe"], help="ablation study")
    # args for merge_self_refine
    parser.add_argument('--self_refine_preds_file', type=str, help="self refine prediction file")
    parser.add_argument('--orig_passed_pred_file', type=str, help="original passed prediction file")
    parser.add_argument('--merged_preds_file', type=str, help="merge the self refine predictions and original passed predictions")
    args = parser.parse_args()
    if args.task == "build_self_refine":
        build_self_refine()
    elif args.task == "merge_self_refine":
        merge_self_refine()