import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--prev_exec_res_file", type=str, default=None)
parser.add_argument("--curr_exec_res_file", type=str, default=None)

args = parser.parse_args()

prev_exec_res = json.load(open(args.prev_exec_res_file, 'r'))
curr_exec_res = json.load(open(args.curr_exec_res_file, 'r'))

if "pass@5" in prev_exec_res:
    if curr_exec_res["pass@5"] > prev_exec_res["pass@5"]:
        print(False)
    else:
        print(True)
else:
    if curr_exec_res["pass@1"] > prev_exec_res["pass@1"]:
        print(False)
    else:
        print(True)

