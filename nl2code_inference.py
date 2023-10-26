#!/usr/bin/env python
# coding=utf-8

import argparse
import logging

import torch
import json
import os

from transformers import (
    default_data_collator,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)
from accelerate.utils import set_seed

from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm


import json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def build_datasets(tokenizer):
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token
    raw_dataset = load_dataset("json", data_files=args.prompt_file)
    raw_dataset = raw_dataset["train"]
    column_names = raw_dataset.column_names

    def prepare_features(examples):
        if "starcoder" in args.model_name_or_path:
            # starcoder could not handle prompt ending with \n
            for idx in range(len(examples["prompt"])):
                examples["prompt"][idx] = examples["prompt"][idx].strip()
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length - args.gen_length
        )

        return tokenized_inputs
    
    tokenized_datasets = raw_dataset.map(
            prepare_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    
    return raw_dataset, tokenized_datasets

def model_inference(raw_dataset, tokenized_datasets, tokenizer):
    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    logger.info(f'{model.dtype=}')

    logger.info(args)
    data_sampler = SequentialSampler(tokenized_datasets)
    dataloader = DataLoader(
        tokenized_datasets, collate_fn=default_data_collator, batch_size=args.batch_size, sampler=data_sampler
    )

    model = accelerator.prepare_model(model)
    
    dataloader = accelerator.prepare_data_loader(dataloader)
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    f_pred = open(f"{args.output_dir}/prediction_tmp{args.temperature}_seq{args.num_return_sequences}.jsonl", "w", encoding="utf-8")  # open file for writing result
    
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token

    n_tasks = len(raw_dataset)
    raw_samples = [raw_dataset[task] for task in range(n_tasks)]
    
    all_preds = []
    all_inputs = []
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            model.eval()
            output_sequences = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=args.max_seq_length,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                do_sample=args.do_sample,
                num_return_sequences=args.num_return_sequences,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=args.repetition_penalty,
            )
            # repeat(args.num_return_sequences, 1) will be wrong, as it makes e.g., [[2], [5], [8] [7]] to be [[2], [5], [8], [7], [2], [5], [8], [7]] 
            # use repeat_interleave instead
            batch_input = batch["input_ids"].repeat_interleave(args.num_return_sequences, dim=0)
            batch_pred = accelerator.pad_across_processes(output_sequences, dim=1, pad_index=tokenizer.pad_token_id)
            batch_input, batch_pred = accelerator.gather((batch_input, batch_pred))
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                input_length = batch_input.shape[1]
                batch_pred = batch_pred[:, input_length:]
                batch_input = batch_input.cpu().numpy()
                batch_pred = batch_pred.cpu().numpy()
                for i, p in zip(batch_input, batch_pred):
                    all_preds.append(p)
                    all_inputs.append(i)

    logger.info("Done generating")
    assert len(all_inputs) == len(all_preds), f"{len(all_inputs)=} {len(all_preds)=}"
    all_inputs = all_inputs[:n_tasks * args.num_return_sequences]
    all_preds = all_preds[:n_tasks * args.num_return_sequences]
    # num of samples = num of prompts * num of generated sequences
    # each prompt generates only one json object, and use 'samples' key to store a list of all generated sequences
    # split the list of all generated sequences into a list of lists, each sublist contains all generated sequences for one prompt
    all_inputs_split = [all_inputs[i:i + args.num_return_sequences] for i in range(0, len(all_inputs), args.num_return_sequences)]
    all_preds_split = [all_preds[i:i + args.num_return_sequences] for i in range(0, len(all_preds), args.num_return_sequences)]

    logger.info("Decoding and writing to file")
    for idx, (p, i) in tqdm(enumerate(zip(all_preds_split, all_inputs_split)), total=len(all_inputs_split)):
        r = raw_samples[idx]
        tmp = {"input": "", "samples": []}
        for k, v in r.items():
            tmp[k] = v
        tmp["input"] = tokenizer.decode(i[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for pred, inp in zip(p, i):
            assert tokenizer.decode(inp, skip_special_tokens=True, clean_up_tokenization_spaces=True) == tmp["input"], f"{tokenizer.decode(inp, skip_special_tokens=True, clean_up_tokenization_spaces=True)=} {tmp['input']=}"
            if "HumanEval" in args.prompt_file or "mbpp"  in args.prompt_file:
                # func completion benchmarks needs truncation
                predict_code = tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                func_code = ""
                start = False
                for line in predict_code.split("\n"):
                    if line.strip() != "":
                        if line.startswith("    ") or line.startswith("\t"):
                            # first function line
                            start = True
                        if not line.startswith("    ") and not line.startswith("\t"):
                            if start:
                                break
                    func_code += line + "\n"
                tmp["samples"].append(func_code)
            else:
                tmp["samples"].append(tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        f_pred.write(json.dumps(tmp) + "\n")
        f_pred.flush()

    f_pred.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model inference args
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Pre-trained Model Path")
    parser.add_argument("--prompt_file", type=str,required=True, help="jsonl file with prompts")
    parser.add_argument("--gen_length", type=int, default=256, help="max length of generated token sequence")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="max length of prompt")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for code completion")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature of 1.0 has no effect, lower tend toward greedy sampling",)
    parser.add_argument("--output_dir", type=str, default="output_dir", help="output directory to save predictions")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.",)
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--num_beams", type=int, default=None, help="num of beam for beam-search")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="primarily useful for CTRL model; in that case, use 1.2")
    args = parser.parse_args()
    if args.num_beams is not None:
        args.do_sample = False
    else:
        args.do_sample = True
        args.num_beams = 1
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token
    raw_dataset, tokenized_dataset = build_datasets(tokenizer)
    model_inference(raw_dataset, tokenized_dataset, tokenizer)