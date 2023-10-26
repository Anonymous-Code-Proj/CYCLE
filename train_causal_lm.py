import json
import logging
import math
import os
import random
from itertools import chain

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import datetime

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from transformers.utils.versions import require_version
import sys


# local imports
from utils import parse_args, REFINE_TEMPLATE

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


def model_init():
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    return config, tokenizer, model

def data_init(tokenizer):
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            if "valid" in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split="valid",
                )
            else:
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
    train_dataset = None
    eval_dataset = None
    # Preprocessing the datasets
    if args.dataset_name == "deepmind/code_contests":
        # code contest specific preprocessing
        for split in ["train", "validation"]:
            # We need to remove problems with long descriptions and without python3 solutions.
            args.desc_length = 512
            py3_idx = []
            logger.info("Filtering out problems without python3 solutions...")
            for idx, sol in tqdm(enumerate(raw_datasets[split]["solutions"]), total=len(raw_datasets[split]["solutions"])):
                if 3 in sol["language"]: # 3 is python3
                    py3_idx.append(idx)
            raw_datasets[split] = raw_datasets[split].select(py3_idx)
            desc_idx = []
            logger.info("Filtering out problems with long descriptions...")
            for idx, desc in tqdm(enumerate(raw_datasets[split]["description"]), total=len(raw_datasets[split]["description"])):
                if len(desc.split()) <= args.desc_length:
                    desc_idx.append(idx)
            raw_datasets[split] = raw_datasets[split].select(desc_idx)
        prompt_key = "description"
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names

        def group_texts(examples):
            # prepare problme description
            desc = examples[prompt_key]
            for i in range(len(desc)):
                desc[i] = '"""\n' + desc[i]
            tokenized_desc = tokenizer(desc)
            suffix = tokenizer('\n"""\n\n\n')
            concat_desc = {}
            for k in tokenized_desc.keys():
                concat_desc[k] = []
                for i in range(len(tokenized_desc[k])):
                    concat_desc[k].append(tokenized_desc[k][i] + suffix[k])

            # prepare solution
            solutions = examples["solutions"]
            features = {"input_ids": [], "attention_mask": []}
            # sample up to threshold solutions per problem
            threshold = 100
            for i in range(len(solutions)): # for each problem
                for j in range(min(threshold, len(solutions[i]["solution"]))): # for each solution
                    if solutions[i]["language"][j] != 3: # 3 is python3
                        continue
                    code = solutions[i]["solution"][j]
                    tokenized_code = tokenizer(code) 
                    # build samples: problem description + code
                    features["input_ids"].append(concat_desc["input_ids"][i] + tokenized_code["input_ids"])
                    features["attention_mask"].append(concat_desc["attention_mask"][i] + tokenized_code["attention_mask"])
                    assert len(features["input_ids"][-1]) == len(features["attention_mask"][-1])

            # Our implementation -- sample-wise chunking
            result = {"input_ids": [], "attention_mask": [], "labels": []}
            for i in range(len(features["input_ids"])):
                for j in range(0, len(features["input_ids"][i]), args.block_size):
                    result["input_ids"].append(features["input_ids"][i][j:j+args.block_size])
                    result["attention_mask"].append(features["attention_mask"][i][j:j+args.block_size])
                    result["labels"].append(features["input_ids"][i][j:j+args.block_size])
            assert len(result["input_ids"]) == len(result["attention_mask"])
            # pad to block_size
            for i in range(len(result["input_ids"])):
                pad_len = args.block_size - len(result["input_ids"][i])
                result["input_ids"][i] += [tokenizer.pad_token_id] * pad_len
                result["attention_mask"][i] += [0] * pad_len
                result["labels"][i] += [-100] * pad_len
            return result

        def prepare_codecontest(examples):
            # prepare problme description
            # max sequence length for problem description
            desc = examples[prompt_key]
            max_desc_seq_length = 1024
            for i in range(len(desc)):
                desc[i] = '"""\n' + desc[i]
            tokenized_desc = tokenizer(
                desc, 
                truncation=True,
                max_length=max_desc_seq_length)
            suffix = tokenizer('\n"""\n\n\n')
            suffix_len = len(suffix['input_ids'])
            concat_desc = {}
            for k in tokenized_desc.keys():
                concat_desc[k] = []
                for i in range(len(tokenized_desc[k])):
                    concat_desc[k].append(tokenized_desc[k][i] + suffix[k])
            
            # prepare code
            # max sequence length for code
            solutions = examples["solutions"]
            max_code_seq_length = args.block_size - max_desc_seq_length - suffix_len
            features = {"input_ids": [], "attention_mask": [], "labels": []}
            # sample up to threshold solutions per problem
            threshold = 100
            for i in range(len(solutions)): # for each problem
                for j in range(min(threshold, len(solutions[i]["solution"]))): # for each solution
                    if solutions[i]["language"][j] != 3: # 3 is python3
                        continue
                    code = solutions[i]["solution"][j]
                    tokenized_code = tokenizer(
                        code, 
                        truncation=True,
                        max_length=max_code_seq_length)
                    # add eos token if tokenized code is shorter than max_code_seq_length
                    if len(tokenized_code['input_ids']) < max_code_seq_length:
                        tokenized_code['input_ids'].append(tokenizer.eos_token_id)
                        tokenized_code['attention_mask'].append(1)
                    # build samples: problem description + code
                    features["input_ids"].append(concat_desc["input_ids"][i] + tokenized_code["input_ids"])
                    features["attention_mask"].append(concat_desc["attention_mask"][i] + tokenized_code["attention_mask"])
                    # only predict code tokens
                    features["labels"].append([-100] * len(concat_desc["input_ids"][i]) + tokenized_code["input_ids"])
                    assert len(features["input_ids"][-1]) == len(features["attention_mask"][-1]) == len(features["labels"][-1])
                    assert len(features["input_ids"][-1]) <= args.block_size and len(features["attention_mask"][-1]) <= args.block_size and len(features["labels"][-1]) <= args.block_size, f"len(features['input_ids'][-1])={len(features['input_ids'][-1])}, len(features['attention_mask'][-1])={len(features['attention_mask'][-1])}, len(features['labels'][-1])={len(features['labels'][-1])}"
            # padding to max_length
            for i in range(len(features["input_ids"])):
                features["input_ids"][i] += [tokenizer.pad_token_id] * (args.block_size - len(features["input_ids"][i]))
                features["attention_mask"][i] += [0] * (args.block_size - len(features["attention_mask"][i]))
                features["labels"][i] += [-100] * (args.block_size - len(features["labels"][i]))

            return features

        if args.predict_type == "code_only":
            with accelerator.main_process_first():
                codecontest_datasets = raw_datasets.map(
                    prepare_codecontest,
                    batched=True,
                    num_proc=1,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Preprocessing Dataset",
                )
        elif args.predict_type == "full":
            with accelerator.main_process_first():
                codecontest_datasets = raw_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=1,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )

        train_dataset = codecontest_datasets["train"]
        eval_dataset = codecontest_datasets["validation"]
    if args.codecontest_finetune:
        data_files = {}
        dataset_args = {}
        if args.codecontest_finetune_train_file is not None:
            data_files["train"] = args.codecontest_finetune_train_file
        if args.codecontest_finetune_validation_file is not None:
            data_files["validation"] = args.codecontest_finetune_validation_file
        extension = "json"
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

        # code contest specific preprocessing
        for split in ["train", "validation"]:
            # We need to remove problems with long descriptions and without python3 solutions.
            args.desc_length = 512
            desc_idx = []
            logger.info("Filtering out problems with long prompts...")
            for idx, desc in tqdm(enumerate(raw_datasets[split]["prompt"]), total=len(raw_datasets[split]["prompt"])):
                if len(desc.split()) <= args.desc_length:
                    desc_idx.append(idx)
            raw_datasets[split] = raw_datasets[split].select(desc_idx)
        prompt_key = "prompt"
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names

        def prepare_codecontest(examples):
            # prepare problme description
            # max sequence length for problem description
            desc = examples[prompt_key]
            max_desc_seq_length = 1024
            for i in range(len(desc)):
                desc[i] = '"""\n' + desc[i]
            tokenized_desc = tokenizer(
                desc, 
                truncation=True,
                max_length=max_desc_seq_length)
            suffix = tokenizer('\n"""\n\n\n') # make sure the suffix will not be truncated
            suffix_len = len(suffix['input_ids'])
            concat_desc = {}
            for k in tokenized_desc.keys():
                concat_desc[k] = []
                for i in range(len(tokenized_desc[k])):
                    concat_desc[k].append(tokenized_desc[k][i] + suffix[k])
            
            # prepare code
            # max sequence length for code
            solutions = examples["py_solutions"]
            max_code_seq_length = args.block_size - max_desc_seq_length - suffix_len
            features = {"input_ids": [], "attention_mask": [], "labels": []}
            for i in range(len(solutions)): # for each problem
                for code in solutions[i]: # for each solution
                    tokenized_code = tokenizer(
                        code, 
                        truncation=True,
                        max_length=max_code_seq_length)
                    # add eos token if tokenized code is shorter than max_code_seq_length
                    if len(tokenized_code['input_ids']) < max_code_seq_length:
                        tokenized_code['input_ids'].append(tokenizer.eos_token_id)
                        tokenized_code['attention_mask'].append(1)
                    # build samples: problem description + code
                    features["input_ids"].append(concat_desc["input_ids"][i] + tokenized_code["input_ids"])
                    features["attention_mask"].append(concat_desc["attention_mask"][i] + tokenized_code["attention_mask"])
                    # only predict code tokens
                    features["labels"].append([-100] * len(concat_desc["input_ids"][i]) + tokenized_code["input_ids"])
                    assert len(features["input_ids"][-1]) == len(features["attention_mask"][-1]) == len(features["labels"][-1])
                    assert len(features["input_ids"][-1]) <= args.block_size and len(features["attention_mask"][-1]) <= args.block_size and len(features["labels"][-1]) <= args.block_size, f"len(features['input_ids'][-1])={len(features['input_ids'][-1])}, len(features['attention_mask'][-1])={len(features['attention_mask'][-1])}, len(features['labels'][-1])={len(features['labels'][-1])}"
            # padding to max_length
            for i in range(len(features["input_ids"])):
                features["input_ids"][i] += [tokenizer.pad_token_id] * (args.block_size - len(features["input_ids"][i]))
                features["attention_mask"][i] += [0] * (args.block_size - len(features["attention_mask"][i]))
                features["labels"][i] += [-100] * (args.block_size - len(features["labels"][i]))

            return features
        
        with accelerator.main_process_first():
            codecontest_datasets = raw_datasets.map(
                prepare_codecontest,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Preprocessing Dataset",
            )
        if args.codecontest_finetune_data_ratio is not None:
            codecontest_datasets["train"] = codecontest_datasets["train"].shuffle(seed=args.seed).select(range(int(len(codecontest_datasets["train"]) * args.codecontest_finetune_data_ratio)))
            codecontest_datasets["validation"] = codecontest_datasets["validation"].shuffle(seed=args.seed).select(range(int(len(codecontest_datasets["validation"]) * args.codecontest_finetune_data_ratio)))
               
        if train_dataset is None:
            train_dataset = codecontest_datasets["train"]
            eval_dataset = codecontest_datasets["validation"]
        else:
            train_dataset = concatenate_datasets([train_dataset, codecontest_datasets["train"]])
            eval_dataset = concatenate_datasets([eval_dataset, codecontest_datasets["validation"]])

    # training on self-refined dataset
    if args.self_refine:
        data_files = {}
        dataset_args = {}
        if args.refine_train_file is not None:
            data_files["train"] = args.refine_train_file
        if args.refine_validation_file is not None:
            data_files["validation"] = args.refine_validation_file
        extension = args.refine_train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        if extension == "jsonl":
            extension = "json"
        refine_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in refine_datasets.keys():
            refine_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            refine_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
        # only predict the refined solution, given the prompt (including wrong solution)
        def tokenize_refine_dataset(examples):
            max_desc_seq_length = 1024 + 512 # 1024 for prompt, 512 for wrong solution
            tokenized_prompt = tokenizer(examples["prompt"], truncation=True, max_length=max_desc_seq_length)
            # max_code_seq_length = args.block_size - max_desc_seq_length
            # assert max_code_seq_length > 0
            # dynamic truncation for solution
            assert len(examples["solution"]) == len(examples["prompt"])
            tokenized_solutions = {"input_ids": [], "attention_mask": []}
            for i in range(len(examples["solution"])):
                prompt_length = len(tokenized_prompt['input_ids'][i])
                max_sol_length = args.block_size - prompt_length
                tokenized_sol= tokenizer(examples["solution"][i], truncation=True, max_length=max_sol_length)
                # add eos token if tokenized solution is shorter than max_code_seq_length
                if len(tokenized_sol['input_ids']) < max_sol_length:
                    tokenized_sol['input_ids'].append(tokenizer.eos_token_id)
                    tokenized_sol['attention_mask'].append(1)
                tokenized_solutions['input_ids'].append(tokenized_sol['input_ids'])
                tokenized_solutions['attention_mask'].append(tokenized_sol['attention_mask'])
            
            features = {"input_ids": [], "attention_mask": [], "labels": []}
            assert len(tokenized_prompt['input_ids']) == len(tokenized_solutions['input_ids'])
            for i in range(len(tokenized_prompt['input_ids'])):
                features["input_ids"].append(tokenized_prompt['input_ids'][i] + tokenized_solutions['input_ids'][i])
                if args.fcm_ratio is not None:
                    temp_start = REFINE_TEMPLATE.split("\n")[0]
                    tokenized_temp_start = tokenizer(temp_start)
                    # find tokenized_temp_start in tokenized_prompt['input_ids'][i]
                    start_idx = -1
                    for j in range(len(tokenized_prompt['input_ids'][i])):
                        if tokenized_prompt['input_ids'][i][j:j+len(tokenized_temp_start['input_ids'])] == tokenized_temp_start['input_ids']:
                            start_idx = j
                            break
                    # assign attention mask randomly
                    prompt_attention_mask = tokenized_prompt['attention_mask'][i][:]
                    prompt_attention_mask = torch.tensor(prompt_attention_mask)
                    probablity_matrix = torch.full((len(prompt_attention_mask),), args.fcm_ratio)
                    masked_indices = torch.bernoulli(probablity_matrix).bool()
                    if start_idx != -1:
                        masked_indices[:start_idx] = False # we don't mask the problem description
                    prompt_attention_mask[masked_indices] = 0
                    # change back to list
                    prompt_attention_mask = prompt_attention_mask.tolist()
                    features["attention_mask"].append(prompt_attention_mask + tokenized_solutions['attention_mask'][i])
                else:
                    features["attention_mask"].append(tokenized_prompt['attention_mask'][i] + tokenized_solutions['attention_mask'][i])
                features["labels"].append([-100] * len(tokenized_prompt['input_ids'][i]) + tokenized_solutions['input_ids'][i])
                assert len(features["input_ids"][-1]) <= args.block_size and len(features["attention_mask"][-1]) <= args.block_size and len(features["labels"][-1]) <= args.block_size, f"len(features['input_ids'][-1])={len(features['input_ids'][-1])}, len(features['attention_mask'][-1])={len(features['attention_mask'][-1])}, len(features['labels'][-1])={len(features['labels'][-1])}"
            # padding to max_length
            for i in range(len(features["input_ids"])):
                features["input_ids"][i] += [tokenizer.pad_token_id] * (args.block_size - len(features["input_ids"][i]))
                features["attention_mask"][i] += [0] * (args.block_size - len(features["attention_mask"][i]))
                features["labels"][i] += [-100] * (args.block_size - len(features["labels"][i]))
            
            return features

        refine_column_names = refine_datasets["train"].column_names
        with accelerator.main_process_first():
            refine_proc_datasets = refine_datasets.map(
                tokenize_refine_dataset,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=refine_column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on refine dataset",
            )

        if train_dataset is None:
            train_dataset = refine_proc_datasets["train"]
            eval_dataset = refine_proc_datasets["validation"]
        else:
            train_dataset = concatenate_datasets([train_dataset, refine_proc_datasets["train"]])
            eval_dataset = concatenate_datasets([eval_dataset, refine_proc_datasets["validation"]])


    return train_dataset, eval_dataset

def get_optimizer_scheduler(model, train_dataloader_len):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.95), eps=1e-8,lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    num_train_epochs = args.num_train_epochs
    if args.max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        max_train_steps = args.max_train_steps
    print(f"Debugging info: Line 238 num_train_epochs: {num_train_epochs}, num_update_steps_per_epoch: {num_update_steps_per_epoch}, max_train_steps: {max_train_steps}")
        
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps,
    )

    return optimizer, lr_scheduler

def train(model, tokenizer, train_dataset, eval_dataset, train_dataloader, eval_dataloader, optimizer, lr_scheduler):
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # if train_dataloader is a distributed sampler, on each machine it only sees num_train_examples / num_devices examples
    # --> len(train_dataloader) == num_train_examples / (args.per_device_train_batch_size * accelerator.num_processes)
    num_train_epochs = args.num_train_epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        print(f"Debugging info: Line 256 num_train_epochs: {num_train_epochs}, num_update_steps_per_epoch: {num_update_steps_per_epoch}, max_train_steps: {max_train_steps}")
        print(f"Debugging info: Line 257 {num_train_epochs} * {num_update_steps_per_epoch} = {num_train_epochs * num_update_steps_per_epoch}")
    else:
        max_train_steps = args.max_train_steps
    logger.info(f"Debugging info: Line 258 max_train_steps: {max_train_steps}, num_train_epochs: {num_train_epochs}, num_update_steps_per_epoch: {num_update_steps_per_epoch}, len(train_dataloader): {len(train_dataloader)}, accelerator.num_processes: {accelerator.num_processes}, args.gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    logger.info(f"Debugging info: Line 261 max_train_steps: {max_train_steps}, num_train_epochs: {num_train_epochs}, num_update_steps_per_epoch: {num_update_steps_per_epoch}, len(train_dataloader): {len(train_dataloader)}, accelerator.num_processes: {accelerator.num_processes}, args.gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"Debugging info: Line 278 max_train_steps: {max_train_steps}, args.num_train_epochs: {num_train_epochs}, num_update_steps_per_epoch: {num_update_steps_per_epoch}, len(train_dataloader): {len(train_dataloader)}, accelerator.num_processes: {accelerator.num_processes}, args.gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.with_tracking:
                    # log learning rate
                    accelerator.log({"loss": loss.item()}, step=completed_steps)
                    accelerator.log({"learning_rate": optimizer.param_groups[0]["lr"]}, step=completed_steps)
                    # log training loss every `args.logging_steps` steps
                    if completed_steps % args.logging_steps == 0:
                        logger.info(f"epoch {epoch}: batch {step}: train_loss {loss.item()}")

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= max_train_steps:
                break

        # Evaluation! at the end of each epoch
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


def main():
    # Initialize Model
    config, tokenizer, model = model_init()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token

    # Initialize datasets
    train_dataset, eval_dataset = data_init(tokenizer)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 5):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        # debug: print out decoded text
        print(f"****\ninput_ids\n****{tokenizer.decode(train_dataset[index]['input_ids'])}")
        # labels will be those not -100
        labels = [x for x in train_dataset[index]['labels'] if x != -100]
        print(f"****\nlabels\n****{tokenizer.decode(labels)}")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Initialize optimizer and lr scheduler
    print(f"Debugging: {len(train_dataloader)}")
    optimizer, lr_scheduler = get_optimizer_scheduler(model, len(train_dataloader))
    # print out lr_scheduler type, num_warmup_steps and num_training_steps
    for k, v in lr_scheduler.__dict__.items():
        print(f"Debugging Line 445: {k}: {v}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # print out lr_scheduler type, num_warmup_steps and num_training_steps
    for k, v in lr_scheduler.scheduler.__dict__.items():
        print(f"Debugging Line 456: {k}: {v}")

    train(model, tokenizer, train_dataset, eval_dataset, train_dataloader, eval_dataloader, optimizer, lr_scheduler)

if __name__ == "__main__":
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    if sys.gettrace() is None:
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=14400))

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    

    main()

