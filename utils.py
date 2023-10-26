import argparse
from transformers import (
    MODEL_MAPPING,
    SchedulerType
)
import errno
import os
import signal
import functools
from nltk.tokenize import word_tokenize
import re

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

REFINE_TEMPLATE = (
    "### Bad Solution:\n\n"
    "{mod_completion}\n\n"
    "### Execution Failure:\n"
    "{mod_result}\n\n"
    "### Good solution:\n"
)

REFINE_TEMPLATE_BASELINE = (
    "Bad Solution:\n\n"
    "{completion}\n\n"
    "Execution Failure:\n"
    "{result}"
)

def tokenize_nltk(text):
    words = word_tokenize(text)
    output_list = []
    for w in words:
        w_list = re.findall(r'\w+', w)
        output_list.extend(w_list)
    return output_list

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    # newly added parameters
    parser.add_argument(
        "--fcm_ratio",
        type=float,
        default=None,
        help="The ratio of the number of tokens to be masked in the FCM task.",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."
            "Only applicable for models with checkpointed layers (e.g. BERT)."
        ), 
    )
    parser.add_argument(
        "--predict_type",
        type=str,
        default=None,
        choices=["full", "code_only", "instruction"],
        help="Predict full sequence (w/ prompt) or only the source code (w/o prompt)",
    )
    parser.add_argument(
        "--pretrain_data_ratio",
        type=float,
        default=None,
        help="Prevent overfitting to the task-specific data by using some pretraining data. The pre-training sample size will be ratio * task-specific data",
    )
    parser.add_argument(
        "--vanilla_codelm_pretrain",
        action="store_true",
        help="conduct vanilla code language modeling pretraining",
    )
    parser.add_argument(
        "--pretrain_file",
        type=str,
        default=None,
        help="A jsonl file containing the vanilla code language modeling pretraining data.",
    )
    parser.add_argument(
        "--with_instruct_data",
        action="store_true",
        help="use instruction data for pretraining",
    )
    parser.add_argument(
        "--instruct_train_file", type=str, default=None, help="A jsonl file containing the instruct training data."
    )
    parser.add_argument(
        "--instruct_validation_file", type=str, default=None, help="A jsonl validation data."
    )
    parser.add_argument(
        "--self_refine",
        action="store_true",
        help="use self-refinement training",
    )
    parser.add_argument(
        "--refine_train_file",
        type=str,
        default=None,
        help="A jsonl file containing the self-refinement training data.",
    )
    parser.add_argument(
        "--refine_validation_file",
        type=str,
        default=None,
        help="A jsonl validation data.",
    )
    parser.add_argument(
        "--codecontest_finetune",
        action="store_true",
        help="use codecontest finetuning",
    )
    parser.add_argument(
        "--codecontest_finetune_train_file",
        type=str,
        default=None,
        help="A jsonl file containing the codecontest finetuning training data.",
    )
    parser.add_argument(
        "--codecontest_finetune_validation_file",
        type=str,
        default=None,
        help="A jsonl validation data.",
    )
    parser.add_argument(
        "--codecontest_finetune_data_ratio",
        type=float,
        default=None,
        help="Support ablation study on the amount of codecontest finetuning data, 0.1 makes the refine data to be 75%, 0.3 makes the refine data to be 50%",
    )
    args = parser.parse_args()

    return args
