# mypy: ignore-errors
# This script is written based on: https://github.com/MeetKai/functionary/blob/main/functionary/train/train_lora.py
import os
import sys
from typing import Dict
from datasets import load_dataset
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from qa_expert.prompt_utils import get_additional_tokens

from train.custom_datasets import FAPackedDataset, CustomDataset

from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import bitsandbytes as bnb

import transformers
from transformers import (
    LlamaTokenizerFast,
    LlamaTokenizer,
    BitsAndBytesConfig,
)
import torch
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional
import random
from torch.utils.data import DataLoader
from train.new_mistral import MistralForCausalLM
import deepspeed

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
print("local rank: ", LOCAL_RANK)


class DataCollatorForMaskingLabels:
    """This data collator is used for dynamic padding.
    All the data points will be padded to the max length of the mini-batch instead of the whole dataset
    This will reduce the training time considerably when your data points are not uniform in terms of length
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.padding_side = self.tokenizer.padding_side

    def __call__(self, examples, return_tensors=None) -> Any:
        input_lengs = []
        for ex in examples:
            input_lengs.append(len(ex["input_ids"]))
        max_leng = max(input_lengs)
        result: Dict[str, Any] = {key: [] for key in examples[0].keys()}
        added_pad_dic = {"input_ids": self.tokenizer.pad_token_id, "labels": -100, "attention_mask": 0}

        for example in examples:
            pad_leng = max_leng - len(example["input_ids"])
            for key in result:
                if self.padding_side == "right":
                    result[key].append(example[key] + [added_pad_dic[key] for _ in range(pad_leng)])
                else:
                    result[key].append([added_pad_dic[key] for _ in range(pad_leng)] + example[key])

        for key in result:
            result[key] = torch.tensor(result[key])
        return result


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    model_type: str = field(default="llama")
    use_lora: bool = field(default=True)
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    qlora: bool = field(default=False, metadata={"help": "whether using qlora or not"})


@dataclass
class DataArguments:
    train_path: str = field(default="", metadata={"help": "Path to the training data."})
    validation_path: str = field(default="", metadata={"help": "Path to the evaluation data"})
    hf_data_path: str = field(
        default="khaimaitien/qa-expert-multi-hop-qa-V1.0", metadata={"help": "dataset from HF hub"}
    )
    packing: bool = field(default=False, metadata={"help": "Whether use packing or not"})
    train_ratio: float = field(default=1, metadata={"help": "percentage of training data to use"})
    validation_ratio: float = field(default=1, metadata={"help": "percentage of validation data to use"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    return config


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_device_map(training_args: TrainingArguments, model_args: ModelArguments) -> Optional[Dict]:
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if model_args.qlora:
        if ddp and training_args.fsdp:
            print("FSDP is incompatible with QLORA")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            print("FSDP and ZeRO3 are both currently incompatible with QLoRA.")
    return device_map


def load_model(data_args: DataArguments, training_args: TrainingArguments, model_args: ModelArguments, tokenizer: Any):
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    if model_args.model_type == "llama":
        model_max_length = model_args.model_max_length
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        print_rank0(f"rope scaling for llamam original context length: {orig_ctx_len}, extended to: {model_max_length}")
        if orig_ctx_len and model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        config.use_cache = False

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    if data_args.packing and model_args.model_type == "mistral":  # have to monkey-patch
        model_class = MistralForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    print_rank0("QLORA: ", model_args.qlora)

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        device_map=get_device_map(training_args, model_args),
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=compute_dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            use_flash_attention_2=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if model_args.qlora
        else None,
    )
    print_rank0("model = ", model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    if model_args.qlora and model_args.use_lora:
        model = prepare_model_for_kbit_training(model)
    if model_args.use_lora:
        print("USE LORA TRAINING, START FINDING LINEAR LAYERS NOW")
        modules = find_all_linear_names(model)
        print_rank0("linear modules: ", modules)  # ["query_key_value"]
        model = get_peft_model(model, create_peft_config(modules))
    model.config.use_cache = False
    print_trainable_parameters(model)
    return model


def print_rank0(*arg):
    if LOCAL_RANK == 0:
        print(*arg)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    lora_param_count = 0
    all_param = 0
    embedding_lm_head_param_count = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            print_rank0(f"trainable: {name}, num_params: {num_params}")
            if "lm_head" in name or "embed_tokens" in name:
                embedding_lm_head_param_count += num_params
            else:
                lora_param_count += num_params
    trainable_params = embedding_lm_head_param_count + lora_param_count
    print_rank0(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    print_rank0(f"embedding_lm_head_param_count={embedding_lm_head_param_count}||loara_param={lora_param_count}")
    print_rank0(
        f"embedding_lm_head_param_count %={embedding_lm_head_param_count * 100 / all_param}||loara_param %={lora_param_count * 100 / all_param}"
    )


def print_some_examples(ds, tokenizer):
    data_loader = DataLoader(ds, batch_size=3)
    count = 0
    for batch in data_loader:
        if count == 0:
            print_rank0("keys in batch: ", batch.keys())
        print_rank0("--------------****Example data point****---------------")
        print("device: ", batch["input_ids"].device)
        print_rank0("shape of input_ids: ", batch["input_ids"].shape)  # B x L
        print_rank0("shape of labels: ", batch["labels"].shape)
        print_rank0("shape of attention_mask: ", batch["attention_mask"].shape)
        # print_rank0('input_ids: ', batch["input_ids"].tolist())
        # print_rank0('labels: ', batch["labels"].tolist())
        print_rank0("attention mask: ", batch["attention_mask"])
        input_ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()
        for i in range(len(labels)):
            if labels[i] == -100:
                labels[i] = tokenizer.pad_token_id
        print_rank0("++++input_ids: ")
        print_rank0(tokenizer.decode(input_ids))
        print_rank0("++++labels: ")
        print_rank0(tokenizer.decode(labels))
        count += 1
        if count == 3:
            break


def read_dataset(data_args: DataArguments, training_args: TrainingArguments, tokenizer: Any, ds_type: str):
    ds_class = CustomDataset
    if data_args.packing:
        ds_class = FAPackedDataset  # if packing --> Use PackedDataset

    # The way we read dataset is:
    # Rank 0 will process the dataset and save the result to cached_folder, other ranks will read from the cached_folder
    cached_folder = os.path.join(training_args.output_dir, f"{ds_type}_cached")

    if training_args.local_rank > 0:  # If this is not rank 0, stay here, wait for rank 0 to process the data
        print(f"process: {LOCAL_RANK} wait for main process to prepare the training data")
        torch.distributed.barrier()
    else:  # rank 0 process the data and save to cached_folder
        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)
        if not os.path.exists(cached_folder):
            os.mkdir(cached_folder)

        data_path = data_args.train_path if ds_type == "train" else data_args.validation_path
        data_ratio = data_args.train_ratio if ds_type == "train" else data_args.validation_ratio
        with open(data_path, "r") as file:
            raw_data = json.loads(file.read())
            if data_ratio < 1:
                size = int(len(raw_data) * data_ratio)
                raw_data = raw_data[:size]

        print(f"{ds_type} size: : {len(raw_data)}")
        # ignore_cached=True to ignore the cached if exist, rank 0 will always process the data
        ds = ds_class(raw_data, tokenizer, cached_folder=cached_folder, ignore_cached=True)
        print(f"process: {LOCAL_RANK} finish processing data")
        torch.distributed.barrier()  # allow other ranks to execute

    # All ranks will read the processed data from cached_path created by rank 0
    ds = ds_class(None, tokenizer, cached_folder=cached_folder, ignore_cached=False)
    if LOCAL_RANK == 0:
        ds.stat()  # print some statistics about the dataset
    return ds


def train():
    set_seed(100)
    argument_parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = argument_parser.parse_args_into_dataclasses()
    pretrained_model = model_args.model_name_or_path

    # initialize tokenizer
    # if model_args.model_type == "llama":
    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model, legacy=True, model_max_length=model_args.model_max_length
    )
    tokenizer.pad_token = tokenizer.eos_token  # Llama needs this
    if model_args.model_type == "mistral":
        print_rank0("set padding_side = left for Mistral")
        tokenizer.padding_side = "left"
    added_tokens = get_additional_tokens()
    print_rank0("added token: ", added_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})
    print_rank0("total number of tokens: ", len(tokenizer))
    print_rank0("tokenizer: ", tokenizer)

    # read data
    train_ds = read_dataset(data_args, training_args, tokenizer, "train")
    valid_ds = read_dataset(data_args, training_args, tokenizer, "validation")
    print_rank0(f"train_size: {len(train_ds)}; validation_size: {len(valid_ds)}")

    print_some_examples(train_ds, tokenizer)
    model = load_model(data_args, training_args, model_args, tokenizer)

    print_rank0("training args: \n", training_args.to_json_string())
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=training_args,
    )

    print_rank0("Training ...")
    trainer.train()


if __name__ == "__main__":
    train()
