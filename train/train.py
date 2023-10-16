# mypy: ignore-errors
import os
import sys
from typing import Dict
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

os.environ["WANDB_LOG_MODEL"] = "all"

from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import bitsandbytes as bnb

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    BitsAndBytesConfig,
    MistralForCausalLM,
)
import torch
import math
from qa_expert.prompt_utils import SpecialToken, preprare_training_inputs, convert_multi_qa_format_to_messages
import os
from dataclasses import dataclass, field
from typing import Any, Optional
import random
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, DataCollatorWithPadding

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
print("local rank: ", LOCAL_RANK)


class DataCollatorForMaskingLabels:
    """This data collator is used for dynamic padding.
    All the data points will be padded to the max length of the mini-batch instead of the whole dataset
    This will reduce the training time considerably when your data points are not uniform in terms of length
    """

    def __init__(self, tokenizer, padding_side="left") -> None:
        self.tokenizer = tokenizer
        self.padding_side = padding_side

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
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


@dataclass
class PaddingArguments:
    padding: Optional[str] = field(default="longest")
    max_sequence_length: Optional[int] = field(default=4096)


@dataclass
class DataArguments:
    train_path: str = field(default="", metadata={"help": "Path to the training data."})
    validation_path: str = field(default="", metadata={"help": "Path to the evaluation data"})
    hf_data_path: str = field(default="khaimaitien/qa-expert-multi-hop-qa-V1.0", metadata={"help": "dataset from HF hub"})


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
    cls = bnb.nn.Linear4bit  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_device_map() -> Optional[Dict]:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    return device_map


def load_model(training_args, model_args, tokenizer):
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

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        device_map=get_device_map(),
        trust_remote_code=True,
        use_flash_attention_2=True,
        quantization_config=create_bnb_config(),
    )
    print_rank0("model = ", model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(model)
    print_rank0("linear modules: ", modules)  # ["query_key_value"]

    model = get_peft_model(model, create_peft_config(modules))
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
    data_loader = DataLoader(ds, batch_size=3, collate_fn=DataCollatorForMaskingLabels(tokenizer, "left"))
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


def train():
    set_seed(100)
    argument_parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, PaddingArguments)
    )
    model_args, data_args, training_args, padding_args = argument_parser.parse_args_into_dataclasses()
    pretrained_model = model_args.model_name_or_path

    # initialize tokenizer
    # if model_args.model_type == "llama":
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token  # Llama needs this
    if model_args.model_type == "mistral":
        print_rank0("set padding_side = left for Mistral")
        tokenizer.padding_side = "left"
    added_tokens = [tok.value for tok in SpecialToken]
    print("added token: ", added_tokens)
    tokenizer.add_tokens(added_tokens)
    print("total number of tokens: ", len(tokenizer))
    print("tokenizer: ", tokenizer)

    # read data
    if data_args.train_path:
        ds = load_dataset("json", data_files={"train": data_args.train_path, "validation": data_args.validation_path})
    elif data_args.hf_data_path:
        ds = load_dataset(data_args.hf_data_path)
    print(ds)
    print_rank0(f"padding_args: padding={padding_args.padding}, max_length: {padding_args.max_sequence_length}")

    def generate_prompt(example):
        messages = convert_multi_qa_format_to_messages(example)
        input_dic = preprare_training_inputs(
            messages, tokenizer, padding=padding_args.padding, max_length=padding_args.max_sequence_length
        )
        return input_dic

    original_columns = list(ds["train"].features.keys())
    print_rank0("original columns: ", original_columns)
    ds = ds.shuffle().map(generate_prompt, remove_columns=original_columns)
    print_rank0("ds after removing columns: ", ds)
    # just print out to see if there are any errors
    print_some_examples(ds["train"], tokenizer)
    model = load_model(training_args, model_args, tokenizer)

    train_ds, valid_ds = ds["train"], ds["validation"]
    print_rank0(f"train_size: {len(train_ds)}; validation_size: {len(valid_ds)}")
    print_rank0("training args: \n", training_args.to_json_string())
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=training_args,
        data_collator=DataCollatorForMaskingLabels(tokenizer, "left"),
    )
    model.config.use_cache = False

    # Verifying the datatypes before training
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    print_rank0("Training ...")
    trainer.train()


if __name__ == "__main__":
    train()
