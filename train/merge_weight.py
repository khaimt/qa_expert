import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoModelForCausalLM, LlamaTokenizer
from qa_expert.prompt_utils import get_additional_tokens
from peft import PeftModel
import torch
import typer


def merge_weight(save_folder: str, pretrained_path: str, checkpoint: str):
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_path, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token  # Llama needs this
    added_tokens = get_additional_tokens()
    print("added token: ", added_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        device_map="auto",
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
    )
    print("model = ", model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_model = PeftModel.from_pretrained(model, checkpoint, torch_dtype=torch.float16)
    lora_model = lora_model.merge_and_unload()
    lora_model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)
    print("final lora model: ", lora_model)


if __name__ == "__main__":
    typer.run(merge_weight)
