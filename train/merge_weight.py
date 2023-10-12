import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoModelForCausalLM, LlamaTokenizer
from qa_expert.prompt_utils import SpecialToken
from peft import PeftModel, PeftConfig
import torch


def merge_weight(save_folder, pretrained, checkpoint, model_type="mistral"):
    tokenizer = LlamaTokenizer.from_pretrained(pretrained, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token  # Llama needs this
    if model_type == "mistral":
        print("set padding_side = left for Mistral")
        tokenizer.padding_side = "left"
    added_tokens = [tok.value for tok in SpecialToken]
    print("added token: ", added_tokens)
    tokenizer.add_tokens(added_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained,
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


def main():
    merge_weight("merged_models/mistral", "pretrained/Mistral-7B-v0.1", "models/qa_v1_mistral/checkpoint-400")


if __name__ == "__main__":
    main()
