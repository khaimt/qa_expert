from transformers import LlamaTokenizer
from train.monkey_patched_mistral_packed_attention_mask import MistralForCausalLM
from train import custom_datasets
import torch
import copy
import typer
import os
import math
from qa_expert import prompt_utils
import json


def read_raw_data():
    cur_folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_folder, "test.json"), "r") as f:
        return json.loads(f.read())


def prepare_input_dic(input_dic, device):
    result = copy.deepcopy(input_dic)
    for key in result:
        result[key] = torch.unsqueeze(input_dic[key], 0)
        result[key] = result[key].to(device)
    result["return_dict"] = True
    result["loss_reduction"] = "sum"
    return result


def compute_loss_from_ds(ds, model, device):
    total_loss = 0
    for i in range(len(ds)):
        input_dic = ds[i]
        input_dic = prepare_input_dic(input_dic, device)
        with torch.no_grad():
            loss = model.forward(**input_dic).loss.item()
            total_loss += loss
    return total_loss


def main(pretrained_path: str, device: str = typer.Option("cuda:0")):
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_path, legacy=True, model_max_length=4096)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": prompt_utils.get_additional_tokens()})
    
    model = MistralForCausalLM.from_pretrained(
            pretrained_path, torch_dtype=torch.bfloat16, device_map=device, use_flash_attention_2=False
        )
    model.resize_token_embeddings(len(tokenizer))

    model.eval()

    raw_data = read_raw_data()
    for padding_side in ["left", "right"]:
        print("test padding_side: ", padding_side)
        tokenizer.padding_side = padding_side
        normal_ds = custom_datasets.CustomDataset(raw_data, tokenizer)
        packed_ds = custom_datasets.PackedDataset(raw_data, tokenizer)
        print("number of data points from normal ds: ", len(normal_ds))
        print("number of data points from packed ds: ", len(packed_ds))
        normal_loss = compute_loss_from_ds(normal_ds, model, device)
        mk_loss = compute_loss_from_ds(packed_ds, model, device)
        diff = math.fabs(normal_loss - mk_loss)
        diff_percent = diff * 100 / max(normal_loss, mk_loss)
        print(f"normal_loss: {normal_loss}, mk_loss={mk_loss}, diff_percent={diff_percent}%")


if __name__ == "__main__":
    typer.run(main)
