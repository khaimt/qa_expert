from transformers import LlamaTokenizer
import os
import json
import sys
from typing import Dict
import typer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from qa_expert import prompt_utils


def read_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def compute(pretrained_path: str, data_folder: str, threshold: int = typer.Argument(default=1024)):
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_path, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    added_tokens = [tok.value for tok in prompt_utils.SpecialToken]
    tokenizer.add_tokens(added_tokens)
    leng_dic: Dict[int, int] = {}
    sum_length = 0
    count = 0
    for ds in ["train", "validation", "test"]:
        path = os.path.join(data_folder, f"{ds}.json")
        if not os.path.exists(path):
            continue
        examples = read_json(path)
        print(f"handle: {path}, number of examples: {len(examples)}")
        for example in examples:
            messages = prompt_utils.convert_multi_qa_format_to_messages(example)
            input_dic = prompt_utils.preprare_training_inputs(messages, tokenizer, padding=True, max_length=8000)
            length = len(input_dic["input_ids"])
            leng_dic[length] = leng_dic.get(length, 0) + 1
            sum_length += length
            count += 1

    sorted_keys = sorted(list(leng_dic.keys()), key=lambda x: -x)
    total_count = 0
    for key in sorted_keys:
        if key > threshold:
            total_count += leng_dic[key]
            print(f"length={key}, count={leng_dic[key]}, acc_count: {total_count}")
    print("total_count=", total_count)
    print("avg_length: ", sum_length / count)
    print("total items: ", count)
    max_leng = max(list(leng_dic.keys()))
    print("number of leng: ", len(leng_dic))
    print("max_leng: ", max_leng, "frequencies: ", leng_dic[max_leng])
    print("distribution over top 50")


if __name__ == "__main__":
    typer.run(compute)
