from transformers import LlamaTokenizer
import os
import json
import sys
from typing import Dict
import typer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from qa_expert import prompt_utils


def compute(pretrained_path: str, data_folder: str, threshold: int = typer.Option(default=1024)):
    """_summary_

    Args:
        pretrained_path (str): the pretrained model
        data_folder (str): The folder containing train.json and validation.json
        threshold (int, optional): The threshold that we will output the statistics of the number of data points with length bigger than this threshold
        For example, if threshold=1024, the script will output:

        length=2509, count=1, acc_count: 1
        length=2354, count=1, acc_count: 2
        length=2097, count=1, acc_count: 3
        length=2071, count=1, acc_count: 4
        ...
        length=1028, count=8, acc_count: 1235
        length=1027, count=7, acc_count: 1242
        length=1026, count=7, acc_count: 1249
        length=1025, count=3, acc_count: 1252

        + count: the number of data points with this length;
        + acc_count: the number of total data points longer than or equal this length
        --> the purpose is to support us to choose the max_sequence_length in finetuning
    """
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
        with open(path, "r") as f:
            examples = json.loads(f.read())
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


if __name__ == "__main__":
    typer.run(compute)
