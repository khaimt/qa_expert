from transformers import LlamaTokenizerFast
import os
import json
import sys
from typing import Dict
import typer
import datetime
from qa_expert import prompt_utils
from gen_data import utility


def main(pretrained_path: str, train_path: str, save_folder: str, max_length: int):
    tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_path, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": prompt_utils.get_additional_tokens()})

    with open(train_path, "r") as f:
        examples = json.loads(f.read())
    print(f"handle: {train_path}, number of examples: {len(examples)}")

    all_prompts = []
    for example in examples:
        messages = prompt_utils.convert_multi_qa_format_to_messages(example)
        prompt = prompt_utils.get_prompt_from_messages(messages)
        all_prompts.append(prompt)

    count_dic: Dict[int, int] = {}
    batches = utility.get_batch_indices(len(all_prompts), batch_size=2000)
    t1 = datetime.datetime.now()
    for index, (start, end) in enumerate(batches):
        inputs = tokenizer(all_prompts[start:end])["input_ids"]
        for item in inputs:
            length = len(item)
            count_dic[length] = count_dic.get(length, 0) + 1
        t2 = datetime.datetime.now()
        acc_time = (t2 - t1).total_seconds()
        avg_time = acc_time / (index + 1)
        print(f"{index} / {len(batches)}; avg_time: {avg_time}; remaining time: {avg_time * (len(batches) - index -1)}")

    sorted_lengths = sorted(count_dic.items(), key=lambda x: x[0])
    acc_count = 0
    pairs = []
    rows = []
    for length, count in sorted_lengths:
        acc_count += count
        pairs.append((length, acc_count))
        rows.append((str(length), str(count)))
    rows.reverse()

    utility.save_csv([("length", "count")] + rows, f"{save_folder}/length_dic_count.csv")
    total_count = acc_count
    assert total_count == len(all_prompts)
    pairs.reverse()
    acc_rows = [("length", "accumulated_count", "percentage")]
    for i in range(len(pairs)):
        length, count = pairs[i]
        acc_rows.append((str(length), str(count), str(count / total_count)))
    utility.save_csv(acc_rows, f"{save_folder}/accumulated.csv")

    lengths = []
    for key in count_dic:
        frequency = count_dic[key]
        key = min(key, max_length)
        lengths.extend([key for _ in range(frequency)])
    assert len(lengths) == len(all_prompts)
    groups = utility.merge_data_points_by_length(lengths, max_length)
    original_ave_length = sum(lengths) / len(lengths)
    packed_lengths = []
    for indices in groups:
        packed_lengths.append(sum(lengths[index] for index in indices))
    packed_ave_length = sum(packed_lengths) / len(packed_lengths)
    print("number of data points after being packed: ", len(groups))
    print(f"original average length: {original_ave_length}, packed average length: {packed_ave_length}")


if __name__ == "__main__":
    typer.run(main)
