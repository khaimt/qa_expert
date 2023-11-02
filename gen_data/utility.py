import json
from pathlib import Path
import os
import sys
from qa_expert.prompt_utils import Message
from typing import List
import csv
from typing import Tuple


def read_category(path: str) -> List[str]:
    category_content = read_text(path)
    categories = set()
    for line in category_content.split("\n"):
        parts = line.split(",")
        for item in parts:
            if len(item.strip()) > 0:
                categories.add(item.strip().title())
    return list(categories)


def read_category_lines(path: str):
    text = read_text(path)
    items = []
    for item in text.split("\n"):
        if len(item.strip()) > 0:
            items.append(item.strip())
    return items


def read_text(path):
    with open(path, "r") as f:
        return f.read()


def create_folder(folder: str):
    if os.path.exists(folder):
        return
    path = Path(folder)
    path.mkdir(parents=True)


def read_jsonl(path):
    result = []
    with open(path, "r") as f:
        for line in f:
            temp_line = line.strip()
            if len(temp_line) > 0:
                result.append(json.loads(temp_line))
    return result


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def save_json(data, path):
    with open(path, "w") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))


def save_text(text, path):
    with open(path, "w") as f:
        f.write(text)


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def beautify_json(path):
    data = read_json(path)
    save_json(data, path)


def save_csv(rows, path):
    with open(path, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


# this code is copied from: https://github.com/MeetKai/functionary/blob/main/functionary/train/custom_datasets.py#L193
def merge_data_points_by_length(lengths: List[int], max_length: int) -> List[List[int]]:
    """given lengths of data points, we merge them into groups such that the sum of lengths
    in each group is less than max_length. This is known as: https://en.wikipedia.org/wiki/Bin_packing_problem
    Here is the greedy algorithm
    Args:
        lengths (List[int]): _description_
        max_length (int): _description_

    Returns:
        _type_: groups of indices: [[index1, index2, ...], [], ...]
    """
    items = [{"length": length, "index": i} for i, length in enumerate(lengths)]
    items = sorted(items, key=lambda x: x["index"])
    merges = []
    current_sum = 0
    current_list = []
    for i in range(len(items)):
        cur_length = items[i]["length"]
        if cur_length + current_sum <= max_length:
            current_sum += items[i]["length"]
            current_list.append(i)
        else:
            merges.append(current_list)
            current_list = [i]
            current_sum = cur_length
    if len(current_list) > 0:
        merges.append(current_list)
    result = []
    for merge in merges:
        sub_items = [items[index]["index"] for index in merge]
        result.append(sub_items)
    return result


def get_batch_indices(size: int, batch_size: int) -> List[Tuple[int, int]]:
    result = []
    for i in range(size // batch_size + 1):
        start = i * batch_size
        end = i * batch_size + batch_size
        if end > size:
            end = size
        if end > start:
            result.append((start, end))
    return result


def main():
    beautify_json(sys.argv[1])


if __name__ == "__main__":
    main()
