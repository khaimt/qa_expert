import json
from pathlib import Path
import os
import sys
from qa_expert.prompt_utils import Message
from typing import List
import csv


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


def main():
    beautify_json(sys.argv[1])


if __name__ == "__main__":
    main()
