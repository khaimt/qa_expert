import json 
from pathlib import Path
import os


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
    with open(path, "r") as f:
        return json.loads(f.read())


def save_json(data, path):
    with open(path, "w") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")