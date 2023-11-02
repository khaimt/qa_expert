from gen_data import utility
import typer
import os
from typing import Dict


def update_dic_count(dic, c_value):
    dic[c_value] = dic.get(c_value, 0) + 1


def dump_count_dic(dic, path):
    result = ""
    for k, v in sorted(dic.items(), key=lambda x: -x[1]):
        assert len(k) > 0
        result += f"{k}, {v}\n"
    utility.save_text(result, path)


def main(train_path: str, save_folder: str):
    utility.create_folder(save_folder)
    items = utility.read_json(train_path)
    print("total of items: ", len(items))
    result: Dict[str, Dict] = {"llm": {}, "multihop": {}, "tag": {}, "sub_questions": {}, "negative": {}}
    others: Dict[str, Dict] = {"entity": {}, "attribute": {}}
    for item in items:
        llm = item["meta_info"]["llm"]
        update_dic_count(result["llm"], llm)
        update_dic_count(result["tag"], item["tag"])
        update_dic_count(result["multihop"], str(item["multihop"]))
        update_dic_count(result["sub_questions"], len(item["sub_questions"]))
        update_dic_count(result["negative"], str("negatives" in item["meta_info"]))
        meta_info = item["meta_info"]
        for attr in ["attribute_1", "attribute_2", "comparison_attribute"]:
            if attr in meta_info:
                update_dic_count(others["attribute"], meta_info[attr])
        for entity in ["entity_1", "entity_2", "entity"]:
            if entity in meta_info:
                update_dic_count(others["entity"], meta_info[entity])

    utility.save_json(result, os.path.join(save_folder, "stat.json"))

    dump_count_dic(others["entity"], os.path.join(save_folder, "entity.csv"))
    dump_count_dic(others["attribute"], os.path.join(save_folder, "attribute.csv"))


if __name__ == "__main__":
    typer.run(main)
