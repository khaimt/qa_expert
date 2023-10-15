import openai
import os
import sys
import utility
import datetime
import time
import re
from abc import ABC, abstractmethod
from typing import Any, Union, Dict, List, Tuple, Callable
import json
import typer


def get_response_from_chat_model(input_dic: Dict) -> Dict[str, Any]:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input_dic["prompt"]}],
        temperature=input_dic.get("temperature", 0),
        timeout=5,
    )
    # print("response: ", response)
    answer = response["choices"][0]
    finish_reason = answer["finish_reason"]
    total_tokens = response["usage"]["total_tokens"]
    return {
        "finish_reason": finish_reason,
        "total_tokens": total_tokens,
        "response": answer["message"]["content"].strip(),
    }


def get_response_from_completion_model(input_dic: Dict):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=input_dic["prompt"],
        temperature=input_dic.get("temperature", 0),
        max_tokens=1024,
        timeout=5,
    )
    text_response = response["choices"][0]["text"]
    finish_reason = response["choices"][0]["finish_reason"]
    total_tokens = response["usage"]["total_tokens"]
    return {"finish_reason": finish_reason, "total_tokens": total_tokens, "response": text_response.strip()}


def try_loop(function: Callable, params: Dict, max_count=5) -> Any:
    for i in range(max_count):
        try:
            result = function(params)
            return result
        except Exception as e:
            print("exception: ", str(e))
            time.sleep(2)
    return None


def generate_data_point_from_category(category, prompt):
    final_prompt = prompt.replace("{category}", category)
    result = try_loop(get_response_from_completion_model, {"prompt": final_prompt, "temperature": 1})
    if result is None:
        return None
    # parse text_response
    finish_reason, total_tokens, text_response = result["finish_reason"], result["total_tokens"], result["response"]
    fields = [
        "Entries",
        "Attribute to compare",
        "Question for 2 entries",
        "Question 1",
        "Question 2",
        "Supporting paragraph 1",
        "Supporting paragraph 2",
        "Answer 1",
        "Answer 2",
        "Thought",
        "Final answer",
    ]
    dic = {}
    for i in range(len(fields)):
        cur_field = fields[i]
        if i < len(fields) - 1:
            next_field = fields[i + 1]
            pattern = rf"\+\s{cur_field}:(?P<content>(.|\n)+)\+\s{next_field}"
        else:  # last field
            pattern = rf"\+\s{cur_field}:(?P<content>(.|\n)+)"
        match = re.search(pattern, text_response)
        if match is not None:
            content = match.group("content").strip()
            dic[cur_field] = content
        else:
            print(f"cannot parse: {cur_field}")
            print(text_response)
            print("--------------------")
            return None
    thought = dic["Thought"]
    final_answer = dic["Final answer"]
    record = {
        "question": dic["Question for 2 entries"],
        "sub_questions": [
            {"question": dic["Question 1"], "long_answer": dic["Answer 1"], "paragraph": dic["Supporting paragraph 1"]},
            {"question": dic["Question 2"], "long_answer": dic["Answer 2"], "paragraph": dic["Supporting paragraph 2"]},
        ],
        "answer": f"{thought}\nFinal answer:{final_answer}",
        "final_answer": f"{thought}\nFinal answer:{final_answer}",
        "entries": dic["Entries"],
        "category": category,
    }
    return {"finish_reason": finish_reason, "total_tokens": total_tokens, "result": record}


class GenTask(ABC):
    def __init__(self, save_path: str, **kwargs) -> None:
        self.save_path = save_path
        self.result = []
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                self.result = json.loads(f.read())

    @abstractmethod
    def count_number_of_remaining_items(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def handle_item(self, item: Any) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_items_for_handling(self):
        raise NotImplementedError

    def run(self):
        total_count = self.count_number_of_remaining_items()
        acc_tokens = 0
        handled_count = 0
        t1 = datetime.datetime.now()
        print(f"number of items to handle: {total_count}")
        for item in self.get_items_for_handling():
            item_res = self.handle_item(item)
            self.result.append(item_res["result"])
            utility.save_json(self.result, self.save_path)
            acc_tokens += item_res["total_tokens"]
            t2 = datetime.datetime.now()

            handled_count += 1
            avg_time = (t2 - t1).total_seconds() / handled_count
            avg_tokens = acc_tokens / handled_count
            remaining_count = total_count - handled_count

            estimated_total_money = avg_tokens * total_count * 0.002 / 1000
            current_money = acc_tokens * 0.002 / 1000

            print(
                (
                    f"{handled_count}/{total_count}, avg_time: {avg_time}, remaining time: {avg_time * remaining_count},"
                    f"avg_token: {avg_tokens}, current_money: {current_money}, total_money: {estimated_total_money}"
                )
            )


class GenAnswer(GenTask):
    def __init__(self, save_path: str, **kwargs) -> None:
        super().__init__(save_path, **kwargs)
        input_path = kwargs["input_path"]
        self.input_items = utility.read_json(input_path)
        self.sub_question_prompt = utility.read_text(kwargs["subquestion_prompt"])
        self.final_question_prompt = utility.read_text(kwargs["final_prompt"])

    def get_items_for_handling(self):
        # start generating
        for item in self.input_items[len(self.result) :]:
            yield item

    def count_number_of_remaining_items(self) -> int:
        return len(self.input_items) - len(self.result)

    def handle_item(self, item: Any) -> Dict:
        subs = item["sub_questions"]
        total_token_count = 0
        # first generate answer for sub-question
        for sub in subs:
            if "long_answer" not in sub:
                question = sub["question"]
                context = sub["paragraph"]
                final_prompt = self.sub_question_prompt.format(**{"question": question, "context": context})
                result = try_loop(get_response_from_chat_model, {"prompt": final_prompt})
                if result is None:
                    print("cannot use though trying 5 times for sub-questions")
                    sys.exit(1)
                finish_reason, tok_count, response = result["finish_reason"], result["total_tokens"], result["response"]
                total_token_count += tok_count
                if finish_reason == "stop":
                    sub["long_answer"] = response
                else:
                    print("finish_reason is not stop: ", finish_reason)
                    sys.exit(1)
        ## get the final answer here
        question = item["question"]
        subs = item["sub_questions"]
        facts = [sub["long_answer"] for sub in subs]
        fact_str = "\n".join([f"+ {fact}" for fact in facts])
        final_prompt = self.final_question_prompt.format(**{"question": question, "facts": fact_str})
        result = try_loop(
            get_response_from_completion_model, {"prompt": final_prompt}
        )  # get_response_from_openai(final_prompt)
        if result is None:
            print("cannot use though trying 5 times for final answer")
            sys.exit(1)
        finish_reason, tok_count, response = result["finish_reason"], result["total_tokens"], result["response"]
        total_token_count += tok_count
        if finish_reason == "stop":
            item["final_answer"] = response
        else:
            print("finsih_reason is not stop at final_question: ", finish_reason)
            sys.exit(1)
        return {"total_tokens": total_token_count, "result": item}


class GenComparisonQA(GenTask):
    def __init__(self, save_path: str, **kwargs) -> None:
        super().__init__(save_path, **kwargs)
        category_content = utility.read_text(kwargs["category_path"])
        self.prompt_template = utility.read_text(kwargs["prompt"])
        self.num_items_per_category = kwargs.get("num_items_per_category", 30)
        categories = set()
        for line in category_content.split("\n"):
            parts = line.split(",")
            for item in parts:
                if len(item.strip()) > 0:
                    categories.add(item.strip().title())
        self.categories = list(categories)
        print("list of category: ", self.categories)

        # update the current number of items in each category
        self.current_category_count = {cat: 0 for cat in categories}
        for item in self.result:
            self.current_category_count[item["category"]] += 1

        # compute the remaining number for each category
        self.category_count = {}  # mapping from category --> number of remaining items
        for category in self.current_category_count:
            self.category_count[category] = self.num_items_per_category - self.current_category_count[category]
            if self.category_count[category] <= 0:  # no need to call --> delete
                del self.category_count[category]

    def count_number_of_remaining_items(self) -> int:
        return self.num_items_per_category * len(self.categories) - len(self.result)

    def get_items_for_handling(self):
        while len(self.category_count) > 0:
            for category in self.categories:
                if self.category_count[category] > 0:  # if still need to generate for this category
                    self.category_count[category] -= 1
                    if self.category_count[category] == 0:  # delete if this category is done
                        del self.category_count[category]
                    yield category

    def handle_item(self, item: Any) -> Dict:
        for _ in range(10):
            result = generate_data_point_from_category(item, self.prompt_template)
            if result is not None:
                return result
        return {}


def generate_comparison_data_points(save_path, num_per_label):
    category_path = "prompts/categories.txt"
    promt_path = "prompts/comparison_gen.txt"
    kwargs = {"category_path": category_path, "prompt": promt_path, "num_items_per_category": num_per_label}
    task = GenComparisonQA(save_path, **kwargs)
    task.run()


def fill_in_sub_answers_long_answers(input_path, save_path):
    kwargs = {
        "input_path": input_path,
        "subquestion_prompt": "prompts/answer_gen.txt",
        "final_prompt": "prompts/final_answer_gen.txt",
    }
    task = GenAnswer(save_path, **kwargs)
    task.run()


def post_process_entry_text(entry_text: str) -> str:
    prefixs = ["generate 2 random entries of Event. The format:", "Generate 2 random entries of Event. The format:"]
    for prefix in prefixs:
        if entry_text.startswith(prefix):
            entry_text = entry_text[len(prefix) :].strip()
    return entry_text


def remove_low_quality_data(data_path: str, save_path: str, keep_answer: bool = False):
    items = utility.read_json(data_path)
    invalid_entry_count = 0
    failed_items = []
    good_items = []
    for item in items:
        if item is None:
            continue
        entry_text = post_process_entry_text(item["entries"])
        entries = entry_text.split(";")
        if len(entries) != 2:
            invalid_entry_count += 1
        else:
            entries = [entry.strip() for entry in entries]
            question = item["question"]
            check1 = entries[0].lower() in question.lower()
            check2 = entries[1].lower() in question.lower()
            check3 = entries[0].lower() in item["sub_questions"][0]["question"].lower()
            check4 = entries[1].lower() in item["sub_questions"][1]["question"].lower()
            if check1 and check2 and check3 and check4:
                good_items.append(item)
            else:
                failed_items.append(item)
    print("total number of items: ", len(items))
    print("failed at entries: ", invalid_entry_count)
    print("number of items failed at check: ", len(failed_items))
    print("number of good_items: ", len(good_items))
    if keep_answer:
        utility.save_json(good_items, save_path)
    else:
        result = []
        for item in good_items:
            new_item = {"question": item["question"], "answer": None}
            subs = []
            for sub in item["sub_questions"]:
                subs.append({"question": sub["question"], "paragraph": sub["paragraph"]})
            new_item["sub_questions"] = subs
            result.append(new_item)
        utility.save_json(result, save_path)


def main(
    num_items_per_category: int = typer.Option(default=100),
    output_folder: str = typer.Option(default="gen_qa"),
    re_generate_answer: bool = typer.Option(False, "--re-generate-answer"),
    category_path: str = typer.Option(default="extra_files/categories.txt"),
    continue_gen: bool = typer.Option(True, "--no-continue"),
):
    """this function is used to generate multi-hop Q&A

    Args:
        num_items_per_category (int, optional): number of generated items for each category. Defaults to typer.Option(default=100).
        output_folder (str, optional): where to save the result. Defaults to typer.Option(default="gen_qa").
        re_generate_answer (bool, optional): If we re-generate the answers to single questions and final answer to the multi-hop question or not.
            if re-generate, we will use the prompt template for generating the answer + temperature=0
        category_path (str, optional): The path to list of categories. Defaults to typer.Option(default="extra_files/categories.txt").
        continue_gen (bool, optional): if we continue to generate from current result or not. Defaults to typer.Option(True, "--no-continue").
    """
    if not os.path.exists(output_folder):
        utility.create_folder(output_folder)
    kwargs = {
        "category_path": category_path,
        "prompt": "extra_files/comparison_gen.txt",
        "num_items_per_category": num_items_per_category,
    }
    multi_hop_qa_path = os.path.join(output_folder, "raw_multi_hop_qa.json")
    if os.path.exists(multi_hop_qa_path) and not continue_gen:
        os.remove(multi_hop_qa_path)
    print("Start to generate multi-hop QA now")
    task = GenComparisonQA(multi_hop_qa_path, **kwargs)
    task.run()

    high_quality_data_path = os.path.join(output_folder, "filtered.json")
    remove_low_quality_data(multi_hop_qa_path, high_quality_data_path, keep_answer=not re_generate_answer)
    final_path = os.path.join(output_folder, "final.json")
    if re_generate_answer:
        print("Start to re-generate answers for single questions and final multi-hop questions")
        if os.path.exists(final_path) and not continue_gen:
            os.remove(final_path)

        kwargs = {
            "input_path": high_quality_data_path,
            "subquestion_prompt": "extra_files/answer_gen.txt",
            "final_prompt": "extra_files/final_answer_gen.txt",
        }
        answer_task = GenAnswer(final_path, **kwargs)
        answer_task.run()
    else:
        items = utility.read_json(high_quality_data_path)
        for item in items:
            item["answer"] = None
            del item["entries"]
            del item["category"]
        utility.save_json(items, final_path)


if __name__ == "__main__":
    typer.run(main)