import openai
import os
import sys
from gen_data import utility
import datetime
import time
import re
from abc import ABC, abstractmethod
from typing import Any, Union, Dict, List, Tuple, Callable, Optional
import json
import typer
from gen_data.template_gen import FixedTemplateGen, PatternTemplateGen
import copy
import random


def merge_thought_answer(thought: str, answer: str, summary: str = "") -> str:
    if len(summary) == 0:
        return thought + "\nAnswer: " + answer
    return f"Summary: {summary} {thought}" + "\nAnswer: " + answer


class GenTask(ABC):
    def __init__(self, save_path: str, **kwargs) -> None:
        self.save_path = save_path
        self.result = []
        self.handle_count = 0
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                self.result = json.loads(f.read())
        self.llm = kwargs.get("llm", "gpt-3.5-turbo-instruct")

    @abstractmethod
    def count_number_of_remaining_items(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def handle_item(self, item: Any) -> Optional[Dict]:
        raise NotImplementedError

    @abstractmethod
    def get_items_for_handling(self):
        raise NotImplementedError

    def run(self):
        total_count = self.count_number_of_remaining_items()
        acc_tokens = 0
        handled_count = 0
        t1 = datetime.datetime.now()
        print(f"number of REMAINING items to handle: {total_count}")
        for item in self.get_items_for_handling():
            item_res = self.handle_item(item)
            if item_res is None:
                continue
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
            self.handle_count += 1


class GenAnswer(GenTask):
    def __init__(self, save_path: str, **kwargs) -> None:
        super().__init__(save_path, **kwargs)
        input_path = kwargs["input_path"]
        self.input_items = utility.read_json(input_path)
        self.sub_question_prompt = utility.read_text(kwargs["subquestion_prompt"])
        self.sub_question_gen = FixedTemplateGen(
            self.sub_question_prompt, llm=self.llm, prompt_type=kwargs.get("prompt_type", None)
        )
        self.final_question_prompt = utility.read_text(kwargs["final_prompt"])
        self.final_question_gen = FixedTemplateGen(
            self.final_question_prompt, llm=self.llm, prompt_type=kwargs.get("prompt_type", None)
        )
        self.use_avail_answers = kwargs.get("use_avail_answers", True)

    def get_items_for_handling(self):
        # start generating
        for item in self.input_items[len(self.result) :]:
            yield item

    def count_number_of_remaining_items(self) -> int:
        return len(self.input_items) - len(self.result)

    def process_output_dic(self, output_dic: Dict) -> Dict:
        result = copy.deepcopy(output_dic)
        response = output_dic["result"]
        thought = response["Thought"]
        summary = response.get("Summary", "")
        answer = response["Answer"]
        answer_str = merge_thought_answer(thought, answer, summary)
        result["result"] = answer_str
        return result

    def handle_item(self, item: Any) -> Dict:
        subs = item["sub_questions"]
        total_token_count = 0
        # first generate answer for sub-question
        for sub in subs:
            question = sub["question"]
            context = sub["paragraph"]
            if sub.get("long_answer", None) is not None:  # do not overwrite if exists
                if self.use_avail_answers:
                    continue
            output_dic = self.sub_question_gen.generate({"question": question, "context": context}, temperature=0)
            if output_dic is None:
                print("cannot generate answer to this question: ", question)
                return {"total_tokens": total_token_count, "result": item}
            output_dic = self.process_output_dic(output_dic)
            response, finish_reason, tok_count = (
                output_dic["result"],
                output_dic["finish_reason"],
                output_dic["total_tokens"],
            )
            total_token_count += tok_count
            if finish_reason == "stop":
                sub["long_answer"] = response
            else:
                print("finish_reason is not stop: ", finish_reason)
                sys.exit(1)
        ## get the final answer here
        # if single question, no need to generate final_answer,
        # if multihop but only 1 single quesion --> unanswerable still need final-answer
        if item["multihop"]:
            final_answer = item.get("final_answer", None)
            if final_answer is not None and self.use_avail_answers:  # no need to compute
                return {"total_tokens": total_token_count, "result": item}
            else:
                question = item["question"]
                subs = item["sub_questions"]
                facts = [sub["long_answer"] for sub in subs]
                fact_str = "\n".join([f"+ {fact}" for fact in facts])
                output_dic = self.final_question_gen.generate({"question": question, "facts": fact_str}, temperature=0)
                if output_dic is None:
                    print("cannot generate final answer to this question: ", question)
                    return {"total_tokens": total_token_count, "result": item}
                output_dic = self.process_output_dic(output_dic)
                finish_reason, tok_count, response = (
                    output_dic["finish_reason"],
                    output_dic["total_tokens"],
                    output_dic["result"],
                )
                total_token_count += tok_count
                if finish_reason == "stop":
                    item["final_answer"] = response
                else:
                    print("finsih_reason is not stop at final_question: ", finish_reason)
                    sys.exit(1)
                item["meta_info"]["llm"] = self.llm
        else:
            item["final_answer"] = subs[0]["long_answer"]
        return {"total_tokens": total_token_count, "result": item}


class GenDataCategory(GenTask):
    def __init__(self, save_path: str, **kwargs) -> None:
        super().__init__(save_path, **kwargs)
        self.prompt_template = utility.read_text(kwargs["prompt"])
        self.template_gen = FixedTemplateGen(
            self.prompt_template, self.llm, prompt_type=kwargs.get("prompt_type", None)
        )
        self.num_items_per_category = kwargs.get("num_items_per_category", 30)
        self.categories = utility.read_category_lines(kwargs["category_path"])
        # we have a big number of category so we don't need to set temperature = 1
        self.temperature = kwargs.get("temperature", 0)
        print("temperature used to generate data: ", self.temperature)
        print("list of category: ", self.categories)

        # update the current number of items in each category
        self.current_category_count = {cat: 0 for cat in self.categories}
        for item in self.result:
            self.current_category_count[item["meta_info"]["category"]] += 1

        # compute the remaining number for each category
        self.category_count = {}  # mapping from category --> number of remaining items
        for category in self.current_category_count:
            self.category_count[category] = self.num_items_per_category - self.current_category_count[category]
            if self.category_count[category] <= 0:  # no need to call --> delete
                del self.category_count[category]
        self.total_calls = 0

    def check_valid_result(self, result: Dict) -> bool:
        return True

    def count_number_of_remaining_items(self) -> int:
        return self.num_items_per_category * len(self.categories) - len(self.result)

    def get_items_for_handling(self):
        while len(self.category_count) > 0:
            for category in list(self.category_count.keys()):
                if self.category_count[category] > 0:  # if still need to generate for this category
                    self.category_count[category] -= 1
                    if self.category_count[category] == 0:  # delete if this category is done
                        del self.category_count[category]
                    yield category

    def handle_item(self, item: Any) -> Optional[Dict]:
        # here item is a category; we found that to make it more diverse we should add: catetory in
        # for example: "company in", "award in" --> to ask LLM to generate sub category
        q_types = ["wh question", "yes/no question"]
        slot_dic = {
            "category": f"{item}",
            "question_type": random.choice(q_types),
            "popularity_1": str(random.randint(1, 5)),
            "popularity_2": str(random.randint(1, 5)),
            "popularity": str(random.randint(1, 5)),
        }
        previous_token_count = 0
        for i in range(5):
            # if failed, --> increase temperature
            temperature = self.temperature
            if i > 0:
                if self.temperature == 0:  # if not the first time and temperature = 0
                    temperature = 1

            output_dic = self.template_gen.generate(slot_dic, temperature=temperature)
            self.total_calls += 1  # every time we generate, we add 1
            print("avg number of calls per category: ", self.total_calls / (self.handle_count + 1))
            if output_dic is not None:
                result = self.process_output_dic(output_dic)
                result["result"]["meta_info"]["category"] = item
                result["result"]["meta_info"]["slot_dic"] = slot_dic
                if self.check_valid_result(result):
                    result["total_tokens"] += previous_token_count
                    return result
                else:
                    print("+++++++++ result is not valid: ")
                    print(json.dumps(result["result"], ensure_ascii=False, indent=4))
                previous_token_count += result["total_tokens"]
            print(f"retry at: {i} for slot_dic: {slot_dic}")
        return None

    def process_output_dic(self, output_dic: Dict) -> Dict:
        parsed_result = output_dic["result"]
        thought = parsed_result.get("Final Thought", None)
        final_answer = parsed_result.get("Final answer", None)
        summary = parsed_result.get("Summary", None)
        if thought is not None:
            answer_str = merge_thought_answer(thought, final_answer, summary)
        else:
            answer_str = None

        answer1 = None
        if "Answer 1" in parsed_result:
            answer1 = merge_thought_answer(parsed_result["Thought 1"], parsed_result["Answer 1"])

        answer2 = None
        if "Answer 2" in parsed_result:
            answer2 = merge_thought_answer(parsed_result["Thought 2"], parsed_result["Answer 2"])

        record = {
            "sub_questions": [
                {
                    "question": parsed_result["Question 1"],
                    "long_answer": answer1,
                    "paragraph": parsed_result["Knowledge 1"],
                },
                {
                    "question": parsed_result["Question 2"],
                    "long_answer": answer2,
                    "paragraph": parsed_result["Knowledge 2"],
                },
            ],
            "answer": answer_str,
            "final_answer": answer_str,
            # "meta_info": {"entity_1": parsed_result["Entity 1"], "entity_2": parsed_result["Entity 2"], "src": "gen_qa"},
            "multihop": True,
        }
        result = copy.deepcopy(output_dic)
        result["result"] = record
        return result


class GenEntityComparison(GenDataCategory):
    def check_valid_result(self, result: Dict) -> bool:
        item = result["result"]
        entity1 = item["meta_info"]["entity_1"]
        entity2 = item["meta_info"]["entity_2"]
        # Chatgpt might generate: Entity 1: a random entity in this category
        if "entity" in entity1.lower() or "entity" in entity2.lower():
            return False
        question = item["question"]
        sub_question1 = item["sub_questions"][0]["question"].lower()
        paragraph1 = item["sub_questions"][0]["paragraph"]
        sub_question2 = item["sub_questions"][1]["question"].lower()
        paragraph2 = item["sub_questions"][1]["paragraph"]
        checks = []
        checks.append(entity1.lower() in question.lower())  # entity 1 must be in final question
        checks.append(entity2.lower() in question.lower())  # entity 2 must be in final question
        # single question 1 must contain entity 1 and not contain entity 2
        checks.append(entity1.lower() in sub_question1 and entity2.lower() not in sub_question1)
        # single question 2 must contain entity 2 and not contain entity 1
        checks.append(entity2.lower() in sub_question2 and entity1.lower() not in sub_question2)
        checks.append("entity" not in entity1.lower())  # no entity in entity 1
        checks.append("entity" not in entity2.lower())  # no entity in entity 2
        checks.append("entity 1" not in paragraph1)  # no "entity 1" in paragraph 1
        checks.append("entity 2" not in paragraph2)  # no "entity 2" in paragraph 2
        checks.append(question.lower() != sub_question1.lower())  # single question 1 must be not final question
        checks.append(question.lower() != sub_question2.lower())  # single question 2 must be not final question
        checks.append("level of popularity" not in entity1.lower())
        checks.append("level of popularity" not in entity2.lower())
        if all(checks):
            return True
        return False

    def process_output_dic(self, output_dic) -> Dict:
        parsed_result = output_dic["result"]
        result = super().process_output_dic(output_dic)
        result["result"]["meta_info"] = {
            "entity_1": parsed_result["Entity 1"],
            "entity_2": parsed_result["Entity 2"],
            "comparison_attribute": parsed_result["Selected Attribute"],
            "list_of_attributes": parsed_result["List of Attributes"],
            "src": "gen_qa",
            "llm": self.llm,
        }
        result["result"]["question"] = parsed_result["Question for 2 entities"]
        return result


class GenAttributeMerge(GenDataCategory):
    def process_output_dic(self, output_dic) -> Dict:
        parsed_result = output_dic["result"]
        result = super().process_output_dic(output_dic)
        result["result"]["meta_info"] = {
            "attribute_1": parsed_result["Attribute 1"],
            "attribute_2": parsed_result["Attribute 2"],
            "src": "gen_qa",
            "list_of_attributes": parsed_result["List of Attributes"],
            "entity": parsed_result["Entity"],
            "llm": self.llm,
        }
        result["result"]["question"] = parsed_result["Merged Question"]
        return result


class GenSubCategory(GenTask):
    def __init__(self, save_path: str, **kwargs) -> None:
        super().__init__(save_path, **kwargs)
        self.categories = utility.read_category(kwargs["category_path"])
        self.prompt = utility.read_text(kwargs["prompt"])
        self.template_gen = PatternTemplateGen(
            self.prompt,
            self.llm,
            pattern=r"(Sub-category \d+):\s+(?P<subcategory>.*)",
            prompt_type=kwargs.get("prompt_type", None),
        )
        self.cat_dic = {}
        if os.path.exists(save_path):
            self.cat_dic = utility.read_json(save_path)

    def count_number_of_remaining_items(self) -> int:
        return len(self.categories) - len(self.cat_dic)

    def handle_item(self, item: Any) -> Optional[Dict]:
        llm_output = self.template_gen.generate({"category": item}, temperature=0)
        if llm_output is None:
            print(f"cannot generate sub-categories for:{item}")
            sys.exit(1)
        result = llm_output["result"]
        sub_categories = []
        for line_result in result:
            sub_categories.append(line_result["subcategory"])
        llm_output["result"] = sub_categories
        return llm_output

    def get_items_for_handling(self):
        for cat in self.categories:
            if cat not in self.cat_dic:
                yield cat


class GenNegativeParagraph(GenTask):
    def __init__(self, save_path: str, **kwargs) -> None:
        super().__init__(save_path, **kwargs)
        input_path = kwargs["input_path"]
        self.paragraph_gen = PatternTemplateGen(
            utility.read_text(kwargs["paragraph_prompt"]), self.llm, pattern=r"Paragraph:(?P<content>((.)|(\s))+)"
        )
        self.entity_gen = PatternTemplateGen(
            utility.read_text(kwargs["new_entity_prompt"]), self.llm, pattern=r":(?P<entity>(.|\n)+)"
        )
        self.gen_num = kwargs["gen_num"]
        examples = utility.read_json(input_path)
        self.examples = []
        for ex in examples:
            if self.check_valid_example(ex):
                self.examples.append(ex)
        print(f"number of valid examples: {len(self.examples)} / {len(examples)}")

        # each time we will randomly select an item by index to handle
        self.remaining_indices = set([i for i in range(len(self.examples))])
        for item in self.result:
            # remove indices we handled before
            self.remaining_indices.remove(item["meta_info"]["index"])

    def check_valid_example(self, ex):
        """We only choose example that attribute (attribute_1, attribute2 or comparison_attribute) is in the list_of_attributes
        and list_of_attributes contains: comma and doesn't contain: \n
        Returns:
            _type_: _description_
        """
        attr_list_str = ex["meta_info"]["list_of_attributes"]
        if "\n" in attr_list_str:
            return False
        if "," not in attr_list_str:
            return False
        attributes = set([item.strip() for item in attr_list_str.split(",") if len(item.strip()) > 0])
        cur_attrs = []
        for name in ["attribute_1", "attribute_2", "comparison_attribute"]:
            if name in ex["meta_info"]:
                cur_attrs.append(ex["meta_info"][name])
        for attr in cur_attrs:
            if attr not in attributes:
                return False
            attributes.remove(attr)
        if len(attributes) == 0:
            return False
        return True

    def count_number_of_remaining_items(self) -> int:
        return self.gen_num - len(self.result)

    def generate_attribute_paragraph(self, entity, attribute) -> Tuple[str, int]:
        slot_dic = {"entity": entity, "attribute": attribute}
        result = self.paragraph_gen.generate(slot_dic, temperature=1)
        if result is None:
            print(f"failed to generate a paragraph for: {entity}, atribute:{attribute}")
            sys.exit(1)
        return result["result"][0]["content"].strip(), result["total_tokens"]

    def generate_new_entity(self, avail_entities, category) -> Tuple[str, int]:
        # TODO generate a new entity
        slot_dic = {"entities": "; ".join(avail_entities), "category": category}
        result = self.entity_gen.generate(slot_dic, temperature=1)
        if result is None:
            print(f"failed to generate new entity from avail_entities: {avail_entities}, category:{category}")
            sys.exit(1)
        return result["result"][0]["entity"].strip(), result["total_tokens"]

    def handle_item(self, item: Any) -> Optional[Dict]:
        """Given used attributes; we will pick out 2 new attributes from the list;
            also get list of avail entities:
            To generate negative we will take the following possibilities:
            Currently we have: q1=(entity, attr1); q2=(entity2, attr2)
            --> we randomly choose to change: question 1, question 2 or both
            for each chosen question, we will randomly choose:
                + new entity
                + new attr
                + both new entity and attr
        Args:
            item (Any): _description_

        Returns:
            Optional[Dict]: _description_
        """
        meta_info = item["meta_info"]
        cur_attrs = []
        for name in ["attribute_1", "attribute_2", "comparison_attribute"]:
            if name in meta_info:
                cur_attrs.append(meta_info[name])

        attr_list_str = meta_info["list_of_attributes"]
        all_attributes = set([item.strip() for item in attr_list_str.split(",") if len(item.strip()) > 0])

        for attr in cur_attrs:
            all_attributes.remove(attr)
        new_attributes = list(all_attributes)
        random.shuffle(new_attributes)

        avail_entities = []
        for ent in ["entity_1", "entity_2", "entity"]:
            if ent in meta_info:
                avail_entities.append(meta_info[ent])

        # there are 3 types of generating: only q1, only q2 and both q1 and q2
        changed_indices = random.choice([[0], [1], [0, 1]])
        sub_questions = item["sub_questions"]
        meta_info["negatives"] = []
        total_tokens = 0
        for index in changed_indices:
            sub_question = sub_questions[index]
            change_entity = 0
            change_attr = 1
            change_both = 2
            choice = random.choice([change_entity, change_attr, change_both])

            if "entity" in meta_info:  # from attributes
                entity = meta_info["entity"]
            else:  # from comparing 2 entities
                entity = meta_info[f"entity_{index + 1}"]

            if "comparison_attribute" in meta_info:
                attr = meta_info["comparison_attribute"]
            else:
                attr = meta_info[f"attribute_{index + 1}"]

            if choice == change_entity or choice == change_both:
                for i in range(5):
                    entity, token_count = self.generate_new_entity(avail_entities, meta_info["category"])
                    if len(entity) > 0:
                        break
                total_tokens += token_count
                avail_entities.append(entity)  # add new
            if choice == change_attr or choice == change_both:
                attr = random.choice(new_attributes)
                new_attributes.remove(attr)  # remove after choosing
            para, token_count = self.generate_attribute_paragraph(entity, attr)
            total_tokens += token_count

            sub_question["paragraph"] = para
            sub_question["long_answer"] = None
            meta_info["negatives"].append(
                {"change_type": choice, "entity": entity, "attribute": attr, "q_index": index}
            )
        item["final_answer"] = None
        item["answer"] = None
        return {"result": item, "total_tokens": total_tokens}

    def get_items_for_handling(self):
        for _ in range(self.count_number_of_remaining_items()):
            index = random.sample(self.remaining_indices, 1)[0]
            item = self.examples[index]
            item["meta_info"]["index"] = index
            self.remaining_indices.remove(index)
            yield item
