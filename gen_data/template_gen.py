from typing import Dict, List, Any, Optional, Callable, Tuple
import re
import openai
import time
from abc import ABC, abstractmethod
import json
import sys
import requests
import datetime
from openai import OpenAI
import traceback


openai_client = OpenAI()


class CustomizedPromptGen(ABC):
    @abstractmethod
    def get_prompt(self, input_dic: Dict) -> str:
        raise NotImplementedError


class WizardLMPromptGen(CustomizedPromptGen):
    def get_prompt(self, input_dic: Dict) -> str:
        prompt = input_dic["prompt_input"]
        output_prefix = input_dic["output_prefix"]
        new_prompt = (
            f"A chat between a curious user and an artificial intelligence assistant. "
            f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: {prompt} ASSISTANT:{output_prefix}"
        )
        return new_prompt


class OpenAIPromptGen(CustomizedPromptGen):
    def get_prompt(self, input_dic: Dict) -> str:
        input_prompt = input_dic["prompt_input"]
        output_prefix = input_dic["output_prefix"]
        result = input_prompt + "\nPlease generate now:"
        if len(output_prefix) > 0:
            result += f"\n{output_prefix}"
        return result


class OpenOrcaPrompGen(CustomizedPromptGen):
    def get_prompt(self, input_dic: Dict) -> str:
        input_prompt = input_dic["prompt_input"]
        output_prefix = input_dic["output_prefix"]

        index = input_prompt.find("\n")
        sys_prompt = input_prompt[:index].strip()
        user_query = input_prompt[index + 1 :].strip()
        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"

        def get_text_role(role, content, add_suffix=True):
            result = prefix + f"{role}\n" + content
            if add_suffix:
                result += suffix
            return result

        # sys_format = prefix + "system\n" + sys_prompt + suffix
        # user_format = prefix + "user\n" + user_query + suffix
        # assistant_format = prefix + "assistant\n"
        sys_format = get_text_role("system", sys_prompt)
        user_format = get_text_role("user", user_query)
        assistant_format = get_text_role("assistant", output_prefix, False)
        return sys_format + user_format + assistant_format


PROMPT_GEN_DIC = {"wizardlm": WizardLMPromptGen(), "openai": OpenAIPromptGen(), "open_orca": OpenOrcaPrompGen()}


def get_final_prompt(input_dic: Dict) -> str:
    prompt_type = input_dic["prompt_type"]
    prompt_gen = PROMPT_GEN_DIC[prompt_type]
    return prompt_gen.get_prompt(input_dic)


def parse_fields_from_template(prompt_template: str) -> List[str]:
    fields = []
    for match in re.finditer(r"(\n|^)\+(?P<field>.+?):", prompt_template):
        field = match.group("field").strip()
        fields.append(field)
    return fields


def parse_output_prefix(prompt_template: str) -> Tuple[str, str]:
    sep_text = "please generate now:"
    index = prompt_template.lower().find(sep_text)
    instruction = prompt_template[:index].strip()
    output_prefix = prompt_template[index + len(sep_text) :].strip()
    return instruction, output_prefix


def get_response_from_chat_model(input_dic: Dict) -> Dict[str, Any]:
    final_prompt = get_final_prompt(input_dic)
    response = openai.ChatCompletion.create(
        model=input_dic["llm"],
        messages=[{"role": "user", "content": final_prompt}],
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
        "response": answer["message"]["content"],
    }


def get_response_from_completion_model(input_dic: Dict) -> Dict[str, Any]:
    final_prompt = get_final_prompt(input_dic)
    # print("-----------------")
    # print("final_prompt: ")
    # print(final_prompt)
    for i in range(5):
        try:
            response = openai_client.completions.create(
                model=input_dic["llm"],
                prompt=final_prompt,
                temperature=input_dic.get("temperature", 0),
                max_tokens=1024,
                timeout=5,
            )
            response = response.dict()
            # response = openai.Completion.create(
            #     model=input_dic["llm"],
            #     prompt=final_prompt,
            #     temperature=input_dic.get("temperature", 0),
            #     max_tokens=1024,
            #     timeout=5,
            # )
            text_response = response["choices"][0]["text"]
            # print("text response: ", text_response)
            finish_reason = response["choices"][0]["finish_reason"]
            total_tokens = response["usage"]["total_tokens"]
            return {"finish_reason": finish_reason, "total_tokens": total_tokens, "response": text_response}
        except Exception as e:
            traceback.print_exc()
            print(f"exception from OpenAI side at: {i}, error: {str(e)} ")
            time.sleep(3)
    sys.exit(1)


def get_response_from_service_model(input_dic: Dict) -> Dict[str, Any]:
    final_prompt = get_final_prompt(input_dic)
    # print("++++++++++++")
    # print(final_prompt)
    temperature = input_dic.get("temperature", 0.001)
    if temperature == 0:
        temperature = 0.001
    data = {"prompt": final_prompt, "temperature": temperature, "max_new_token": 512}
    print(json.dumps(data, ensure_ascii=False, indent=4))
    endpoint = input_dic["llm"]
    # t1 = datetime.datetime.now()
    response = requests.post(endpoint, json=data)
    # t2 = datetime.datetime.now()
    # print("exe_time: ", (t2 - t1).total_seconds())
    output = json.loads(response.text)
    return {"finish_reason": "stop", "total_tokens": 0, "response": output["result"]}


def format_prompt(slot_dic: Optional[Dict[str, str]], prompt: str) -> str:
    if slot_dic is None:
        return prompt
    for slot_key, slot_value in slot_dic.items():
        prompt = prompt.replace("{" + slot_key + "}", slot_value)
    return prompt


def try_loop(function: Callable, params: Dict, max_count=5) -> Any:
    for i in range(max_count):
        try:
            result = function(params)
            return result
        except Exception as e:
            print(f"try to call function: {function.__name__} at: {i} exception: {str(e)}")
            time.sleep(2)
    return None


class TemplateGen(ABC):
    def __init__(self, prompt_template: str, llm: str = "gpt-3.5-turbo-instruct", **kwargs) -> None:
        self.prompt_template = prompt_template
        self.llm = llm
        self.prompt_input, self.prompt_output_prefix = parse_output_prefix(self.prompt_template)
        self.prompt_type = kwargs.get("prompt_type", None)
        if self.prompt_type is None:
            self.prompt_type = "openai"

    @abstractmethod
    def parse_output(self, llm_output: str) -> Any:
        raise NotImplementedError

    def call_llm(self, prompt_input: str, prompt_output: str, temperature: float) -> Dict:
        gen_inputs = {
            "prompt_input": prompt_input,
            "output_prefix": prompt_output,
            "temperature": temperature,
            "llm": self.llm,
            "prompt_type": self.prompt_type,
        }
        if self.llm.startswith("http://"):
            gen_func = get_response_from_service_model
        elif "instruct" in self.llm:
            gen_func = get_response_from_completion_model
        else:
            gen_func = get_response_from_chat_model
        response = try_loop(gen_func, gen_inputs)
        if response is not None:
            response["response"] = prompt_output + response["response"]
        return response

    def generate(
        self, slot_dic: Optional[Dict[str, str]] = None, temperature: float = 0, max_times=5
    ) -> Optional[Dict]:
        prompt_input = format_prompt(slot_dic, self.prompt_input)
        prompt_output_prefix = format_prompt(slot_dic, self.prompt_output_prefix)

        for i in range(max_times):
            if i > 0 and temperature == 0:  # if not the first time and temperature == 0
                temperature = 0.7  # if failed at temperature = 0, try bigger temperature
            if i > 0:
                print(f"try with temperature: {temperature}")
            response = self.call_llm(prompt_input, prompt_output_prefix, temperature)
            if response is None:
                return None
            output = self.parse_output(response["response"])
            if output is not None:
                break
            else:
                print(f"cannot parse the llm_output for: {slot_dic}, try again at: {i}")
        response["result"] = output
        return response


class FixedTemplateGen(TemplateGen):
    """This class is used for template with the following teplate:
    Instruction
    ...
    + field1: xxx
    + field2: xxx
    ...
    + fieldn: xxx
    ------------
    Please generate now:

    """

    def __init__(self, prompt_template: str, llm: str = "gpt-3.5-turbo-instruct", **kwargs) -> None:
        super().__init__(prompt_template, llm, **kwargs)
        assert "please generate now:" in prompt_template.lower()
        self.fields = parse_fields_from_template(self.prompt_input)
        print("all fields: ", self.fields)

    def get_output_prefix(self, prompt):
        return parse_output_prefix(prompt)

    def parse_output(self, llm_output: str) -> Optional[Dict]:
        # print("---llmouput-----")
        # print(llm_output)
        # first replace: ------------- if existed
        llm_output = re.sub("[-]+", "", llm_output)
        dic = {}
        fields = self.fields
        for i in range(len(fields)):
            cur_field = fields[i]
            if i < len(fields) - 1:
                next_field = fields[i + 1]
                pattern = rf"\+(\s+)?{cur_field}:(?P<content>(.|\n)+)\+(\s+)?{next_field}"
            else:  # last field
                pattern = rf"\+(\s+)?{cur_field}:(?P<content>(.|\n)+)"
            match = re.search(pattern, llm_output)
            if match is not None:
                content = match.group("content").strip()
                dic[cur_field.strip()] = content
            else:
                print(f"patterns: {pattern}")
                print(f"cannot parse: {cur_field}")
                print(llm_output)
                print("--------------------")
                return None
        return dic


class PatternTemplateGen(TemplateGen):
    def __init__(self, prompt_template: str, llm: str = "gpt-3.5-turbo-instruct", **kwargs) -> None:
        """this will parse the llm_output into lines and extract information from each line

        Args:
            prompt_template (str): _description_
            llm (str, optional): _description_. Defaults to "gpt-3.5-turbo-instruct".
            kwargs: containing pattern is a regex
        """
        super().__init__(prompt_template, llm, **kwargs)
        self.pattern = kwargs["pattern"]

    def parse_output(self, llm_output: str) -> List[Dict]:
        # print("llm_putput: ")
        # print(llm_output)
        result = []
        for match in re.finditer(self.pattern, llm_output):
            if match is not None:
                result.append(match.groupdict())
        return result
