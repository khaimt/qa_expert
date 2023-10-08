from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Callable
from prompt_utils import SpecialToken, get_prompt_from_messages
from prompt_utils import Message, Role, FunctionCall
from transformers import GenerationConfig, LlamaTokenizer, AutoModelForCausalLM, GenerationConfig
import re
import json 
import torch


def parse_function_info(function_info: str) -> Optional[FunctionCall]:
    if function_info.startswith("retrieve:"):
        arguments_text = function_info[len("retrieve:"): ].strip()
        return FunctionCall(name="retrieve", arguments=arguments_text)
    return None


def parse_generated_content(generated_content: str) -> Message:
    # check if we need to call function or not; pattern == <|bof|>function_content<|eof|>
    # <|bof|> retrieve:\n{"query": "where does the joy luck club take place"} <|eof|>
    match = re.search("<\|bof\|>(?P<func_info>(.|\n)*)<\|eof\|>", generated_content)
    if match is not None:
        func_info = match.group("func_info").strip()
        function_call = parse_function_info(func_info)
        start = match.start()
        content = generated_content[: start].strip()
        if len(content) == 0:
            content = None
        return Message(role=Role.assistant, content=content, function_call=function_call)
    return Message(role=Role.assistant, content=generated_content)


class ModelInference(ABC):
    def __init__(self, model_path_or_service: str, **kwargs) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.001) -> str:
        raise NotImplementedError 
    
    def generate_message(self, messages: List[Message], temperature=0.001) -> Message:
        prompt = get_prompt_from_messages(messages + [Message(role=Role.assistant)]) 
        generated_content = self.generate(prompt, temperature)
        # print("-------prompt to gen: ")
        # print(prompt)
        # print("-------")
        # print("generated content: ")
        # print(generated_content)
        # print("----------------")
        
        return parse_generated_content(generated_content)
    
    def generate_answer(self, question: str, retriever_func: Callable, temperature=0.001, verbose=False) -> str:
        messages = [Message(role=Role.user, content=question)]
        while True:
            mess = self.generate_message(messages, temperature)
            messages.append(mess)
            if mess.function_call is not None:
                arguments = json.loads(mess.function_call.arguments)
                query = arguments["query"]
                if verbose:
                    print("-----------------")
                    if mess.content is not None:
                        print(f"+ Thought: {mess.content}")
                    print(f"+ Retrieve information: query={query}")
                context = retriever_func(arguments["query"])
                if verbose:
                    print(f"+ retrieved context: {context}")
                messages.append(Message(role=Role.function, content=context))
            else:
                if verbose:
                    print(f"Reponse: {mess.content}")
                return mess.content


class HFInference(ModelInference):
    def __init__(self, model_path_or_service: str, **kwargs) -> None:
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path_or_service, legacy=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_or_service, device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.encode(SpecialToken.eot)[-1]
    
    def generate(self, prompt: str, temperature: float = 0.001) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        gen_config = GenerationConfig(**{"max_new_tokens": 512, "do_sample": True, "temperature": temperature, "eos_token_id": self.eos_token_id })
        output = self.model.generate(input_ids, gen_config)
        
        output_ids = output[0].tolist()
        generated_ids = output_ids[input_ids.shape[1]: ]
        if generated_ids[-1] == self.eos_token_id:  # remove end_of_turn if existed
            generated_ids = generated_ids[: -1]
        generated_content = self.tokenizer.decode(generated_ids)
        return generated_content