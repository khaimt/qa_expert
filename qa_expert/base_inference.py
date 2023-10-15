from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Callable, Tuple
from qa_expert.prompt_utils import get_prompt_from_messages
from qa_expert.prompt_utils import Message, Role, FunctionCall
import re
import json


def parse_function_info(function_info: str) -> Optional[FunctionCall]:
    if function_info.startswith("retrieve:"):
        arguments_text = function_info[len("retrieve:") :].strip()
        return FunctionCall(name="retrieve", arguments=arguments_text)
    return None


def parse_generated_content(generated_content: str) -> Message:
    # check if we need to call function or not; pattern == <|bof|>function_content<|eof|>
    # <|bof|> retrieve:\n{"query": "where does the joy luck club take place"} <|eof|>
    match = re.search(r"<\|bof\|>(?P<func_info>(.|\n)*)<\|eof\|>", generated_content)
    if match is not None:
        func_info = match.group("func_info").strip()
        function_call = parse_function_info(func_info)
        start = match.start()
        content = generated_content[:start].strip()
        r_content = None if len(content) == 0 else content
        return Message(role=Role.assistant, content=r_content, function_call=function_call)
    return Message(role=Role.assistant, content=generated_content)


class ModelInference(ABC):
    def __init__(self, model_path_or_service: str, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0) -> str:
        raise NotImplementedError

    def generate_answer_for_single_question(
        self, question: str, context: str, temperature: float = 0.001
    ) -> Optional[str]:
        messages = [
            Message(role=Role.user, content=question),
            Message(role=Role.assistant, content=None, function_call=FunctionCall.from_query(question)),
            Message(role=Role.function, content=context),
        ]
        res_message = self.generate_message(messages, temperature=temperature)
        return res_message.content

    def get_retrieval_query(self, question: str, temperature: float = 0) -> Optional[str]:
        messages = [Message(role=Role.user, content=question)]
        assistant_message = self.generate_message(messages, temperature)
        if assistant_message.function_call is not None:  # call function
            argument_str = assistant_message.function_call.arguments
            if argument_str is not None:
                arguments = json.loads(argument_str)
                return arguments["query"]
        return None

    def generate_message(self, messages: List[Message], temperature: float = 0) -> Message:
        prompt = get_prompt_from_messages(messages + [Message(role=Role.assistant)])
        generated_content = self.generate(prompt, temperature)
        return parse_generated_content(generated_content)

    def generate_answer(
        self, question: str, retriever_func: Callable[[str], str], temperature: float = 0, verbose=False
    ) -> Tuple[Optional[str], List[Message]]:
        """This function returns the answer of the question and also the intermediate result: queries, retrieved contexts, thought, answers

        Args:
            question (str): The question to answer
            retriever_func (Callable): The retrieval function, input is: query (str) and output is the relevant context (str)
            temperature (float, optional): Temperature for generation in LLM. Defaults to 0.
            verbose (bool, optional): Defaults to False.

        Returns:
            Tuple[Optional[str], List[Message]]:
                + the answer(str) to the question and
                + list of messages: containing intermediate steps: query, retrieved context, ...
        """
        messages = [Message(role=Role.user, content=question)]
        if verbose:
            print(f"User: {question}")
        while True:
            mess = self.generate_message(messages, temperature)
            messages.append(mess)
            if mess.function_call is not None and mess.function_call.arguments is not None:
                arguments = json.loads(mess.function_call.arguments)
                query = arguments["query"]
                if verbose:
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
                return mess.content, messages
