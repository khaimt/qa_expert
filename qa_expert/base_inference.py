from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Callable, Tuple
from qa_expert.prompt_utils import get_prompt_from_messages
from qa_expert.prompt_utils import Message, Role, FunctionCall
import re
import json
from colorama import Fore, Back, Style


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


def parse_final_answer(generated_text: str) -> Tuple[Optional[str], Optional[str], str]:
    # First check if this is the final answer of multi-hop question
    # if multi-top, it will follows the template: answer_to_last_single_question\nSummary:xxx\nAnswer:yyy
    # only multi-hop question contains: Summary
    # print(Fore.CYAN + f"generated_text: {generated_text}")
    match = re.search(r"\nSummary:(?P<summary>(.|\n)*)\nAnswer:(?P<final_answer>(.|\n)*)", generated_text)
    if match is not None:
        print("hit match")
        start = match.start()
        last_question_answer = generated_text[:start].strip()
        summary = match.group("summary").strip()
        final_answer = match.group("final_answer").strip()
        return last_question_answer, summary, final_answer
    else:  # this is the answer to single question, the template is: xxx\Answer:yyy where xxx is Thought (like chain-of-thought)
        return generated_text, None, generated_text


class ModelInference(ABC):
    def __init__(self, model_path_or_service: str, *args, **kwargs) -> None:
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
            print(Fore.RED + f"User: {question}")
        while True:
            mess = self.generate_message(messages, temperature)
            messages.append(mess)
            if mess.function_call is not None and mess.function_call.arguments is not None:
                arguments = json.loads(mess.function_call.arguments)
                query = arguments["query"]
                if verbose:
                    if mess.content is not None:
                        print(Fore.GREEN + f"+ Thought: {mess.content}")
                    print(Fore.BLUE + f"+ Retrieve information: query={query}")
                context = retriever_func(arguments["query"])
                context = context.replace("\n", " ")
                if verbose:
                    print(Fore.YELLOW + f"+ retrieved context: {context}")
                messages.append(Message(role=Role.function, content=context))
            else:
                if verbose:
                    last_question_answer, summary, final_answer = parse_final_answer(str(mess.content))
                    if summary is None:
                        print(Fore.MAGENTA + f"{final_answer}" + Fore.RESET)
                    else:
                        print(Fore.GREEN + f"+ Thought: {last_question_answer}")
                        print(Fore.MAGENTA + f"+ Summary: {summary}\n\nAnswer: {final_answer}" + Fore.RESET)
                return mess.content, messages
