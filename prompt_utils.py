from enum import Enum
from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Union 
import json 


SYSTEM_MESSAGE = """This is a conversation between user and an artificial intelligence assistant. User will ask question and the assistant will use the function: `retrieve` to retrieve the most relevant context to answer the question. For complex question, assistant might need to use the function `retrieve` multiple times.
`retrieve`: this function is useful for retrieving the most relevant information given a query. The only parameter of this function is: query(str): the query for retrieving relevant content.
"""

class SpecialToken(str, Enum):
    bof = "<|bof|>"
    eof = "<|eof|>"
    eot = "<|eot|>"


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    function = "function"
    system = "system"


class FunctionCall(BaseModel):
    name: str = "retrieve"
    arguments: Optional[str] = None
    

class Message(BaseModel):
    role: Role
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    

def get_prompt_of_message(message: Message) -> str:
    """This function is used to create the prompt string of the message

    Args:
        message (Message): The message to create the prompt

    Returns:
        str: string
    """
    # at first we handle the case where message = {"role": "assistant", "content": None, "function_call": None} 
    # This will be used at inference where we append a new message that content and function_call are both None
    if (message.role == Role.assistant) and (message.function_call is None) and (message.content is None):  
        return f"assistant:"
    prompt_str = ""
    if message.role == Role.system:
        prompt_str = f"system: {message.content}"
    if message.role == Role.user:
        prompt_str = f"user: {message.content}"
    if message.role == Role.function:
        prompt_str = f"retrieved context: {message.content}"
    if message.role == Role.assistant:
        if message.function_call is None:
            prompt_str = f"assistant: {message.content}"
        elif message.content is None:  # function call only
            #  <|bof>retrieve:\n{"query": "who is Donald Trump"}<|eof|>
            prompt_str = f"assistant: {SpecialToken.bof}{message.function_call.name}:\n{message.function_call.arguments}{SpecialToken.eof}"
        else:  # function call and content
            #  Content here\n<|bof>retrieve:\n{"query": "who is Donald Trump"}<|eof|>
            prompt_str = f"assistant: {message.content}\n{SpecialToken.bof}{message.function_call.name}:\n{message.function_call.arguments}{SpecialToken.eof}"
    return prompt_str + SpecialToken.eot + "\n"
        

def get_prompt_from_messages(messages: List[Message]) -> str:
    """get the final prompt from list of messages

    Args:
        messages (List[Message]): list of messages

    Returns:
        str: final_prompt
    """
    final_prompt = ""
    for message in [Message(role="system", content=SYSTEM_MESSAGE)] + messages:
        final_prompt += get_prompt_of_message(message)
    return final_prompt


def get_id_to_sp_token(tokenizer: Any) -> Dict[int, SpecialToken]:
    """return a dictionary mapping from token_id --> SpecialToken

    Args:
        tokenizer (Any): tokenizer

    Returns:
        Dict[int, SpecialToken]: the mapping
    """
    result = {}
    for token in SpecialToken:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 2:
            assert token_ids[0] == 29871  # This is the issue of Llamatokenizer
        else:
            assert len(token_ids) == 1
        tok_id = token_ids[-1]
        result[tok_id] = token


def get_assistant_prefix_tokens(tokenizer: Any) -> List[int]:
    """get token_ids of "\nassistant:" in the prompt. Note that directly use: tokenizer.encode("\nassistant") is not accurate

    Args:
        tokenizer (Any): tokenizer

    Returns:
        List[int]: list of token_ids
    """
    text = f"{SpecialToken.eot.value}\nassistant:"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if token_ids[0] == 29871:
        token_ids = token_ids[1: ]
    return token_ids


def convert_multi_qa_format_to_messages(qa_item: Dict) -> List[Message]:
    """Convert from qa data points to list of messages from OpenAI

    Args:
        qa_item (Dict): _description_

    Returns:
        _type_: _description_
    """
    question = qa_item["question"]
    messages = []
    messages.append({"role": Role.user, "content": question})
    if len(qa_item["sub_questions"]) > 1:
        pre_answer = None
        for sub in qa_item["sub_questions"]:
            args = {"query": sub["question"]}
            messages.append({"role": Role.assistant, "content": pre_answer, "function_call": {"name": "retrieve", "arguments": json.dumps(args, ensure_ascii=False)}})
            messages.append({"role": Role.function, "content": sub["paragraph"]})
            pre_answer = sub["long_answer"].strip()
        messages.append({"role": Role.assistant, "content": pre_answer + "\nSummary:" + qa_item["final_answer"]})
    else:
        args = {"query": question}
        messages.append({"role": Role.assistant, "content": None, "function_call": {"name": "retrieve", "arguments": json.dumps(args, ensure_ascii=False)}})
        messages.append({"role": Role.function, "content": qa_item["sub_questions"][0]["paragraph"]})
        messages.append({"role": Role.assistant, "content": qa_item["final_answer"]})
    return [Message(**mess) for mess in messages]


def preprare_training_inputs(messages: List[Message], tokenizer: Any, padding: Union[str, bool] = "max_length", max_length: Optional[int] = None, verbose: bool = False):
    final_prompt = get_prompt_from_messages(messages)
    if verbose:
        print("final_prompt:\n", final_prompt)
    max_length = max_length if max_length is not None else tokenizer.model_max_length
    input_dic = tokenizer(final_prompt, padding=padding, max_length=max_length, truncation=True)
    input_ids = input_dic["input_ids"]
    
    labels = [-100 for _ in range(len(input_ids))]
    start = 0
    assistant_prefix = get_assistant_prefix_tokens(tokenizer)
    eot_token_id = tokenizer.encode(SpecialToken.eot, add_special_tokens=False)[-1]
    if verbose:
        print("assistant_prefix: ", assistant_prefix)
        print("eot token: ", eot_token_id)
    index = 0
    while index < len(input_ids):
        if index + len(assistant_prefix) > len(input_ids):
            break
        if input_ids[index: index + len(assistant_prefix)] == assistant_prefix:
            eot_index = None
            for i in range(index + len(assistant_prefix), len(input_ids)):
                if input_ids[i] == eot_token_id:
                    eot_index = i
                    break
            end_index = eot_index if eot_index is not None else len(input_ids) - 1
            for i in range(index + len(assistant_prefix), end_index + 1):
                labels[i] = input_ids[i]
            if verbose:
                    chunk = labels[index + len(assistant_prefix): end_index + 1]
                    print("----------------------------")
                    print("+++ chunk assistant to compute loss: ", tokenizer.decode(chunk))
                    print("chunk tokens: ", chunk)
            index = end_index + 1
        else:
            index += 1

    input_dic["labels"] = labels
    return input_dic


def test():
    test_cases = [
       {
        "question": "How many people visit the place Urtica dioica is found every year?",
        "answer": "over 120 million",
        "sub_questions": [
            {
                "question": "Where are Urtica dioica found?",
                "answer": "the Alps",
                "paragraph": "Alps. The extreme and stressful climatic conditions give way to the growth of plant species with secondary metabolites important for medicinal purposes. Origanum vulgare, Prunella vulgaris, Solanum nigrum and Urtica dioica are some of the more useful medicinal species found in the Alps.",
                "long_answer": "Urtica dioica is found in the Alps."
            },
            {
                "question": "How many people visit the the Alps every year?",
                "answer": "over 120 million",
                "paragraph": "Alps. At present the Alps are one of the more popular tourist destinations in the world with many resorts such Oberstdorf, in Bavaria, Saalbach in Austria, Davos in Switzerland, Chamonix in France, and Cortina d'Ampezzo in Italy recording more than a million annual visitors. With over 120 million visitors a year tourism is integral to the Alpine economy with much it coming from winter sports although summer visitors are an important component of the tourism industry.",
                "long_answer": "Over 120 million people visit the Alps every year."
            }
        ],
        "final_answer": "Urtica dioica is found in the Alps, which is a popular tourist destination with over 120 million visitors every year. \nFinal answer: Over 120 million people visit the place where Urtica dioica is found every year.",
        "src": "musique.json"
    }
    ]
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf", legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens([e.value for e in SpecialToken])
    for case in test_cases:
        print("-------------------")
        messages = convert_multi_qa_format_to_messages(case)
        input_dic = preprare_training_inputs(messages, tokenizer, padding=True, max_length=1024, verbose=True)
        print(input_dic)


if __name__ == "__main__":
    test()
    