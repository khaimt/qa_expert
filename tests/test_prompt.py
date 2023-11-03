from gen_data import utility
from qa_expert import prompt_utils
from qa_expert.prompt_utils import Message, Role, SpecialToken, get_additional_tokens
import unittest
import os
from transformers import LlamaTokenizerFast, LlamaTokenizer
from typing import List
import re


def extract_unmasked_chunks(labels: List[int]) -> List[List[int]]:
    """This function is used to extract unmasked chunks of integer
    For example, labels = [-100, -100, 1, 2, 3, -100, -100, 4, 5] --> chunks = [[1,2,3], [4,5]]
    Args:
        labels (List[int]): list of integer containing token_id and -100

    Returns:
        List[List[int]]: list of chunk, for example: [[1,2,3], [4,5]]
    """
    chunks = []
    chunk = []
    for token_id in labels:
        if token_id != -100:
            chunk.append(token_id)
        else:
            if len(chunk) > 0:
                chunks.append(chunk)
                chunk = []
    if len(chunk) > 0:
        chunks.append(chunk)
    return chunks


class PrompTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PrompTest, self).__init__(*args, **kwargs)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        test_case_file = os.path.join(current_folder, "test_cases.json")
        self.test_cases = utility.read_json(test_case_file)

    def test_end_with_eot(self):
        for case in self.test_cases:
            messages = prompt_utils.convert_multi_qa_format_to_messages(case)
            for mess in messages:
                prompt = prompt_utils.get_prompt_of_message(mess).strip()
                self.assertTrue(prompt.endswith(SpecialToken.eot))
        infer_mess = Message(role=Role.assistant)
        prompt = prompt_utils.get_prompt_of_message(infer_mess).strip()
        self.assertFalse(prompt.endswith(SpecialToken.eot))

    def test_tokenizer(self):
        print("**************test slow tokenizer**********")
        self.run_test_with_tokenizer(False)
        print("**************test fast tokenizer**********")
        self.run_test_with_tokenizer(True)

    def run_test_with_tokenizer(self, use_fast: bool):
        tokenizer_class = LlamaTokenizerFast if use_fast else LlamaTokenizer
        tokenizer = tokenizer_class.from_pretrained("mistralai/Mistral-7B-v0.1", legacy=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens": get_additional_tokens()})
        for case in self.test_cases:
            messages = prompt_utils.convert_multi_qa_format_to_messages(case)
            input_dic = prompt_utils.preprare_training_inputs(
                messages, tokenizer, padding=True, max_length=1024, verbose=True
            )
            self.assertEqual(len(input_dic["input_ids"]), len(input_dic["labels"]))
            self.assertEqual(len(input_dic["labels"]), len(input_dic["attention_mask"]))

            # check if only prompt of assistants are included in computing loss
            assistant_messages = [item for item in messages if item.role == Role.assistant]
            labels = input_dic["labels"]
            chunks = extract_unmasked_chunks(labels)  # note that mask here means: -100
            self.assertEqual(len(chunks), len(assistant_messages))
            for chunk, message in zip(chunks, assistant_messages):
                prefix_asisstant = "assistant:"
                prompt = prompt_utils.get_prompt_of_message(message)
                text_for_pred = prompt[len(prefix_asisstant) :]
                unmasked_text = tokenizer.decode(chunk)
                self.assertEqual(re.sub("\s", "", unmasked_text), re.sub("\s", "", text_for_pred))


if __name__ == "__main__":
    unittest.main()
