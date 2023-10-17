from qa_expert.prompt_utils import SpecialToken
from llama_cpp import Llama
from qa_expert.base_inference import ModelInference
from transformers import LlamaTokenizer


class LLamaCppInference(ModelInference):
    def __init__(self, model_path_or_service: str, *args, **kwargs) -> None:
        """This is the inference code for using Llama.cpp. For 7B model + 4bit quantization, you only need about 4-5GB in Memory
        Note that you need to pass model_path_or_service=path to gguf file
        and an additional parameter for Huggingface tokenizer, for example: khaimaitien/qa-expert-7B-V1.0-GGUF (this repo containing tokenizer's files)
        This is because I found that when using convert.py to convert from HuggingFace's format to GGUF format, added_tokens are not correctly included.
        This might be handled in the future but currently, still an issue.
        You can take a look at here: https://github.com/ggerganov/llama.cpp/pull/3633
        So the temporary solution is using tokenizer of Huggingface to get token_ids
        Args:
            model_path_or_service (str): path to gguf file
            args[0]: path to tokenizer, either local folder or Huggingface repo
                + added_tokens.jon
                + special_tokens_map.json
                + tokenizer_config.json
                + tokenizer.model
            You can directly download from: https://huggingface.co/khaimaitien/qa-expert-7B-V1.0-GGUF
        For example:
        You can download the file from repo: khaimaitien/qa-expert-7B-V1.0-GGUF by ```git clone khaimaitien/qa-expert-7B-V1.0-GGUF```
        and save to: qa-expert-7B-V1.0-GGUF
        usage:
        model_inference = LLamaCppInference("qa-expert-7B-V1.0-GGUF/qa-expert-7B-V1.0.q4_0.gguf", "qa-expert-7B-V1.0-GGUF")
        """
        self.llm = Llama(model_path=model_path_or_service, n_ctx=kwargs.get("n_ctx", 2048))
        self.tokenizer = LlamaTokenizer.from_pretrained(args[0], legacy=True)
        self.eos_token_id = self.tokenizer.encode(SpecialToken.eot)[-1]

    def generate(self, prompt: str, temperature: float = 0) -> str:
        temperature = temperature if temperature > 0 else 0.0001
        token_ids = self.tokenizer.encode(prompt)
        output_generator = self.llm.generate(
            token_ids, temp=temperature
        )  # (prompt, max_tokens=512, stop=[SpecialToken.eot], temperature=temperature)
        gen_tokens = []
        for token_id in output_generator:
            if token_id == self.eos_token_id:
                break
            gen_tokens.append(token_id)
        output = self.tokenizer.decode(gen_tokens)
        return output
