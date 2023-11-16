from qa_expert.prompt_utils import SpecialToken
from llama_cpp import Llama
from qa_expert.base_inference import ModelInference
from transformers import LlamaTokenizer
import os


class LLamaCppInference(ModelInference):
    def __init__(self, model_path_or_service: str, *args, **kwargs) -> None:
        """This is the inference code for using Llama.cpp. For 7B model + 4bit quantization, you only need about 4-5GB in Memory.
        model_path_or_service: one of the following files (download from: https://huggingface.co/khaimaitien/qa-expert-7B-V1.0-GGUF/tree/main)
            + qa-expert-7B-V1.0.q4_0.gguf
            + qa-expert-7B-V1.0.q8_0.gguf
            + qa-expert-7B-V1.0.f16.gguf
        """
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)  # load all layers to GPU
        self.llm = Llama(model_path=model_path_or_service, n_ctx=kwargs.get("n_ctx", 2048), n_gpu_layers=n_gpu_layers)
        # Note that must use tokenizer from HF because we added new tokens
        # tokenizer from Llama is not correct
        self.tokenizer = LlamaTokenizer.from_pretrained("khaimaitien/qa-expert-7B-V1.0-GGUF", legacy=True)
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
