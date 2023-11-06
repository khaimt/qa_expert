from qa_expert.prompt_utils import SpecialToken
from llama_cpp import Llama
from qa_expert.base_inference import ModelInference
from transformers import LlamaTokenizer
import os


class LLamaCppInference(ModelInference):
    def __init__(self, model_path_or_service: str, *args, **kwargs) -> None:
        """This is the inference code for using Llama.cpp. For 7B model + 4bit quantization, you only need about 4-5GB in Memory
        Note that you need to pass model_path_or_service=folder you download from: https://huggingface.co/khaimaitien/qa-expert-7B-V1.0-GGUF/tree/main
        """
        # one of: q4_0; q8_0; f16 make sure that folder: model_path_or_service contains: qa-expert-7B-V1.0.{data_type}.gguf
        data_type = kwargs.get("data_type", "q4_0")
        gguf_path = os.path.join(model_path_or_service, f"qa-expert-7B-V1.0.{data_type}.gguf")
        print("load model from: ", gguf_path)
        assert os.path.exists(
            gguf_path
        ), f"Cannot find: {gguf_path}, please download folder: https://huggingface.co/khaimaitien/qa-expert-7B-V1.0-GGUF/tree/main"
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)
        print("n_gpu_layers=", n_gpu_layers)
        self.llm = Llama(model_path=gguf_path, n_ctx=kwargs.get("n_ctx", 2048), n_gpu_layers=n_gpu_layers)
        # We use separate tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path_or_service, legacy=True)
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
