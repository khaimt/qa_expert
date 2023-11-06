from vllm import LLM, SamplingParams
from qa_expert.prompt_utils import SpecialToken
from qa_expert.base_inference import ModelInference
from transformers import LlamaTokenizer


class VllmInference(ModelInference):
    def __init__(self, model_path_or_service: str, *args, **kwargs) -> None:
        self.llm = LLM(model=model_path_or_service)
        tokenizer = LlamaTokenizer.from_pretrained(model_path_or_service, legacy=True)
        self.llm.set_tokenizer(tokenizer)
        self.eos_token_id = tokenizer.encode(SpecialToken.eot)[-1]

    def generate(self, prompt: str, temperature: float = 0.001) -> str:
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=1024, stop_token_ids=[self.eos_token_id], skip_special_tokens=False
        )
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        if generated_text.endswith(SpecialToken.eot):
            generated_text = generated_text[: -len(SpecialToken.eot)].strip()
        return generated_text
