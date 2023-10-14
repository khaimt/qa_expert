from vllm import LLM, SamplingParams
from qa_expert.prompt_utils import SpecialToken
from qa_expert.base_inference import ModelInference


class VllmInference(ModelInference):
    def __init__(self, model_path_or_service: str, **kwargs) -> None:
        self.llm = LLM(model=model_path_or_service)
        tokenizer = self.llm.get_tokenizer()
        self.eos_token_id = tokenizer.encode(SpecialToken.eot)[-1]

    def generate(self, prompt: str, temperature: float = 0.001) -> str:
        sampling_params = SamplingParams(temperature=temperature, max_tokens=1024, stop_token_ids=[self.eos_token_id])
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text
