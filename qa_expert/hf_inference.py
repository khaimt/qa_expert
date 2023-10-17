from qa_expert.prompt_utils import SpecialToken
from transformers import GenerationConfig, LlamaTokenizer, AutoModelForCausalLM
from qa_expert.base_inference import ModelInference
import torch


class HFInference(ModelInference):
    def __init__(self, model_path_or_service: str, *args, **kwargs) -> None:
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path_or_service, legacy=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_or_service,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.encode(SpecialToken.eot)[-1]

    def generate(self, prompt: str, temperature: float = 0.001) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        gen_config = GenerationConfig(
            **{"max_new_tokens": 512, "do_sample": True, "temperature": temperature, "eos_token_id": self.eos_token_id}
        )
        if temperature == 0:
            gen_config["temperature"] = 0.0001
        output = self.model.generate(input_ids, gen_config)

        output_ids = output[0].tolist()
        generated_ids = output_ids[input_ids.shape[1] :]
        if generated_ids[-1] == self.eos_token_id:  # remove end_of_turn if existed
            generated_ids = generated_ids[:-1]
        generated_content = self.tokenizer.decode(generated_ids)
        return generated_content
