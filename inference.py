from transformers import AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch 
from prompt_utils import SpecialToken, get_prompt_from_messages, Message, Role

model_path = "merged_models/mistral"
model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto",
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
        )
print("model: ", model)
tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)
print("tokenizer: ", tokenizer)
messages = [Message(role=Role.user, content="When did humans begin living in the city where Joy Luck Club takes place?")]
prompt = get_prompt_from_messages(messages)
print("final prompt: ")
print(prompt)
print("-------------------")

eos_token_id = tokenizer.encode(SpecialToken.eot)[-1]
print("eot token: ", eos_token_id)
gen_config = GenerationConfig(**{"max_new_tokens": 128, "temperature": 0.01, "eos_token_id": eos_token_id })
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
print("input_ids: ", input_ids)
output = model.generate(input_ids, gen_config)
output_ids = output[0].tolist()
print("output_ids: ", output_ids)
print("text: ", tokenizer.decode(output_ids))