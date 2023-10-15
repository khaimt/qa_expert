from typing import List
import requests
from qa_expert.base_inference import ModelInference
import openai
import json

from qa_expert.prompt_utils import Message


class ServerInference(ModelInference):
    def __init__(self, model_path_or_service: str, **kwargs) -> None:
        self.api_base = model_path_or_service
        self.api_key = "qa_expert"

    def generate(self, prompt: str, temperature: float = 0.001) -> str:
        data = {"prompt": prompt, "temperature": temperature}
        response = requests.post(f"{self.api_base}/prompt/completions", json=data, timeout=600)
        return json.loads(response.text)["output"]
