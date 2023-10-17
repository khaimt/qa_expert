from typing import Union, List, Optional, Dict
import argparse
import uuid
from fastapi import FastAPI
from pydantic import BaseModel, Field
import time
from qa_expert.prompt_utils import Message
from qa_expert.hf_inference import HFInference
from qa_expert.base_inference import ModelInference
import os
from qa_expert import get_inference_model, InferenceType


class ChatInput(BaseModel):
    messages: List[Message]
    functions: Optional[List[Dict]] = None
    temperature: float = 0.9
    stream: bool = False


class Choice(BaseModel):
    message: Message
    finish_reason: str = "stop"
    index: int = 0


class ResponseOpenAI(BaseModel):
    id: str
    object: str = "chat.completion"
    created: float = Field(default_factory=time.time)
    choices: List[Choice]


class PromptInput(BaseModel):
    prompt: str
    temperature: float = 0


class SingleQAInput(BaseModel):
    question: str
    context: str
    temperature: float = 0


app = FastAPI()
MODEL_PATH = os.environ.get("MODEL_PATH", "khaimaitien/qa-expert-7B-V1.0")
INFERENCE_TYPE = os.environ.get("INFERENCE_TYPE", "hf").lower()
TOKENIZER_PATH = os.environ.get("TOKENIZER", "")  # GGUF needs this 
print(f"start model, inference_type: {INFERENCE_TYPE}, model_path: {MODEL_PATH}, tokenizer_path: {TOKENIZER_PATH}")
assert INFERENCE_TYPE in [item.value for item in InferenceType]
model_inference = get_inference_model(InferenceType(INFERENCE_TYPE), MODEL_PATH, TOKENIZER_PATH)


@app.post("/v1/chat/completions", response_model=ResponseOpenAI)
async def chat_endpoint(chat_input: ChatInput):
    request_id = str(uuid.uuid4())
    message = model_inference.generate_message(chat_input.messages)
    return ResponseOpenAI(id=request_id, choices=[Choice(message=message)])


@app.post("/v1/prompt/completions")
async def generate(prompt_input: PromptInput):
    result = model_inference.generate(prompt_input.prompt, prompt_input.temperature)
    return {"output": result}


@app.post("/v1/single_qa")
async def single_qa(inputs: SingleQAInput):
    result = model_inference.generate_answer_for_single_question(inputs.question, inputs.context, inputs.temperature)
    return {"answer": result}
