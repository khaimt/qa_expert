# QA Expert: LLM for Multi-hop Question Answering

QA Expert is a Language Model (LLM) specifically fine-tuned for the task of Question Answering, with a strong emphasis on addressing Multi-hop Question Answering scenarios.

<p align="center">
  <img src="assets/1-shot-example.jpg", width="375", height="325">
</p>
<p align="center">
An example of 1-shot question (single question) and how QA Expert LLM handle multi-hop Q&A</p>

<p align="center">
  <img src="assets/hotpot_qa_bridge.jpg", width="375", height="325">
  <img src="assets/hotpot_qa_compare.jpg", width="375", height="325">
</p>
<p align="center">
Examples of 2-shot questions and how QA Expert LLM handle multi-hop Q&A. The left is an example of bridging entitiy and the right is an example of comparing entities</p>

Multi-hop Question Answering is a task that necessitates the retrieval of multiple contexts, followed by their integration to deduce the answer to the question. 

QA Expert will analyze the question and the retrieval result to decide whether additional retrieval is required or if the answer should be generated. So the output of QA expert is either <b>calling retrieval function with a query </b> or <b>generating the final answer</b>, this is a little bit similar to the output of OpenAI function calling: call a function or generate text response.

## News
- [2023/10/12] We released our finetuned model: <b>khaimaitien/qa-expert-7B-V1.0</b>based on [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) + our training data: [khaimaitien/qa-expert-multi-hop-qa-V1.0](https://huggingface.co/datasets/khaimaitien/qa-expert-multi-hop-qa-V1.0)

## Content
1. [Usage](#usage)
2. [Fine-tuning Data](#fine-tuning-data)
3. [Fine-tuning](#fine-tuning)
4. [Citation](#citation)

## Usage
### Model Download

Below are the finetuned models. Basically from evaluation we found that <b>the 7B model - [khaimaitien/qa-expert-7B-V1.0](https://huggingface.co/khaimaitien/qa-expert-7B-V1.0) is better than 13B model - [khaimaitien/qa-expert-llama2-13B-V1.0](https://huggingface.co/khaimaitien/qa-expert-llama2-13B-V1.0)</b>, this might be due to the superiority of the base model [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) over [Llama-2-13b-hf](https://huggingface.co/NousResearch/Llama-2-13b-hf)

So you should use: [khaimaitien/qa-expert-7B-V1.0](https://huggingface.co/khaimaitien/qa-expert-7B-V1.0) for better performance, latency and memory.
| Size | Hugging Face Repo | Base Model |
| ---  | --- | --- |
| 7B | [khaimaitien/qa-expert-7B-V1.0](https://huggingface.co/khaimaitien/qa-expert-7B-V1.0) | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) |
|13B| [khaimaitien/qa-expert-llama2-13B-V1.0](https://huggingface.co/khaimaitien/qa-expert-llama2-13B-V1.0)|[NousResearch/Llama-2-13b-hf](https://huggingface.co/NousResearch/Llama-2-13b-hf)|

You can also find GGUF versions:

| Size | Hugging Face Repo |
| ---  | --- |
| 7B | [khaimaitien/qa-expert-7B-V1.0-GGUF](https://huggingface.co/khaimaitien/qa-expert-7B-V1.0-GGUF) | 
|13B| [khaimaitien/qa-expert-llama2-13B-V1.0-GGUF](https://huggingface.co/khaimaitien/qa-expert-llama2-13B-V1.0-GGUF)|

### Inference 
Curently we support 4 types of inference:
+ Using [Huggingface Transformers](https://github.com/huggingface/transformers)
+ Using [Vllm](https://github.com/vllm-project/vllm)
+ Using [llama.cpp](https://github.com/ggerganov/llama.cpp)

First please install the requirements:
```
torch==2.1.0
sentence_transformers==2.2.2
transformers==4.34.0
pydantic==1.10
flash-attn==2.3.2
llama-index==0.8.45.post1
```

The example for using transformers HuggingFace:

```python 

from qa_expert import get_inference_model, InferenceType

def retrieve(query: str) -> str:
    # You need to implement this retrieval function, input is a query and output is a string
    # This can be treated as the function to call in function calling of OpenAI
    return context

model_inference = get_inference_model(InferenceType.hf, "khaimaitien/qa-expert-7B-V1.0")
answer, messages = model_inference.generate_answer(question, retriever_func)
```
For Vllm, you need to install Vllm (```pip install vllm```) and change the InferenceType to vllm:
```python
model_inference = get_inference_model(InferenceType.vllm, "khaimaitien/qa-expert-7B-V1.0")
```
For LLama.cpp, you need to install: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and transformers version >= 4.34.0
```python
# Please download gguf files from here: https://huggingface.co/khaimaitien/qa-expert-7B-V1.0-GGUF/tree/main
# by: git clone https://huggingface.co/khaimaitien/qa-expert-7B-V1.0-GGUF
# There are 2 versions: q4_0: 4bit quantization and q8_0: 8bit quantization 
# Note that here we need to pass an additional parameter for the folder of tokenizer.
# model_inference = get_inference_model(InferenceType.llama_cpp, path_to_gguf, path_to_tokenizer)
model_inference = get_inference_model(InferenceType.llama_cpp, "qa-expert-7B-V1.0-GGUF/qa-expert-7B-V1.0.q4_0.gguf", "qa-expert-7B-V1.0-GGUF")
# If you only download the gguf file instead the whole repo: https://huggingface.co/khaimaitien/qa-expert-7B-V1.0-GGUF/resolve/main/qa-expert-7B-V1.0.q4_0.gguf
# you can use:
model_inference = get_inference_model(InferenceType.llama_cpp, "qa-expert-7B-V1.0.q4_0.gguf", "khaimaitien/qa-expert-7B-V1.0-GGUF")
```

### Server Usage


## Fine-tuning Data
Please take a look at the section **Training Data** of [train/README.md](train/README.md#training-data)

## Fine-tuning
Please take a look at [train/README.md](train/README.md)
## Evaluation
Please take a look at the Section **Evaluation** of [train/README.md](train/README.md#evaluation)
## Citation
If you feel my work is helpful, please kindly cite as:
```bibtex
@Misc{qa-expert,
      title={QA Expert: LLM for Multi-hop Question Answering},
      author={Khai Mai},
      howpublished={\url{https://github.com/khaimt/qa_expert}},
      year={2023},
}
```