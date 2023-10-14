# QA Expert: LLM for Multi-hop Question Answering

QA Expert is a Language Model (LLM) specifically fine-tuned for the task of Question Answering, with a strong emphasis on addressing Multi-hop Question Answering scenarios.

<p align="center">
  <img src="assets/1-shot-example.jpg", width="450", height="375">
</p>
<p align="center">
An example of 1-shot question (single question) and how QA Expert LLM handle multi-hop Q&A</p>

<p align="center">
  <img src="assets/hotpot_qa_bridge.jpg", width="450", height="375">
  <img src="assets/hotpot_qa_compare.jpg", width="450", height="375">
</p>
<p align="center">
Examples of 2-shot questions and how QA Expert LLM handle multi-hop Q&A. The left is an example of bridging entitiy and the right is an example of comparing entities</p>

Multi-hop Question Answering is a task that necessitates the retrieval of multiple contexts, followed by their integration to deduce the answer to the question. 

QA Expert will analyze the question and the retrieval result to decide whether additional retrieval is required or if the answer should be generated. So the output of QA expert is either <b>calling retrieval function with a query </b> or <b>generating the final answer</b>, this is a little bit similar to the output of OpenAI function calling: call a function or generate text response.

## News
- [2023/10/12] We released our finetuned model: <b>khaimaitien/qa-expert-7B-V1.0</b>based on [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) + our training data: [khaimaitien/qa-expert-multi-hop-qa-V1.0](https://huggingface.co/datasets/khaimaitien/qa-expert-multi-hop-qa-V1.0)

## Content
1. [Usage](#usage)
2. [Training data](#training-data)
3. [Fine-tuning](#fine-tuning)
4. [Citation](#citation)

## Usage
### Model Download

| Size | Hugging Face Repo |
| ---  | --- |
| 7B | [khaimaitien/qa-expert-7B-V1.0](https://huggingface.co/khaimaitien/qa-expert-7B-V1.0) | 

### Inference 
As our LLM will 

### Server Usage


## Fine-tuning Data
Please take a look at the README of folder [train/README.md](train/README.md)

## Fine-tuning
Please take a look at the README of folder [train/README.md](train/README.md)
## To-Do

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