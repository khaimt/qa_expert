from transformers import LlamaTokenizer, AutoModelForCausalLM, GenerationConfig
import utility
import json
import typer
import torch
from sentence_transformers import SentenceTransformer, util
from qa_expert.prompt_utils import SpecialToken, get_prompt_from_messages, Message, Role
import numpy as np
from qa_expert.hf_inference import HFInference
from qa_expert.base_inference import ModelInference
import requests
import shutil
import os


def create_paragraph(title, sens):
    return ". ".join(sens + [title])


def download_file(url: str) -> str:
    local_filename = url.split("/")[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)  # type: ignore

    return local_filename


def evaluate_hotpot_qa(
    model_path: str = typer.Option(default="khaimaitien/qa_expert"),
    retriever_path: str = typer.Option(default="intfloat/e5-base-v2"),
    hotpot_qa_dev_path: str = typer.Option(default=""),
    inference_type: str = typer.Option(default="hf"),
):
    if len(hotpot_qa_dev_path) == 0:
        hotpot_qa_dev_path = "hotpot_dev_distractor_v1.json"
        if not os.path.exists((hotpot_qa_dev_path)):
            print("hotpot_dev_distractor_v1.json doesn't exist, start downloading now !")
            hotpot_qa_dev_path = download_file("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json")
            print("finish downloading hotpot_dev_distractor_v1.json ")
    assert inference_type in ["hf", "vllm"]
    if inference_type == "hf":
        model_inference: ModelInference = HFInference(model_path)
    else:
        from qa_expert.vllm_inference import VllmInference

        model_inference = VllmInference(model_path)
    retriever = SentenceTransformer(retriever_path)
    examples = utility.read_json(hotpot_qa_dev_path)
    print("number of items: ", len(examples))
    for example in examples:
        print("-----------------------------------------------------")
        question = example["question"]
        print("question: ", question)
        answer = example["answer"]
        context = example["context"]
        paragraphs = [create_paragraph(p[0], p[1]) for p in context]
        para_vectors = retriever.encode(paragraphs, normalize_embeddings=True)
        num_paragraphs = 2

        def retrieve(query: str):
            query_vec = retriever.encode([query], normalize_embeddings=True)
            scores = util.cos_sim(query_vec, para_vectors)[0].tolist()
            s_indices = np.argsort(scores).tolist()
            s_indices.reverse()
            contexts = [paragraphs[index] for index in s_indices[:num_paragraphs]]
            return " ".join(contexts)

        pred_answer = model_inference.generate_answer(question, retrieve, verbose=True)
        print(f"pred_answer: {pred_answer};")
        print(f"correct answer: {answer}")


if __name__ == "__main__":
    typer.run(evaluate_hotpot_qa)
