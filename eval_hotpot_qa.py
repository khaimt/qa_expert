from transformers import LlamaTokenizer, AutoModelForCausalLM, GenerationConfig
import utility
import json
import typer
from sentence_transformers import SentenceTransformer, util
from qa_expert.prompt_utils import SpecialToken, get_prompt_from_messages, Message, Role
import numpy as np
from qa_expert.hf_inference import HFInference
from qa_expert.base_inference import ModelInference
import requests
import shutil
import os
import datetime
import string
import re


def create_paragraph(title, sens):
    return ". ".join(sens + [title])


def download_file(url: str) -> str:
    local_filename = url.split("/")[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)  # type: ignore

    return local_filename


def normalize_text(s: str) -> str:
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, truth: str) -> float:
    """Compute f1-score based on the individual words in prediction and truth

    Args:
        prediction (str): _description_
        truth (str): _description_

    Returns:
        float: _description_
    """
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def compute_containing_acc(prediction: str, truth: str) -> float:
    """if prediction (complete answer) contains truth (span answer) --> 1 else 0"""
    if normalize_text(truth) in normalize_text(prediction):
        return 1
    return 0


def evaluate_hotpot_qa(
    model_path: str = typer.Option(default="khaimaitien/qa-expert-7B-V1.0"),
    retriever_path: str = typer.Option(default="intfloat/e5-base-v2"),
    hotpot_qa_dev_path: str = typer.Option(default=""),
    inference_type: str = typer.Option(default="hf"),
    save_path: str = typer.Option(default=""),
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
    records = []
    t1 = datetime.datetime.now()
    acc_time = 0.0
    for index, example in enumerate(examples):
        question = example["question"]
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

        pred_answer, messages = model_inference.generate_answer(question, retrieve, verbose=False)
        pred_answer = str(pred_answer)
        t2 = datetime.datetime.now()
        acc_time += (t2 - t1).total_seconds()
        avg_time = acc_time / (index + 1)
        remaining_time = (len(examples) - index - 1) * avg_time
        record = {
            "question": question,
            "span_answer": answer,
            "messages": [mess.model_dump(exclude_none=True) for mess in messages],
            "pred_answer": pred_answer,
        }
        records.append(record)
        f1 = compute_f1(pred_answer, answer)
        containing_acc = compute_containing_acc(pred_answer, answer)
        print(
            f"{index + 1} / {len(examples)}, avg_time: {avg_time}, remaining time: {remaining_time}, F1={f1}, containing_acc: {containing_acc}"
        )
        if len(save_path) > 0:
            utility.save_json(records, save_path)


if __name__ == "__main__":
    typer.run(evaluate_hotpot_qa)