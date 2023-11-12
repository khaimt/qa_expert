from transformers import LlamaTokenizer, AutoModelForCausalLM, GenerationConfig
from gen_data import utility
import json
import typer
from sentence_transformers import SentenceTransformer, util
from qa_expert.prompt_utils import SpecialToken, get_prompt_from_messages, Message, Role
import numpy as np
from qa_expert import get_inference_model, ModelInference, InferenceType
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


def compute_recall(prediction: str, truth: str) -> float:
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

    # prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return rec


def compute_containing_acc(prediction: str, truth: str) -> float:
    """if prediction (complete answer) contains truth (span answer) --> 1 else 0"""
    if normalize_text(truth) in normalize_text(prediction):
        return 1
    return 0


def evaluate_hotpot_qa(
    model_path: str = typer.Option(default="khaimaitien/qa-expert-7B-V1.0"),
    hotpot_qa_dev_path: str = typer.Option(default="extra_data/hotpot_dev_distractor_v1_random_500.json"),
    inference_type: str = typer.Option(default="hf"),
    save_path: str = typer.Option(default=""),
):
    """This function is used to run evaluation on hotpot_qa dataset

    Args:
        model_path (str, optional): model to evaluate. Default="khaimaitien/qa-expert-7B-V1.0"
        hotpot_qa_dev_path (str, optional): hotpot_qa file to eval. Default="eval_data/hotpot_dev_distractor_v1_random_500.json".
        inference_type (str, optional): type of inference, you can use Vllm to reduce the evaluation time . Default="hf"
        save_path (str, optional): where to save the inference result, if empty, inference result is not saved. Default=""

    Returns:
        _type_: _description_
    """
    model_inference: ModelInference = get_inference_model(InferenceType(inference_type), model_path)
    retriever = SentenceTransformer("intfloat/e5-base-v2")

    examples = utility.read_json(hotpot_qa_dev_path)
    print("number of items: ", len(examples))

    records = []
    t1 = datetime.datetime.now()
    acc_time = 0.0
    avg_recall_list, avg_acc_list, is_multi_hop_acc_list = [], [], []

    for index, example in enumerate(examples):
        question = example["question"]
        answer = example["answer"]
        context = example["context"]

        paragraphs = [create_paragraph(p[0], p[1]) for p in context]
        prefix_paragraphs = [f"passage: {p}" for p in paragraphs]  # intfloat/e5-base-v2 requires to add passages:
        para_vectors = retriever.encode(prefix_paragraphs, normalize_embeddings=True)
        num_paragraphs = 3

        def retrieve(query: str):
            # intfloat/e5-base-v2 requires to add query:
            query_vec = retriever.encode([f"query: {query}"], normalize_embeddings=True)
            scores = util.cos_sim(query_vec, para_vectors)[0].tolist()
            s_indices = np.argsort(scores).tolist()
            s_indices.reverse()
            contexts = [paragraphs[index] for index in s_indices[:num_paragraphs]]
            return " ".join(contexts)

        try:
            pred_answer, messages = model_inference.generate_answer(
                question, retrieve, verbose=False, temperature=0.00001
            )
        except Exception as e:
            pred_answer, messages = "", []
            print(f"exception at this question: {question}: {str(e)}")
        pred_answer = str(pred_answer)

        t2 = datetime.datetime.now()
        acc_time = (t2 - t1).total_seconds()
        avg_time = acc_time / (index + 1)
        remaining_time = (len(examples) - index - 1) * avg_time
        record = {
            "question": question,
            "span_answer": answer,
            "messages": [mess.json(exclude_none=True) for mess in messages],
            "pred_answer": pred_answer,
        }
        if len(messages) > 4:
            is_multi_hop_acc_list.append(1)
        else:
            is_multi_hop_acc_list.append(0)
        records.append(record)
        recall = compute_recall(pred_answer, answer)
        avg_recall_list.append(recall)

        containing_acc = compute_containing_acc(pred_answer, answer)
        record["containing"] = containing_acc
        avg_acc_list.append(containing_acc)

        avg_is_multi_hop = sum(is_multi_hop_acc_list) / len(is_multi_hop_acc_list)
        avg_recall = sum(avg_recall_list) / len(avg_recall_list)
        avg_acc = sum(avg_acc_list) / len(avg_acc_list)
        print(
            (
                f"{index + 1} / {len(examples)}, avg_time: {avg_time}, remaining time: {remaining_time},"
                f" Recall={avg_recall}, containing_acc: {avg_acc}, avg_is_multi_hop: {avg_is_multi_hop}"
            )
        )
        if len(save_path) > 0:
            utility.save_json(records, save_path)


if __name__ == "__main__":
    typer.run(evaluate_hotpot_qa)
