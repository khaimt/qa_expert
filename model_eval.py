from transformers import LlamaTokenizer, AutoModelForCausalLM, GenerationConfig
import utility
import json 
import typer
import torch
from sentence_transformers import SentenceTransformer, util
from prompt_utils import SpecialToken, get_prompt_from_messages, Message, Role
import numpy as np
from inference import HFInference


def create_paragraph(title, sens):
    return ". ".join(sens + [title])


def evaluate_hotpot_qa(test_path: str = "datasets/evaluation/hotpot_dev_fullwiki_v1.json", model_path: str = "khaimaitien/qa_expert", retriever_path: str = "intfloat/e5-base-v2"):
    model_inference = HFInference(model_path)
    retriever =  SentenceTransformer(retriever_path)
    examples = utility.read_json(test_path)
    print("number of items: ", len(examples))
    for example in examples:
        print("-----------------------------------------------------")
        question = example["question"]
        print("question: ", question)
        answer = example["answer"]
        context = example["context"]
        paragraphs = [create_paragraph(p[0], p[1]) for p in context]
        para_vectors = retriever.encode(paragraphs, normalize_embeddings=True)
        
        def retrieve(query: str):
            query_vec = retriever.encode([query], normalize_embeddings=True)
            scores = util.cos_sim(query_vec, para_vectors)[0].tolist()
            m_index = np.argmax(scores)
            return paragraphs[m_index]
        
        pred_answer = model_inference.generate_answer(question, retrieve, verbose=True)
        print(f"pred_answer: {pred_answer}; answer: {answer}")


if __name__ == "__main__":
    typer.run(evaluate_hotpot_qa)