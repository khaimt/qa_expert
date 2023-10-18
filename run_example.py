from typing import List, Dict, Any, Tuple
import os
import chromadb
from sentence_transformers import SentenceTransformer
import datetime


def build_paragraphs(text: str, para_leng: int = 1200, step_size: int = 300) -> List[str]:
    result = []
    start = 0
    while start < len(text):
        result.append(text[start : start + para_leng])
        if start + para_leng >= len(text):
            break
        start = start + step_size
    return result


def get_paragraphs_from_folder(folder: str) -> Dict[str, Any]:
    data_frames = {"paragraphs": [], "metadatas": [], "ids": []}
    for name in os.listdir(folder):
        if name.endswith(".txt"):
            with open(os.path.join(folder, name), "r") as f:
                text = f.read().strip()
                paras = build_paragraphs(text)
                for index, p in enumerate(paras):
                    data_frames["paragraphs"].append(p)
                    data_frames["metadatas"].append({"source": name})
                    data_frames["ids"].append(f"{name}_{index}")
    return data_frames


def get_batch_chunk(size: int, batch_size: int) -> List[Tuple[int, int]]:
    result = []
    for index in range(size // batch_size + 1):
        start = index * batch_size 
        end = (index + 1) * batch_size
        if end > size:
            end = size
        if end > start:
            result.append((start, end))
    return result


embed_model = SentenceTransformer("intfloat/e5-base-v2")


chroma_client = chromadb.PersistentClient("choma_data")
collection = chroma_client.get_or_create_collection(name="countries")
collection_count = collection.count()
print("collection_count: ", collection_count)
if collection_count == 0:
    # if not indexed we will index
    data_frames = get_paragraphs_from_folder("extra_data/test")
    # adding paragraph to choma by mini-batch
    batch_size = 100
    para_num = len(data_frames["paragraphs"])
    print("number of paragraphs: ", para_num)
    batches = get_batch_chunk(para_num, batch_size)
    t1 = datetime.datetime.now()
    for index, (start, end) in enumerate(batches):
        vectors = embed_model.encode(data_frames["paragraphs"][start: end], normalize_embeddings=True)
        collection.add(
            embeddings=vectors.tolist(),
            documents=data_frames["paragraphs"][start: end],
            metadatas=data_frames["metadatas"][start: end],
            ids=data_frames["ids"][start: end]
        )
        t2 = datetime.datetime.now()
        avg_time = (t2 - t1).total_seconds() / (index + 1)
        print(f"avg_time: {avg_time} seconds, remaining time: {avg_time * (len(batches) - index - 1)} seconds")

top_paragraphs = 1
def retrieve(query: str):
    results = collection.query(
        query_embeddings=embed_model.encode([query]).tolist(),
        n_results=top_paragraphs
    )
    paragraphs = results["documents"][0]
    distances = results["distances"][0]
    #for distance, para in zip(distances, paragraphs):
    #    print(f"+++ distance={distance}, context={para}",)
    return "\n".join(paragraphs)

from qa_expert import get_inference_model, InferenceType

model_inference = get_inference_model(InferenceType.vllm, "models/qa-expert-7B-V1.0")
while True:
    question = input("USER'S QUESTION: ")
    answer, messages = model_inference.generate_answer(question, retrieve, verbose=True)
    print("FINAL ANSWER: ", answer)
