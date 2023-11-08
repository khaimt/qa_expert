from typing import List, Dict, Any, Tuple
import os
import chromadb
from sentence_transformers import SentenceTransformer
import datetime
from qa_expert import get_inference_model, InferenceType
import typer


def build_paragraphs(text: str, para_leng: int = 1200, step_size: int = 300) -> List[str]:
    """Split text into paragraphs

    Args:
        text (str): text to split
        para_leng (int, optional): length of paragraphs in terms of number of characters. Defaults to 1200.
        step_size (int, optional): step-size to build paragraph. Defaults to 300.

    Returns:
        List[str]: _description_
    """
    result = []
    start = 0
    while start < len(text):
        result.append(text[start : start + para_leng])
        if start + para_leng >= len(text):
            break
        start = start + step_size
    return result


def get_paragraphs_from_folder(folder: str) -> Dict[str, Any]:
    data_frames: Dict[str, List] = {"paragraphs": [], "metadatas": [], "ids": []}
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


def get_batch_chunks(size: int, batch_size: int) -> List[Tuple[int, int]]:
    result = []
    for index in range(size // batch_size + 1):
        start = index * batch_size
        end = (index + 1) * batch_size
        if end > size:
            end = size
        if end > start:
            result.append((start, end))
    return result


def main(
    data_folder: str = typer.Option("extra_data/test_data/cities"),
    qa_model: str = typer.Option("khaimaitien/qa-expert-7B-V1.0"),
    inference_type: str = typer.Option("hf"),
    num_paragraphs: int = typer.Option(1),
):
    embed_model = SentenceTransformer("intfloat/e5-base-v2")

    chroma_client = chromadb.PersistentClient("chroma_data")
    collection_name = data_folder.replace("/", "_")
    collection = chroma_client.get_or_create_collection(name=collection_name)
    print("collection_name: ", collection_name)
    collection_count = collection.count()
    print("collection_count: ", collection_count)

    # Collection_count == 0 means we need to index paragraphs
    if collection_count == 0:
        print("index is not found, start to index data now ...")
        data_frames = get_paragraphs_from_folder(data_folder)
        # adding paragraph to choma by mini-batch
        batch_size = 500
        para_num = len(data_frames["paragraphs"])
        print("number of paragraphs: ", para_num)
        batches = get_batch_chunks(para_num, batch_size)
        t1 = datetime.datetime.now()
        for index, (start, end) in enumerate(batches):
            batch_paragraphs = data_frames["paragraphs"][start:end]
            # intfloat/e5-base-v2 requires us to add: "passage: " before paragraph, more detail: https://huggingface.co/intfloat/e5-base-v2
            batch_paragraphs = [f"passage: {p}" for p in batch_paragraphs]
            vectors = embed_model.encode(batch_paragraphs, normalize_embeddings=True)
            collection.add(
                embeddings=vectors.tolist(),
                documents=data_frames["paragraphs"][start:end],
                metadatas=data_frames["metadatas"][start:end],
                ids=data_frames["ids"][start:end],
            )
            t2 = datetime.datetime.now()
            avg_time = (t2 - t1).total_seconds() / (index + 1)
            print(
                f"avg_time: {avg_time}s per {batch_size}, remaining time: {avg_time * (len(batches) - index - 1)} seconds"
            )

    # retrieval function used for retrieve relevant knowledge
    def retrieve(query: str):
        results = collection.query(
            # intfloat/e5-base-v2 requires us to add: "query: " before query, more detail: https://huggingface.co/intfloat/e5-base-v2
            query_embeddings=embed_model.encode(["query: " + query]).tolist(),
            n_results=num_paragraphs,
        )
        paragraphs = results["documents"][0]
        # distances = results["distances"][0]
        # for distance, para in zip(distances, paragraphs):
        #    print(f"+++ distance={distance}, context={para}",)
        return "\n".join(paragraphs)

    model_inference = get_inference_model(InferenceType(inference_type), qa_model)
    while True:
        question = input("USER'S QUESTION: ")
        answer, messages = model_inference.generate_answer(question, retrieve, verbose=True)


if __name__ == "__main__":
    typer.run(main)
