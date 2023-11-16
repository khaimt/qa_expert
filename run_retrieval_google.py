from qa_expert import get_inference_model, InferenceType
import os
import requests
import json
from typing import Dict
import typer


def get_snippet_content(item: Dict) -> str:
    title = item.get("title", "")
    snippet = item.get("snippet", "")
    content = title + " " + snippet
    content = content.strip()
    return content


def google_search(api_key: str, query: str):
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    payload = json.dumps({"q": query})

    response = requests.request("POST", "https://google.serper.dev/search", headers=headers, data=payload)
    result = json.loads(response.text)
    retrieved_items = []
    if "answerBox" in result:
        content = get_snippet_content(result["answerBox"])
        if len(content) > 0:
            retrieved_items.append(content)
    if "organic" in result:
        for item in result["organic"][:3]:
            content = get_snippet_content(item)
            if len(content) > 0:
                retrieved_items.append(content)
    return " ".join(retrieved_items)


def main(
    api_key: str = typer.Option(default="e9b35305c3b0a79189b7c2dc4c37adbc587d1e65"),
    model_path: str = typer.Option(default="khaimaitien/qa-expert-7B-V1.0"),
    inference_type: str = typer.Option(default="hf"),
):
    def retrieve(query):
        return google_search(api_key, query)

    model = get_inference_model(InferenceType(inference_type), model_path)
    while True:
        user_question = input("User: ")
        final_answer, messages = model.generate_answer(user_question, retriever_func=retrieve, verbose=True)


if __name__ == "__main__":
    typer.run(main)
