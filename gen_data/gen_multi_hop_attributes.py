from gen_data.gen_task import GenAttributeMerge, GenAnswer
from gen_data.gen_task import utility
import typer
import os


def main(
    num_items_per_category: int = typer.Option(default=1),
    output_folder: str = typer.Option(default="gen_qa"),
    re_generate_answer: bool = typer.Option(False, "--re-generate-answer"),
    category_path: str = typer.Option(default="gen_data/other_files/sub_categories.txt"),
    continue_gen: bool = typer.Option(True, "--no-continue"),
    multi_qa_prompt: str = typer.Option(default="gen_data/prompts/2_attributes_wo_answer.txt"),
    temperature: float = typer.Option(default=0),
    llm: str = typer.Option(default="gpt-3.5-turbo-instruct"),
    prompt_type: str = typer.Option(default="openai"),
):
    """this function is used to generate multi-hop Q&A

    Args:
        num_items_per_category (int, optional): number of generated items for each category. Defaults to typer.Option(default=100).
        output_folder (str, optional): where to save the result. Defaults to typer.Option(default="gen_qa").
        re_generate_answer (bool, optional): If we re-generate the answers to single questions and final answer to the multi-hop question or not.
            if re-generate, we will use the prompt template for generating the answer + temperature=0
        category_path (str, optional): The path to list of categories. Defaults to typer.Option(default="extra_files/categories.txt").
        continue_gen (bool, optional): if we continue to generate from current result or not. Defaults to typer.Option(True, "--no-continue").
    """
    if not os.path.exists(output_folder):
        utility.create_folder(output_folder)
    kwargs = {
        "category_path": category_path,
        "prompt": multi_qa_prompt,
        "num_items_per_category": num_items_per_category,
        "temperature": temperature,
        "llm": llm,
        "prompt_type": prompt_type,
    }
    print("kwargs: ", kwargs)
    multi_hop_qa_path = os.path.join(output_folder, "raw_multi_hop_qa.json")
    if os.path.exists(multi_hop_qa_path) and not continue_gen:
        os.remove(multi_hop_qa_path)
    print("Start to generate multi-hop QA now")
    task = GenAttributeMerge(multi_hop_qa_path, **kwargs)
    task.run()

    final_path = os.path.join(output_folder, "final.json")
    if re_generate_answer:
        print("Start to re-generate answers for single questions and final multi-hop questions")
        if os.path.exists(final_path) and not continue_gen:
            os.remove(final_path)

        kwargs = {
            "input_path": multi_hop_qa_path,
            "subquestion_prompt": "gen_data/prompts/answer_gen.txt",
            "final_prompt": "gen_data/prompts/final_answer_gen.txt",
            "llm": llm,
            "prompt_type": prompt_type,
            "temperature": 0.0001,
        }
        answer_task = GenAnswer(final_path, **kwargs)
        answer_task.run()


if __name__ == "__main__":
    typer.run(main)
