from gen_data.gen_task import GenAnswer
import os
import typer


def main(
    input_path: str,
    output_path: str,
    continue_gen: bool = typer.Option(True, "--no-continue"),
    llm: str = typer.Option(default="gpt-3.5-turbo-instruct"),
    prompt_type: str = typer.Option(default="openai"),
):
    print("Start to re-generate answers for single questions and final multi-hop questions")
    if os.path.exists(output_path) and not continue_gen:
        os.remove(output_path)

    kwargs = {
        "input_path": input_path,
        "subquestion_prompt": "gen_data/prompts/answer_gen.txt",
        "final_prompt": "gen_data/prompts/final_answer_gen.txt",
        "llm": llm,
        "prompt_type": prompt_type,
        "temperature": 0.0001,
    }
    answer_task = GenAnswer(output_path, **kwargs)
    answer_task.run()


if __name__ == "__main__":
    typer.run(main)
