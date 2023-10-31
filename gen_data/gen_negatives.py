import typer
from gen_data.gen_task import GenNegativeParagraph


def main(
    input_path: str = typer.Option(""),
    save_path: str = typer.Option(""),
    gen_num: int = typer.Option(1000),
    paragraph_prompt: str = typer.Option("gen_data/prompts/gen_paragraph.txt"),
    new_entity_prompt: str = typer.Option("gen_data/prompts/gen_new_entity.txt"),
):
    assert len(input_path) > 0
    assert len(save_path) > 0
    kwargs = {
        "input_path": input_path,
        "gen_num": gen_num,
        "paragraph_prompt": paragraph_prompt,
        "new_entity_prompt": new_entity_prompt,
    }
    task = GenNegativeParagraph(save_path, **kwargs)
    task.run()


if __name__ == "__main__":
    typer.run(main)
