from gen_data import gen_task, utility
import typer


def generate_sub_category(
    category_path: str, save_path: str, prompt_path: str = typer.Option("gen_data/prompts/sub_category_gen.txt")
):
    task = gen_task.GenSubCategory(save_path, **{"prompt": prompt_path, "category_path": category_path})
    task.run()


if __name__ == "__main__":
    typer.run(generate_sub_category)
