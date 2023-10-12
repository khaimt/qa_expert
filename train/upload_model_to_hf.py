from huggingface_hub import HfApi
import typer

api = HfApi()


def upload_model(model_folder: str, repo_id: str):
    api.upload_folder(
        folder_path=model_folder,
        repo_id=repo_id,
        repo_type="model",
    )


if __name__ == "__main__":
    typer.run(upload_model)
