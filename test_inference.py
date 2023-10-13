import requests
import shutil


def download_file(url: str) -> str:
    local_filename = url.split("/")[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)  # type: ignore

    return local_filename


download_file("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json")
