import os
from huggingface_hub import snapshot_download


def download_and_prepare(repo: str):
    # Project root directory
    root_dir = os.getcwd()
    print(f"Project root: {root_dir}")

    # Download directories
    model_dir = root_dir
    dataset_dir = os.path.join(root_dir, "Data/datasets")

    ignore_patterns = ["*.md", ".gitattributes"]

    print(f"Downloading model '{repo}'...")
    model_path = snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=model_dir,
        ignore_patterns=ignore_patterns,
    )
    print(f"Model downloaded to: {model_path}")

    print(f"Downloading dataset '{repo}'...")
    dataset_path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=dataset_dir,
        ignore_patterns=ignore_patterns,
    )
    print(f"Dataset downloaded to: {dataset_path}")

    print("✅ Done.")


if __name__ == "__main__":
    download_and_prepare("soizhiwen/TimeWak")
