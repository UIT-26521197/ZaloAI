"""Download the pretrained checkpoints used by the LifeMate baseline."""

import argparse
from pathlib import Path


CHECKPOINTS = (
    "vinai/PhoWhisper-medium",
    "Qwen/Qwen3-4B-Instruct-2507",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="Destination directory (defaults to models/ beside this script)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.exit(1, "Install huggingface_hub with: python -m pip install huggingface_hub\n")

    for repo_id in CHECKPOINTS:
        model_name = repo_id.split("/", 1)[1]
        destination = args.models_dir.expanduser().resolve() / model_name
        print(f"Downloading {repo_id} to {destination}", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=destination)
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
