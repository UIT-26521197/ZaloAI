"""Run the pretrained LifeMate baseline from a ZAI/src/<repo> checkout."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from model_cache import stage_models


REPO = Path(__file__).resolve().parent


def find_project_root(repo=REPO):
    """Locate the ZAI folder containing sibling data/ and model(s)/ folders."""
    for candidate in (repo, *repo.parents):
        if (candidate / "data").is_dir() and any(
            (candidate / name).is_dir() for name in ("models", "model")
        ):
            return candidate
    raise FileNotFoundError(
        f"No parent of {repo} contains data/ and models/ (or model/). "
        "Expected ZAI/src/<repo>, ZAI/data, and ZAI/models."
    )


def command_for(args, root, models=None):
    data = root / "data"
    models = models or (root / "models" if (root / "models").is_dir() else root / "model")
    limit = args.limit
    checkpoint_every = args.checkpoint_every or (1 if limit else 25)
    common = ["--checkpoint-every", str(checkpoint_every)]
    if limit:
        common += ["--limit", str(limit)]
    if args.resume:
        common.append("--resume")
    if args.split == "train":
        return [sys.executable, "-u", str(REPO / "evaluate.py"),
                "--data-root", str(data), "--models-dir", str(models),
                "--runs-dir", str(root / "runs"), *common]
    return [sys.executable, "-u", str(REPO / "inference.py"),
            "--data-root", str(data), "--split", args.split,
            "--llm", str(models / "Qwen3-4B-Instruct-2507"),
            "--asr", str(models / "PhoWhisper-medium"),
            "--output", str(root / "result" / "submission.jsonl"), *common]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        help="ZAI directory; automatically discovered by default")
    parser.add_argument("--split", choices=("train", "public_test", "private_test"),
                        default="train")
    parser.add_argument("--limit", type=int, help="Number of samples; omitted means the full split")
    parser.add_argument("--checkpoint-every", type=int,
                        help="Save partial output every N samples (1 for a small run, else 25)")
    parser.add_argument("--resume", action="store_true", help="Continue a matching run")
    parser.add_argument("--cache-dir", type=Path,
                        help="Copy model files once to this local directory before loading")
    parser.add_argument("--no-cache-models", action="store_true",
                        help="Load model files directly from the mounted Drive")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.checkpoint_every is not None and args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.cache_dir and args.no_cache_models:
        parser.error("--cache-dir and --no-cache-models cannot be combined")
    root = args.project_root.resolve() if args.project_root else find_project_root()
    if not (root / "data").is_dir():
        parser.error(f"Missing {root / 'data'}")
    if not any((root / name).is_dir() for name in ("models", "model")):
        parser.error(f"Missing {root / 'models'} (or model/)")
    models = root / "models" if (root / "models").is_dir() else root / "model"
    if not args.no_cache_models:
        cache_dir = args.cache_dir
        if cache_dir is None and (
            os.environ.get("COLAB_RELEASE_TAG") or Path("/content/drive").is_dir()
        ):
            cache_dir = Path("/content/zai-model-cache")
        if cache_dir is not None:
            models = stage_models(models, cache_dir)
    print(f"Project: {root}", flush=True)
    print(f"Model files: {models}", flush=True)
    subprocess.run(command_for(args, root, models), check=True)


if __name__ == "__main__":
    main()
