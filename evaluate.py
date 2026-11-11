"""Evaluate pretrained LifeMate models against the labelled train split."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Optional sample cap")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Save partial prediction JSONL after this many samples")
    parser.add_argument("--resume", action="store_true", help="Continue from matching checkpoints")
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.checkpoint_every < 1:
        parser.error("--limit and --checkpoint-every must be positive")

    split_dir = args.data_root / "train"
    for path in (
        split_dir / "train.jsonl",
        split_dir / "train_tools_catalog.json",
        split_dir / "zaic_tools" / "__init__.py",
    ):
        if not path.is_file():
            parser.error(f"Missing dataset file: {path}")

    asr = args.models_dir / "PhoWhisper-medium"
    llm = args.models_dir / "Qwen3-4B-Instruct-2507"
    for model in (asr, llm):
        if not (model / "config.json").is_file():
            parser.error(f"Missing model config: {model / 'config.json'}")

    runner = [
        sys.executable, "-u", str(Path(__file__).with_name("inference.py")),
        "--data-root", str(args.data_root), "--split", "train",
        "--checkpoint-every", str(args.checkpoint_every),
    ]
    suffix = ""
    if args.limit is not None:
        runner += ["--limit", str(args.limit)]
        suffix = f"-{args.limit}"
    if args.resume:
        runner.append("--resume")

    gold = args.runs_dir / f"gold-train{suffix}.jsonl"
    output = args.runs_dir / f"predictions-train{suffix}.jsonl"
    print("Replaying reference tool calls...", flush=True)
    subprocess.run(runner + ["--replay-gold", "--output", str(gold)], check=True)
    print("Loading ASR and Qwen, then evaluating audio requests...", flush=True)
    subprocess.run(runner + [
        "--asr", str(asr), "--llm", str(llm), "--gold-replay", str(gold),
        "--output", str(output),
    ], check=True)
    metrics_path = output.with_suffix(".metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(f"Samples: {metrics['samples']}")
    print(f"Exact tool-call sequence: {metrics['exact_ordered_tool_calls_rate']:.2%}")
    print(f"Final state match: {metrics['state_hash_match_rate']:.2%}")
    print(f"Mean latency: {metrics['latency_s']['mean']:.2f} seconds")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
