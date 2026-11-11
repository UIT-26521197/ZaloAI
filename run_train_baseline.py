"""Replay train labels and evaluate the LifeMate baseline on the same samples."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Optional sample cap")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    asr = args.models_dir / "PhoWhisper-medium"
    llm = args.models_dir / "Qwen3-4B-Instruct-2507"
    for model in (asr, llm):
        if not (model / "config.json").is_file():
            parser.error(f"Missing model config: {model / 'config.json'}")

    runner = [
        sys.executable, str(Path(__file__).with_name("lifemate_baseline.py")),
        "--data-root", str(args.data_root), "--split", "train",
    ]
    suffix = ""
    if args.limit is not None:
        runner += ["--limit", str(args.limit)]
        suffix = f"-{args.limit}"

    gold = args.runs_dir / f"gold-train{suffix}.jsonl"
    output = args.runs_dir / f"baseline-train{suffix}.jsonl"
    subprocess.run(runner + ["--replay-gold", "--output", str(gold)], check=True)
    subprocess.run(runner + [
        "--asr", str(asr), "--llm", str(llm), "--gold-replay", str(gold),
        "--output", str(output),
    ], check=True)
    print(f"Metrics: {output.with_suffix('.metrics.json')}")


if __name__ == "__main__":
    main()
