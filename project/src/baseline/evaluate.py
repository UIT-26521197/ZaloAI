from argparse import ArgumentParser
from pathlib import Path

from ..environment.adapter import split_rows
from ..evaluation.report import write_report
from ..utils.paths import read_rows


def main():
    parser = ArgumentParser(description="Evaluate an existing baseline trace against train labels")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = {row["id"]: row.get("label") or {} for row in split_rows(args.data_root, "train")}
    write_report(args.output, list(read_rows(args.trace)), labels)


if __name__ == "__main__":
    main()
