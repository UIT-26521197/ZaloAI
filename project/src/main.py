from argparse import ArgumentParser

from .baseline.run import main as baseline
from .inference.run import main as inference
from .training.trainer import main as training


def main():
    parser = ArgumentParser(description="LifeMate pipeline")
    commands = {"baseline": baseline, "training": training, "inference": inference}
    parser.add_argument("mode", choices=commands)
    args, remaining = parser.parse_known_args()
    commands[args.mode](remaining)


if __name__ == "__main__":
    main()
