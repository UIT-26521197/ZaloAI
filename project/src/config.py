from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import os

PROJECT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    mode: str
    split: str
    data_root: Path
    output: Path
    whisper: str
    laya: str
    qwen: str
    qwen_adapter: str | None
    device: str
    laya_device: str
    offline: bool
    seed: int
    route_threshold: float
    max_steps: int
    max_seconds: float
    context_tokens: int
    new_tokens: int
    limit: int | None
    ids: Path | None = None
    validation_fraction: float = 0.1
    epochs: float = 1.0


def load_config(mode: str, argv=None) -> Config:
    parser = ArgumentParser(description=f"LifeMate {mode}")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / f"{mode}.yaml")
    parser.add_argument("--workspace", type=Path, default=os.environ.get("ZAI_ROOT", PROJECT.parent))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=["train", "public_test", "private_test"])
    parser.add_argument("--limit", type=int)
    if mode != "training":
        parser.add_argument("--ids", type=Path, help="JSON list of sample IDs")
    args = parser.parse_args(argv)

    import yaml

    workspace = args.workspace.resolve()
    with args.config.open(encoding="utf-8") as source:
        settings = yaml.safe_load(source)
    default_data = workspace / settings.pop("data_root")
    default_output = (workspace if mode == "training" else PROJECT) / settings.pop("output")
    data = args.data_root or default_data
    output = args.output or default_output
    for key in ("whisper", "laya", "qwen", "qwen_adapter"):
        value = settings.get(key)
        if value and not value.startswith("hf:"):
            settings[key] = str(workspace / value)
    split = args.split or settings.pop("split")
    settings.pop("split", None)
    if mode in {"baseline", "training"} and split != "train":
        parser.error(f"{mode} requires the labeled train split")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = Config(
        mode, split, data.resolve(), output.resolve(),
        limit=args.limit, ids=getattr(args, "ids", None), **settings,
    )
    if not 0 < config.new_tokens < config.context_tokens <= 16384:
        parser.error("require 0 < new_tokens < context_tokens <= 16384")
    if not 0 < config.max_seconds <= 30 or config.max_steps < 1:
        parser.error("require 0 < max_seconds <= 30 and max_steps >= 1")
    if not 0 <= config.route_threshold <= 1 or not 0 < config.validation_fraction < 1:
        parser.error("invalid route_threshold or validation_fraction")
    if config.epochs <= 0:
        parser.error("epochs must be positive")
    if mode == "training" and config.qwen_adapter:
        parser.error("training starts from the configured base Qwen checkpoint")
    return config
