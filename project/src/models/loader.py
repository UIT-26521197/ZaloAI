from dataclasses import dataclass
from pathlib import Path
import os

from .laya import LayaModel
from .qwen import QwenModel
from .whisper import WhisperModel


def model_source(value: str, offline: bool) -> str:
    if value.startswith("hf:"):
        if offline:
            raise ValueError(f"Offline configuration requires a local directory: {value}")
        return value.removeprefix("hf:")
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory missing: {path}")
    return str(path)


@dataclass
class Models:
    whisper: WhisperModel
    laya: LayaModel
    qwen: QwenModel


def initialize_runtime(config):
    if config.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("USE_TF", "0")

    from transformers import set_seed

    set_seed(config.seed)


def load_models(config):
    whisper, laya, qwen = (
        model_source(getattr(config, name), config.offline) for name in ("whisper", "laya", "qwen")
    )
    adapter = model_source(config.qwen_adapter, True) if config.qwen_adapter else None
    initialize_runtime(config)
    return Models(
        WhisperModel(whisper, config.device, config.offline),
        LayaModel(laya, config.laya_device),
        QwenModel(qwen, config.device, config.offline, config.context_tokens, config.new_tokens, adapter),
    )
