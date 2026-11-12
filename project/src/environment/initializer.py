import sys
from importlib import import_module


def load_environment(data_root, split):
    for parent in (data_root / split, data_root / "train", data_root):
        if (parent / "zaic_tools").is_dir():
            sys.path.insert(0, str(parent))
            break
    try:
        tool_env = import_module("zaic_tools").ToolEnv
    except ModuleNotFoundError as exc:
        if exc.name != "zaic_tools":
            raise
        raise RuntimeError(f"Cannot locate the official zaic_tools package under {data_root}") from exc

    def open_sample(sample_id):
        return tool_env(split=split, sample_id=sample_id, root=str(data_root))

    return open_sample
