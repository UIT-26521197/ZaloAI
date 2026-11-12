from ..core.types import Sample
from ..utils.audio import audio_path
from ..utils.paths import read_rows


def split_rows(root, split):
    return read_rows(root / split / f"{split}.jsonl")


def sample_from_row(row, root, split):
    sample_id = row["id"]
    if not isinstance(sample_id, str) or not sample_id or "/" in sample_id or "\\" in sample_id:
        raise ValueError(f"Invalid sample id: {sample_id!r}")
    return Sample(
        sample_id, audio_path(root, split, row["audio_path"]), row["context"],
        row.get("history") or row.get("conversation_history") or [],
    )


def catalog_path(root, split):
    return root / split / f"{split}_tools_catalog.json"
