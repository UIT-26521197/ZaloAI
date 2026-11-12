from itertools import islice
from random import Random

from ..environment.adapter import split_rows


def partition(config):
    rows = list(islice(split_rows(config.data_root, "train"), config.limit))
    groups = {}
    ids = set()
    for row in rows:
        sample_id = row["id"]
        if sample_id in ids:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        ids.add(sample_id)
        groups.setdefault(row.get("conversation_id") or sample_id, []).append(row)
    if len(groups) < 2:
        raise ValueError("Training requires at least two distinct samples or conversations")
    keys = sorted(groups, key=str)
    Random(config.seed).shuffle(keys)
    count = max(1, min(len(keys) - 1, round(len(keys) * config.validation_fraction)))
    return (
        [row for key in keys[count:] for row in groups[key]],
        [row for key in keys[:count] for row in groups[key]],
    )
