from itertools import islice
import json
import logging

from ..config import load_config
from ..core.pipeline import Pipeline
from ..core.registry import ToolRegistry
from ..environment.adapter import catalog_path, sample_from_row, split_rows
from ..environment.initializer import load_environment
from ..evaluation.report import write_report
from ..models.loader import load_models
from ..utils.logging import configure_logging
from ..utils.paths import atomic_text, json_line
from .submission import submission_row, trace_row

LOGGER = logging.getLogger(__name__)


def run(config):
    registry = ToolRegistry.load(catalog_path(config.data_root, config.split))
    source = split_rows(config.data_root, config.split)
    if config.ids:
        requested = json.loads(config.ids.read_text(encoding="utf-8-sig"))
        if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
            raise ValueError("--ids requires a JSON list of sample IDs")
        ids = set(requested)
        selected = [row for row in source if row["id"] in ids]
        missing = ids.difference(row["id"] for row in selected)
        if missing:
            raise ValueError(f"Sample IDs absent from {config.split}: {sorted(missing)[:10]}")
        source = iter(selected)
    rows = list(islice(source, config.limit))
    if not rows:
        raise ValueError(f"No rows in split {config.split}")
    samples = [sample_from_row(row, config.data_root, config.split) for row in rows]
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("Duplicate sample ids")
    open_sample = load_environment(config.data_root, config.split)
    pipeline = Pipeline(config, load_models(config), registry, open_sample)
    traces = []
    labels = {row["id"]: row.get("label") or {} for row in rows} if config.split == "train" else {}
    submission = config.output / "submission.jsonl"
    with atomic_text(submission) as out, atomic_text(config.output / "trace.jsonl") as trace:
        for index, sample in enumerate(samples, 1):
            result = pipeline.run(sample)
            json_line(out, submission_row(result))
            traces.append(trace_row(result))
            json_line(trace, traces[-1])
            if index % 25 == 0 or index == len(samples):
                LOGGER.info("%s %s/%s", config.split, index, len(samples))
    write_report(config.output, traces, labels)
    LOGGER.info("Submission: %s", submission)
    return submission


def main(argv=None):
    configure_logging()
    return run(load_config("inference", argv))


if __name__ == "__main__":
    main()
