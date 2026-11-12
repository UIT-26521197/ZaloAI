import json

from .error_analysis import error_tags
from .metrics import summarize
from ..utils.paths import atomic_text, json_line


def write_report(directory, predictions, labels):
    with atomic_text(directory / "metrics.json") as stream:
        json.dump(summarize(predictions, labels), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with atomic_text(directory / "errors.jsonl") as stream:
        for row in predictions:
            tags = error_tags(row, labels.get(row["id"], {}))
            if tags:
                json_line(stream, {"id": row["id"], "route": row["route"], "errors": tags})
