from collections import Counter
from statistics import mean


def summarize(predictions, labels):
    status = Counter(row["status"] for row in predictions)
    routes = Counter(row["route"]["kind"] for row in predictions)
    call_matches, hash_matches = [], []
    for row in predictions:
        gold = labels.get(row["id"], {})
        if "tool_calls" in gold:
            call_matches.append(row["tool_calls"] == gold["tool_calls"])
        if gold.get("state_hash") is not None:
            hash_matches.append(row["state_hash"] == gold["state_hash"])
    return {
        "samples": len(predictions),
        "status": dict(status),
        "routes": dict(routes),
        "mean_seconds": mean(row["seconds"] for row in predictions) if predictions else None,
        "ordered_tool_calls_labeled": len(call_matches),
        "exact_ordered_tool_calls_rate": mean(call_matches) if call_matches else None,
        "state_hash_labeled": len(hash_matches),
        "state_hash_match_rate": mean(hash_matches) if hash_matches else None,
    }
