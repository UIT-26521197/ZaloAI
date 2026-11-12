def error_tags(prediction, label):
    tags = []
    if prediction["status"] != "done":
        tags.append(prediction["status"])
    if "tool_calls" in label and prediction["tool_calls"] != label["tool_calls"]:
        tags.append("ordered_calls_mismatch")
    if label.get("state_hash") is not None and prediction["state_hash"] != label["state_hash"]:
        tags.append("state_hash_mismatch")
    return tags
