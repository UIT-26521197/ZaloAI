from dataclasses import asdict


def submission_row(result):
    return {"id": result.id, "tool_calls": result.tool_calls, "state_hash": result.state_hash}


def trace_row(result):
    return asdict(result)
