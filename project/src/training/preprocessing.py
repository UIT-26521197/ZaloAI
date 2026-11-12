import json

from ..agent.prompt import messages_for
from ..agent.router import route_request
from ..agent.validator import decode_action
from ..environment.adapter import sample_from_row
from ..environment.executor import ToolExecutor


def examples(rows, config, registry, open_sample, whisper, laya):
    for row in rows:
        sample = sample_from_row(row, config.data_root, "train")
        label = row["label"]
        calls = label["tool_calls"]
        if not isinstance(calls, list):
            raise ValueError(f"{sample.id}: label.tool_calls must be a list")
        transcript = label.get("transcript") or row.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            transcript = whisper.transcribe(sample.audio)
        if not transcript:
            continue
        route = route_request(laya, transcript, config.route_threshold)
        tools = registry.for_context(sample.context)
        observations = []
        executor = ToolExecutor(open_sample(sample.id)) if calls else None
        for target in [*calls, {"stop": True}]:
            prompt = messages_for(transcript, route, sample.context, sample.history, tools, observations)
            completion = json.dumps(target, ensure_ascii=False)
            action = decode_action(completion, tools)
            if action is not None:
                observation = executor.execute(action)
                if "rejected" in observation:
                    raise ValueError(f"{sample.id}: gold call rejected: {action.name}")
                observations.append({"call": action.as_dict(), **observation})
            yield prompt, completion
