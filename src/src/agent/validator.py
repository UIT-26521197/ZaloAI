import json

from ..core.types import ToolCall


def _invalid_constant(value):
    raise ValueError(f"Invalid JSON constant {value}")


def decode_action(text: str, tools: dict) -> ToolCall | None:
    raw = text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.partition("\n")[2][:-3].strip()
    action = json.loads(raw, parse_constant=_invalid_constant)
    if action == {"stop": True} and action["stop"] is True:
        return None
    if not isinstance(action, dict) or action.keys() != {"name", "arguments"}:
        raise ValueError("Expected one tool call or a stop object")
    name, arguments = action["name"], action["arguments"]
    if not isinstance(name, str) or name not in tools or not isinstance(arguments, dict):
        raise ValueError("Tool unavailable or arguments are not an object")
    return ToolCall(name, arguments)
