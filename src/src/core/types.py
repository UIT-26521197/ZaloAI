from dataclasses import dataclass
from pathlib import Path


class ContextLimit(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    id: str
    audio: Path
    context: dict
    history: list


@dataclass(frozen=True)
class Route:
    kind: str
    confidence: float | None = None


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict

    def as_dict(self):
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class Result:
    id: str
    tool_calls: list
    state_hash: str
    transcript: str
    route: Route
    status: str
    seconds: float
