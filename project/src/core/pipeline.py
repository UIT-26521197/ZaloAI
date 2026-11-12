import json
import logging
from time import monotonic

from ..agent.planner import next_action
from ..agent.router import route_request
from ..environment.executor import ToolExecutor
from .types import ContextLimit, Result, Route

LOGGER = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config, models, registry, open_sample):
        self.config = config
        self.models = models
        self.registry = registry
        self.open_sample = open_sample

    def run(self, sample):
        started = monotonic()
        deadline = started + self.config.max_seconds
        executor = ToolExecutor(self.open_sample(sample.id))
        tools = self.registry.for_context(sample.context)
        transcript = self.models.whisper.transcribe(sample.audio)
        route = Route("general")
        status = "empty_transcript" if not transcript else "no_tools"
        if monotonic() >= deadline:
            status = "timeout"
        elif transcript and tools:
            route = route_request(self.models.laya, transcript, self.config.route_threshold)
            status = self._execute(sample, tools, executor, transcript, route, deadline)
        calls, state_hash = executor.snapshot()
        return Result(sample.id, calls, state_hash, transcript, route, status, monotonic() - started)

    def _execute(self, sample, tools, executor, transcript, route, deadline):
        observations = []
        seen = set()
        for _ in range(self.config.max_steps):
            seconds = deadline - monotonic()
            if seconds <= 0:
                return "timeout"
            try:
                call = next_action(
                    self.models.qwen, transcript, route, sample, tools, observations, seconds,
                )
            except ContextLimit:
                return "context_limit"
            except TimeoutError:
                return "timeout"
            except ValueError as exc:
                LOGGER.debug("%s: %s", sample.id, exc)
                observations.append({"error": "JSON hoặc công cụ không hợp lệ. Hãy sửa hoặc dừng."})
                continue
            if monotonic() >= deadline:
                return "timeout"
            if call is None:
                return "done"
            key = json.dumps(call.as_dict(), sort_keys=True, ensure_ascii=False)
            if key in seen:
                return "repeated_call"
            seen.add(key)
            observations.append({"call": call.as_dict(), **executor.execute(call)})
        return "timeout" if monotonic() >= deadline else "step_limit"
