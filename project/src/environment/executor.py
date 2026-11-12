class ToolExecutor:
    def __init__(self, env):
        self.env = env

    def execute(self, call):
        verdict = self.env.validate(call.name, call.arguments)
        if isinstance(verdict, dict):
            valid = verdict.get("ok") is True
        elif isinstance(verdict, bool):
            valid = verdict
        else:
            raise TypeError("Unsupported ToolEnv.validate response")
        if not valid:
            return {"rejected": verdict}
        return {"result": self.env.call(call.name, call.arguments)}

    def snapshot(self):
        return self.env.get_tool_calls(), self.env.get_state_hash()
